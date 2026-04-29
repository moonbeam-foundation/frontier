// This file is part of Frontier.
//
// Copyright (C) Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: GPL-3.0-or-later WITH Classpath-exception-2.0

//! RAII-backed registry of block-notification sinks (replaces append-only `Vec` of senders).

use std::{
	collections::HashMap,
	sync::{Arc, Weak},
};

use futures::channel::mpsc;
use parking_lot::Mutex;

struct Inner<T> {
	sinks: HashMap<u64, mpsc::Sender<T>>,
	next_id: u64,
	/// Bumped when clearing all sinks on major sync; stale [`SinkGuard`]s must not remove
	/// unrelated registrations after IDs advance.
	generation: u64,
}

fn maybe_shrink_sink_map<T>(sinks: &mut HashMap<u64, mpsc::Sender<T>>) {
	let len = sinks.len();
	let cap = sinks.capacity();
	if cap > 4096 && cap > len.saturating_mul(4).max(1) {
		sinks.shrink_to_fit();
	}
}

/// Aggregated stats for [`SinkRegistry`].
///
/// Bounded per-sink `mpsc` channels do not expose queue depth on the sender; use `sinks` and
/// `capacity` for pressure signals. Drops on full channels are visible indirectly (e.g. fewer
/// sinks after `broadcast` prunes lagging receivers).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SinkRegistryStats {
	pub sinks: usize,
	pub capacity: usize,
	pub closed: usize,
}

/// Shared registry of async sinks with RAII-based removal.
///
/// Each [`Self::register`] returns a [`SinkGuard`]; when dropped, the sink is removed.
/// [`Self::broadcast`] also removes closed sinks, sinks whose bounded channel is full
/// (`try_send` returns [`TrySendError::Full`]), and sinks that reject the send.
pub struct SinkRegistry<T> {
	inner: Mutex<Inner<T>>,
}

impl<T> Default for SinkRegistry<T> {
	fn default() -> Self {
		Self::new()
	}
}

impl<T> SinkRegistry<T> {
	pub fn new() -> Self {
		Self {
			inner: Mutex::new(Inner {
				sinks: HashMap::new(),
				next_id: 0,
				generation: 0,
			}),
		}
	}

	/// Current number of registered sinks (for metrics / diagnostics).
	pub fn len(&self) -> usize {
		self.inner.lock().sinks.len()
	}

	pub fn is_empty(&self) -> bool {
		self.inner.lock().sinks.is_empty()
	}

	/// Map size, bucket capacity, and closed senders (for Prometheus / leak hunting).
	pub fn stats(&self) -> SinkRegistryStats {
		let inner = self.inner.lock();
		let mut stats = SinkRegistryStats {
			sinks: inner.sinks.len(),
			capacity: inner.sinks.capacity(),
			closed: 0,
		};
		for sink in inner.sinks.values() {
			if sink.is_closed() {
				stats.closed = stats.closed.saturating_add(1);
			}
		}
		stats
	}

	/// Remove all sinks (used when the node enters major sync).
	pub fn clear_on_major_sync(&self) {
		let mut inner = self.inner.lock();
		inner.generation = inner.generation.saturating_add(1);
		inner.sinks.clear();
		inner.sinks.shrink_to_fit();
	}

	/// Register a new sink with a bounded per-sink queue of `channel_capacity` messages.
	///
	/// Keep [`SinkGuard`] alive while the paired [`mpsc::Receiver`] is in use; dropping it
	/// unregisters the sender.
	pub fn register(
		self: &Arc<Self>,
		channel_capacity: usize,
	) -> (SinkGuard<T>, mpsc::Receiver<T>)
	where
		T: Send + 'static,
	{
		let capacity = channel_capacity.max(1);
		let (sender, receiver) = mpsc::channel(capacity);
		let mut inner = self.inner.lock();
		let id = inner.next_id;
		inner.next_id = inner.next_id.saturating_add(1);
		let generation = inner.generation;
		inner.sinks.insert(id, sender);
		(
			SinkGuard {
				registry: Arc::downgrade(self),
				id,
				generation,
			},
			receiver,
		)
	}

	/// Fan-out one notification per live sink, cloning via `make_msg` for each send attempt.
	///
	/// `max_pending` is kept for call-site compatibility; per-sink capacity is fixed at
	/// [`Self::register`] time.
	pub fn broadcast(&self, _max_pending: usize, mut make_msg: impl FnMut() -> T)
	where
		T: Send,
	{
		let mut inner = self.inner.lock();
		if inner.sinks.is_empty() {
			return;
		}

		inner.sinks.retain(|_id, sink| {
			if sink.is_closed() {
				return false;
			}
			match sink.try_send(make_msg()) {
				Ok(()) => true,
				Err(e) => {
					if e.is_full() {
						log::debug!(
							target: "mapping-sync",
							"Dropping lagging pubsub sink (bounded notification channel full)",
						);
					}
					false
				}
			}
		});
		maybe_shrink_sink_map(&mut inner.sinks);
	}
}

/// When dropped, removes the corresponding sink from its [`SinkRegistry`].
pub struct SinkGuard<T> {
	registry: Weak<SinkRegistry<T>>,
	id: u64,
	generation: u64,
}

impl<T> Drop for SinkGuard<T> {
	fn drop(&mut self) {
		let Some(registry) = self.registry.upgrade() else {
			return;
		};
		let mut inner = registry.inner.lock();
		if inner.generation != self.generation {
			return;
		}
		let _ = inner.sinks.remove(&self.id);
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use futures::{executor::block_on, StreamExt as _};

	#[test]
	fn guard_drop_removes_sink() {
		let registry = Arc::new(SinkRegistry::<u32>::new());
		let (guard, _rx) = registry.register(64);
		assert_eq!(registry.len(), 1);
		drop(guard);
		assert_eq!(registry.len(), 0);
	}

	#[test]
	fn broadcast_prunes_closed_receiver() {
		let registry = Arc::new(SinkRegistry::<u32>::new());
		let (_g1, mut rx1) = registry.register(64);
		let (_g2, rx2) = registry.register(64);
		drop(rx2);
		registry.broadcast(512, || 7);
		assert_eq!(registry.len(), 1);
		assert_eq!(block_on(rx1.next()), Some(7));
	}

	#[test]
	fn stats_reports_sink_count() {
		let registry = Arc::new(SinkRegistry::<u32>::new());
		let (_g1, _rx1) = registry.register(64);
		let (_g2, _rx2) = registry.register(64);
		registry.broadcast(512, || 1);
		registry.broadcast(512, || 2);
		let s = registry.stats();
		assert_eq!(s.sinks, 2);
		assert!(s.capacity >= 2);
	}

	#[test]
	fn clear_on_major_sync_invalidates_old_guards() {
		let registry = Arc::new(SinkRegistry::<u32>::new());
		let (g, _rx) = registry.register(64);
		assert_eq!(registry.len(), 1);
		registry.clear_on_major_sync();
		assert_eq!(registry.len(), 0);
		drop(g);
		assert_eq!(registry.len(), 0);
		let (_g2, mut rx2) = registry.register(64);
		assert_eq!(registry.len(), 1);
		registry.broadcast(512, || 42);
		assert_eq!(block_on(rx2.next()), Some(42));
	}
}
