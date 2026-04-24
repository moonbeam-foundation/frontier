// This file is part of Frontier.
//
// Copyright (C) Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: GPL-3.0-or-later WITH Classpath-exception-2.0

//! RAII-backed registry of block-notification sinks (replaces append-only `Vec` of senders).

use std::{
	collections::HashMap,
	sync::{Arc, Weak},
};

use parking_lot::Mutex;
use sc_utils::mpsc::{tracing_unbounded, TracingUnboundedReceiver, TracingUnboundedSender};

struct Inner<T> {
	sinks: HashMap<u64, TracingUnboundedSender<T>>,
	next_id: u64,
	/// Bumped when clearing all sinks on major sync; stale [`SinkGuard`]s must not remove
	/// unrelated registrations after IDs advance.
	generation: u64,
}

/// Shared registry of async sinks with RAII-based removal.
///
/// Each [`Self::register`] returns a [`SinkGuard`]; when dropped, the sink is removed.
/// [`Self::broadcast`] also removes closed or lagging sinks (same semantics as the former
/// `Vec::retain` loop in [`crate::emit_block_notification`]).
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

	/// Remove all sinks (used when the node enters major sync).
	pub fn clear_on_major_sync(&self) {
		let mut inner = self.inner.lock();
		inner.generation = inner.generation.saturating_add(1);
		inner.sinks.clear();
	}

	/// Register a new sink. Keep [`SinkGuard`] alive while the paired receiver is in use;
	/// dropping it unregisters the sender.
	pub fn register(
		self: &Arc<Self>,
		name: &'static str,
		warn_threshold: usize,
	) -> (SinkGuard<T>, TracingUnboundedReceiver<T>)
	where
		T: Send + 'static,
	{
		let (sender, receiver) = tracing_unbounded(name, warn_threshold);
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

	/// Fan-out one notification per live sink, cloning via `make_msg` for each successful
	/// delivery attempt. Drops closed channels, lagging channels (`len >= max_pending`), and
	/// channels that reject the send — mirroring the previous `Mutex<Vec<_>>::retain` behavior.
	pub fn broadcast(&self, max_pending: usize, mut make_msg: impl FnMut() -> T)
	where
		T: Clone,
	{
		let mut inner = self.inner.lock();
		if inner.sinks.is_empty() {
			return;
		}

		let mut to_remove = Vec::new();
		for (id, sink) in inner.sinks.iter() {
			let id = *id;
			if sink.is_closed() {
				to_remove.push(id);
				continue;
			}
			if sink.len() >= max_pending {
				log::debug!(
					target: "mapping-sync",
					"Dropping lagging pubsub subscriber (pending={}, max={})",
					sink.len(),
					max_pending,
				);
				let _ = sink.close();
				to_remove.push(id);
				continue;
			}
			if sink.unbounded_send(make_msg()).is_err() {
				to_remove.push(id);
			}
		}
		for id in to_remove {
			inner.sinks.remove(&id);
		}
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
		let (guard, _rx) = registry.register("test", 10_000);
		assert_eq!(registry.len(), 1);
		drop(guard);
		assert_eq!(registry.len(), 0);
	}

	#[test]
	fn broadcast_prunes_closed_receiver() {
		let registry = Arc::new(SinkRegistry::<u32>::new());
		let (_g1, mut rx1) = registry.register("a", 10_000);
		let (_g2, rx2) = registry.register("b", 10_000);
		drop(rx2);
		registry.broadcast(512, || 7);
		assert_eq!(registry.len(), 1);
		assert_eq!(block_on(rx1.next()), Some(7));
	}

	#[test]
	fn clear_on_major_sync_invalidates_old_guards() {
		let registry = Arc::new(SinkRegistry::<u32>::new());
		let (g, _rx) = registry.register("t", 10_000);
		assert_eq!(registry.len(), 1);
		registry.clear_on_major_sync();
		assert_eq!(registry.len(), 0);
		drop(g);
		assert_eq!(registry.len(), 0);
		let (_g2, mut rx2) = registry.register("t2", 10_000);
		assert_eq!(registry.len(), 1);
		registry.broadcast(512, || 42);
		assert_eq!(block_on(rx2.next()), Some(42));
	}
}
