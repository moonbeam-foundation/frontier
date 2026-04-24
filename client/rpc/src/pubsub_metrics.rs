// This file is part of Frontier.
//
// Copyright (C) Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: GPL-3.0-or-later WITH Classpath-exception-2.0

//! Prometheus gauges for Ethereum pubsub / logs journal introspection.

use std::{sync::Arc, time::Duration};

use futures::FutureExt as _;
use prometheus_endpoint as pe;
use sc_rpc::SubscriptionTaskExecutor;
use sp_runtime::traits::Block as BlockT;

use fc_mapping_sync::EthereumBlockNotificationSinks;

use crate::LogsJournal;

type PubsubGaugeU64 = pe::Gauge<pe::U64>;
type RegisteredPubsubGauges = (PubsubGaugeU64, PubsubGaugeU64, PubsubGaugeU64);

/// Registers gauges and spawns a task that updates them every 5 seconds.
///
/// No-op when `registry` is `None`.
pub fn spawn_frontier_pubsub_metrics_task<B: BlockT>(
	registry: Option<&pe::Registry>,
	pubsub_notification_sinks: Arc<EthereumBlockNotificationSinks<B>>,
	logs_journal: Arc<LogsJournal>,
	executor: SubscriptionTaskExecutor,
) {
	let Some(registry) = registry else {
		return;
	};

	let (sink_registry_len, journal_entries_total_bytes, broadcast_lag_max) =
		match register_pubsub_gauges(registry) {
			Ok(t) => t,
			Err(e) => {
				log::error!(target: "rpc", "Failed to register Frontier pubsub metrics: {e:?}");
				return;
			}
		};

	executor.spawn(
		"frontier-pubsub-metrics",
		Some("rpc"),
		async move {
			let mut ticker = tokio::time::interval(Duration::from_secs(5));
			loop {
				ticker.tick().await;
				sink_registry_len.set(pubsub_notification_sinks.len() as u64);
				let bytes: u64 = logs_journal
					.total_retained_bytes()
					.try_into()
					.unwrap_or(u64::MAX);
				journal_entries_total_bytes.set(bytes);
				broadcast_lag_max.set(logs_journal.max_logs_broadcast_lag());
			}
		}
		.boxed(),
	);
}

fn register_pubsub_gauges(
	registry: &pe::Registry,
) -> Result<RegisteredPubsubGauges, pe::PrometheusError> {
	// `Gauge::new` returns `Result` (naming); `register` takes the constructed collector.
	let sink_registry_len = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_pubsub_sink_registry_len",
			"Number of registered Ethereum block pubsub notification sinks",
		)?,
		registry,
	)?;
	let journal_entries_total_bytes = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_logs_journal_entries_total_bytes",
			"Total retained bytes in the logs journal entry deque",
		)?,
		registry,
	)?;
	let broadcast_lag_max = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_logs_journal_broadcast_lag_max",
			"Maximum observed logs journal broadcast lag (skipped messages) for any subscriber",
		)?,
		registry,
	)?;
	Ok((
		sink_registry_len,
		journal_entries_total_bytes,
		broadcast_lag_max,
	))
}
