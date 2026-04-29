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

use fc_mapping_sync::{EthereumBlockNotificationSinks, MappingSyncMetrics};

use crate::LogsJournal;

type PubsubGaugeU64 = pe::Gauge<pe::U64>;
type RegisteredPubsubGauges = (
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
	PubsubGaugeU64,
);

/// Registers gauges and spawns a task that updates them every 5 seconds.
///
/// No-op when `registry` is `None`.
///
/// When the node builds the RPC module more than once against the same Prometheus registry
/// (e.g. multiple transports), registration can hit [`pe::PrometheusError::AlreadyReg`]. That is
/// treated as success: metrics already exist and only one updater task should run.
pub fn spawn_frontier_pubsub_metrics_task<B: BlockT>(
	registry: Option<&pe::Registry>,
	pubsub_notification_sinks: Arc<EthereumBlockNotificationSinks<B>>,
	logs_journal: Arc<LogsJournal>,
	executor: SubscriptionTaskExecutor,
	mapping_sync_metrics: Option<Arc<MappingSyncMetrics>>,
) {
	let Some(registry) = registry else {
		return;
	};

	let (
		sink_registry_len,
		sink_registry_capacity,
		sink_closed,
		best_at_import,
		best_at_import_reorg_items,
		current_syncing_tips_len,
		current_syncing_tips_duplicates,
		current_syncing_tips_len_peak,
		current_syncing_tips_nonzero_samples_total,
		current_syncing_tips_len_last_nonzero,
		best_at_import_cap_evictions_total,
		reconcile_transactions_scanned_total,
		reconcile_tx_metadata_lookups_total,
		reconcile_scanned_total,
		reconcile_updated_total,
		one_block_duration_seconds,
		journal_entries_total_bytes,
		broadcast_lag_max,
	) = match register_pubsub_gauges(registry) {
		Ok(t) => t,
		Err(pe::PrometheusError::AlreadyReg) => {
			log::debug!(
				target: "rpc",
				"Frontier pubsub metrics already registered; skipping duplicate setup",
			);
			return;
		}
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
				let s = pubsub_notification_sinks.stats();
				sink_registry_capacity.set(s.capacity as u64);
				sink_closed.set(s.closed as u64);
				if let Some(m) = &mapping_sync_metrics {
					best_at_import.set(
						m.best_at_import_entries
							.load(std::sync::atomic::Ordering::Relaxed) as u64,
					);
					best_at_import_reorg_items.set(
						m.best_at_import_reorg_items
							.load(std::sync::atomic::Ordering::Relaxed) as u64,
					);
					current_syncing_tips_len.set(
						m.current_syncing_tips_len
							.load(std::sync::atomic::Ordering::Relaxed) as u64,
					);
					current_syncing_tips_duplicates.set(
						m.current_syncing_tips_duplicates
							.load(std::sync::atomic::Ordering::Relaxed) as u64,
					);
					current_syncing_tips_len_peak.set(
						m.current_syncing_tips_len_peak
							.load(std::sync::atomic::Ordering::Relaxed) as u64,
					);
					current_syncing_tips_nonzero_samples_total.set(
						m.current_syncing_tips_nonzero_samples_total
							.load(std::sync::atomic::Ordering::Relaxed),
					);
					current_syncing_tips_len_last_nonzero.set(
						m.current_syncing_tips_len_last_nonzero
							.load(std::sync::atomic::Ordering::Relaxed) as u64,
					);
					best_at_import_cap_evictions_total.set(
						m.best_at_import_cap_evictions_total
							.load(std::sync::atomic::Ordering::Relaxed),
					);
					reconcile_transactions_scanned_total.set(
						m.reconcile_transactions_scanned_total
							.load(std::sync::atomic::Ordering::Relaxed),
					);
					reconcile_tx_metadata_lookups_total.set(
						m.reconcile_tx_metadata_lookups_total
							.load(std::sync::atomic::Ordering::Relaxed),
					);
					reconcile_scanned_total.set(
						m.reconcile_scanned_total
							.load(std::sync::atomic::Ordering::Relaxed),
					);
					reconcile_updated_total.set(
						m.reconcile_updated_total
							.load(std::sync::atomic::Ordering::Relaxed),
					);
					one_block_duration_seconds.set(
						m.one_block_duration_micros
							.load(std::sync::atomic::Ordering::Relaxed)
							/ 1_000_000,
					);
				} else {
					best_at_import.set(0);
					best_at_import_reorg_items.set(0);
					current_syncing_tips_len.set(0);
					current_syncing_tips_duplicates.set(0);
					current_syncing_tips_len_peak.set(0);
					current_syncing_tips_nonzero_samples_total.set(0);
					current_syncing_tips_len_last_nonzero.set(0);
					best_at_import_cap_evictions_total.set(0);
					reconcile_transactions_scanned_total.set(0);
					reconcile_tx_metadata_lookups_total.set(0);
					reconcile_scanned_total.set(0);
					reconcile_updated_total.set(0);
					one_block_duration_seconds.set(0);
				}
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
	let sink_registry_capacity = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_pubsub_sink_registry_capacity",
			"HashMap bucket capacity for the block pubsub sink registry (can exceed len after churn)",
		)?,
		registry,
	)?;
	let sink_closed = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_pubsub_sink_closed",
			"Number of closed senders still present before next broadcast prune (usually zero)",
		)?,
		registry,
	)?;
	let best_at_import = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_mapping_sync_best_at_import_entries",
			"Mapping sync KV worker best_at_import map size (zero on SQL-only backend)",
		)?,
		registry,
	)?;
	let best_at_import_reorg_items = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_mapping_sync_best_at_import_reorg_items",
			"Sum of retracted+enacted counts retained in best_at_import reorg payloads",
		)?,
		registry,
	)?;
	let current_syncing_tips_len = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_mapping_sync_current_syncing_tips_len",
			"Current frontier current_syncing_tips length observed by mapping-sync",
		)?,
		registry,
	)?;
	let current_syncing_tips_duplicates = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_mapping_sync_current_syncing_tips_duplicates",
			"Duplicate hashes in current_syncing_tips (len - unique)",
		)?,
		registry,
	)?;
	let current_syncing_tips_len_peak = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_mapping_sync_current_syncing_tips_len_peak",
			"Peak current_syncing_tips length observed since process start",
		)?,
		registry,
	)?;
	let current_syncing_tips_nonzero_samples_total = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_mapping_sync_current_syncing_tips_nonzero_samples_total",
			"Number of metric samples where current_syncing_tips length was non-zero",
		)?,
		registry,
	)?;
	let current_syncing_tips_len_last_nonzero = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_mapping_sync_current_syncing_tips_len_last_nonzero",
			"Last observed non-zero current_syncing_tips length",
		)?,
		registry,
	)?;
	let best_at_import_cap_evictions_total = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_mapping_sync_best_at_import_cap_evictions_total",
			"Total entries evicted because best_at_import exceeded its hard cap",
		)?,
		registry,
	)?;
	let reconcile_transactions_scanned_total = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_reconcile_transactions_scanned_total",
			"Total Ethereum transactions scanned during canonical reconciler passes",
		)?,
		registry,
	)?;
	let reconcile_tx_metadata_lookups_total = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_reconcile_tx_metadata_lookups_total",
			"Total transaction metadata lookups during canonical reconciler passes",
		)?,
		registry,
	)?;
	let reconcile_scanned_total = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_reconcile_scanned_total",
			"Total block numbers scanned across canonical reconciler passes",
		)?,
		registry,
	)?;
	let reconcile_updated_total = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_reconcile_updated_total",
			"Total mapping updates applied by canonical reconciler passes",
		)?,
		registry,
	)?;
	let one_block_duration_seconds = pe::register(
		pe::Gauge::<pe::U64>::new(
			"frontier_mapping_sync_one_block_duration_seconds",
			"Wall-clock duration of the latest mapping-sync step (seconds, integer)",
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
		sink_registry_capacity,
		sink_closed,
		best_at_import,
		best_at_import_reorg_items,
		current_syncing_tips_len,
		current_syncing_tips_duplicates,
		current_syncing_tips_len_peak,
		current_syncing_tips_nonzero_samples_total,
		current_syncing_tips_len_last_nonzero,
		best_at_import_cap_evictions_total,
		reconcile_transactions_scanned_total,
		reconcile_tx_metadata_lookups_total,
		reconcile_scanned_total,
		reconcile_updated_total,
		one_block_duration_seconds,
		journal_entries_total_bytes,
		broadcast_lag_max,
	))
}
