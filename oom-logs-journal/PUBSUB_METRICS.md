# Frontier pubsub / logs journal Prometheus metrics

These metrics must be registered in a new [`src/pubsub_metrics.rs`](src/pubsub_metrics.rs) when `spawn_frontier_pubsub_metrics_task` runs with a non-`None` Prometheus registry. They are all **gauges** (`U64`) and are refreshed approximately **every 5 seconds** by the background task.

There are no Prometheus gauges for per-sink `mpsc` queue depth: bounded `futures` senders do not expose pending counts (see [`SinkRegistry::stats`](../mapping-sync/src/sink_registry.rs)).

| Metric name | Description |
|-------------|-------------|
| `frontier_pubsub_sink_registry_len` | Number of registered Ethereum block pubsub notification sinks |
| `frontier_pubsub_sink_registry_capacity` | HashMap bucket capacity for the block pubsub sink registry (can exceed len after churn) |
| `frontier_pubsub_sink_closed` | Number of closed senders still present before next broadcast prune (usually zero) |
| `frontier_mapping_sync_best_at_import_entries` | Mapping sync KV worker best_at_import map size (zero on SQL-only backend) |
| `frontier_mapping_sync_best_at_import_reorg_items` | Sum of retracted+enacted counts retained in best_at_import reorg payloads |
| `frontier_mapping_sync_current_syncing_tips_len` | Current frontier current_syncing_tips length observed by mapping-sync |
| `frontier_mapping_sync_current_syncing_tips_duplicates` | Duplicate hashes in current_syncing_tips (len - unique) |
| `frontier_mapping_sync_current_syncing_tips_len_peak` | Peak current_syncing_tips length observed since process start |
| `frontier_mapping_sync_current_syncing_tips_nonzero_samples_total` | Number of metric samples where current_syncing_tips length was non-zero |
| `frontier_mapping_sync_current_syncing_tips_len_last_nonzero` | Last observed non-zero current_syncing_tips length |
| `frontier_mapping_sync_best_at_import_cap_evictions_total` | Total entries evicted because best_at_import exceeded its hard cap |
| `frontier_reconcile_transactions_scanned_total` | Total Ethereum transactions scanned during canonical reconciler passes |
| `frontier_reconcile_tx_metadata_lookups_total` | Total transaction metadata lookups during canonical reconciler passes |
| `frontier_reconcile_scanned_total` | Total block numbers scanned across canonical reconciler passes |
| `frontier_reconcile_updated_total` | Total mapping updates applied by canonical reconciler passes |
| `frontier_mapping_sync_one_block_duration_seconds` | Wall-clock duration of the latest mapping-sync step (seconds, integer) |
| `frontier_logs_journal_entries_total_bytes` | Total retained bytes in the logs journal entry deque |
| `frontier_logs_journal_broadcast_lag_max` | Maximum observed logs journal broadcast lag (skipped messages) for any subscriber |

The pubsub task always reads mapping-sync / reconcile values from the `Arc<MappingSyncMetrics>` you pass in; they stay at **zero** until the KV [`MappingSyncWorker`](../../mapping-sync/src/kv/worker.rs) updates that same `Arc` (for example on a SQL-only Frontier backend there is no KV worker). Pubsub sink and logs journal gauges still reflect live data.
