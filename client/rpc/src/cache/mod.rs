// This file is part of Frontier.

// Copyright (C) Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: GPL-3.0-or-later WITH Classpath-exception-2.0

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

pub mod lru_cache;

use std::{
	collections::{BTreeMap, HashMap},
	marker::PhantomData,
	sync::{Arc, Mutex},
};

use ethereum::BlockV3 as EthereumBlock;
use ethereum_types::U256;
use futures::StreamExt;
use scale_codec::Encode;
use tokio::sync::{mpsc, oneshot};
// Substrate
use sc_client_api::{
	backend::{Backend, StorageProvider},
	client::BlockchainEvents,
};
use sc_service::SpawnTaskHandle;
use sp_api::ProvideRuntimeApi;
use sp_blockchain::HeaderBackend;
use sp_runtime::traits::{Block as BlockT, Header as HeaderT, UniqueSaturatedInto};
// Frontier
use fc_rpc_core::types::*;
use fc_storage::StorageOverride;
use fp_rpc::{EthereumRuntimeRPCApi, TransactionStatus};
use prometheus::{HistogramOpts, HistogramVec, IntCounterVec, Opts};

use self::lru_cache::LRUCacheByteLimited;

/// Internal bounded queue between RPC tasks and the cache driver. Larger values
/// reduce `send().await` stalls when many concurrent requests hit the cache.
const CACHE_TASK_CHANNEL_CAPACITY: usize = 512;

type BlockWaitList<B> = HashMap<<B as BlockT>::Hash, Vec<oneshot::Sender<Option<Arc<EthereumBlock>>>>>;
type StatusesWaitList<B> = HashMap<
	<B as BlockT>::Hash,
	Vec<oneshot::Sender<Option<Arc<Vec<TransactionStatus>>>>>,
>;

/// Wrapper so the LRU byte budget uses the encoded size of the payload while
/// storing a cheap-to-clone [`Arc`] for fan-out to concurrent waiters.
#[derive(Clone)]
struct EncodedSizeBlock(Arc<EthereumBlock>);

impl Encode for EncodedSizeBlock {
	fn encode_to<W: scale_codec::Output + ?Sized>(&self, dest: &mut W) {
		self.0.as_ref().encode_to(dest);
	}
}

#[derive(Clone)]
struct EncodedSizeStatuses(Arc<Vec<TransactionStatus>>);

impl Encode for EncodedSizeStatuses {
	fn encode_to<W: scale_codec::Output + ?Sized>(&self, dest: &mut W) {
		self.0.as_slice().encode_to(dest);
	}
}

struct EthBlockDataCacheMetrics {
	coalesced_waiters: HistogramVec,
	waiter_joined: IntCounterVec,
	fetch_started: IntCounterVec,
}

impl EthBlockDataCacheMetrics {
	fn register(registry: &prometheus_endpoint::Registry) -> Result<Self, prometheus_endpoint::PrometheusError> {
		let coalesced_waiters = prometheus_endpoint::register(
			HistogramVec::new(
				HistogramOpts::new(
					"frontier_eth_block_data_cache_coalesced_waiters",
					"How many concurrent RPC waiters were satisfied by one storage fetch (per kind).",
				)
				.buckets(vec![
					1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0, 144.0, 233.0,
				]),
				&["kind"],
			)?,
			registry,
		)?;
		let waiter_joined = prometheus_endpoint::register(
			IntCounterVec::new(
				Opts::new(
					"frontier_eth_block_data_cache_waiter_joined_total",
					"Waiters attached to an in-flight fetch (coalescing) by kind.",
				),
				&["kind"],
			)?,
			registry,
		)?;
		let fetch_started = prometheus_endpoint::register(
			IntCounterVec::new(
				Opts::new(
					"frontier_eth_block_data_cache_fetch_started_total",
					"In-flight storage fetches started by the block data cache (per kind).",
				),
				&["kind"],
			)?,
			registry,
		)?;
		Ok(Self {
			coalesced_waiters,
			waiter_joined,
			fetch_started,
		})
	}

	fn observe_coalesced(&self, kind: &'static str, waiters: usize) {
		self.coalesced_waiters
			.with_label_values(&[kind])
			.observe(waiters as f64);
	}

	fn inc_waiter_joined(&self, kind: &'static str) {
		self.waiter_joined.with_label_values(&[kind]).inc();
	}

	fn inc_fetch_started(&self, kind: &'static str) {
		self.fetch_started.with_label_values(&[kind]).inc();
	}
}

enum EthBlockDataCacheMessage<B: BlockT> {
	RequestCurrentBlock {
		block_hash: B::Hash,
		response_tx: oneshot::Sender<Option<Arc<EthereumBlock>>>,
	},
	FetchedCurrentBlock {
		block_hash: B::Hash,
		block: Option<EthereumBlock>,
	},

	RequestCurrentTransactionStatuses {
		block_hash: B::Hash,
		response_tx: oneshot::Sender<Option<Arc<Vec<TransactionStatus>>>>,
	},
	FetchedCurrentTransactionStatuses {
		block_hash: B::Hash,
		statuses: Option<Vec<TransactionStatus>>,
	},
}

/// Manage LRU caches for block data and their transaction statuses.
/// These are large and take a lot of time to fetch from the database.
/// Storing them in an LRU cache will allow to reduce database accesses
/// when many subsequent requests are related to the same blocks.
pub struct EthBlockDataCacheTask<B: BlockT>(mpsc::Sender<EthBlockDataCacheMessage<B>>);

impl<B: BlockT> EthBlockDataCacheTask<B> {
	pub fn new(
		spawn_handle: SpawnTaskHandle,
		storage_override: Arc<dyn StorageOverride<B>>,
		blocks_cache_max_size: usize,
		statuses_cache_max_size: usize,
		prometheus_registry: Option<prometheus_endpoint::Registry>,
	) -> Self {
		let (task_tx, mut task_rx) = mpsc::channel(CACHE_TASK_CHANNEL_CAPACITY);
		let outer_task_tx = task_tx.clone();
		let outer_spawn_handle = spawn_handle.clone();

		let cache_metrics = prometheus_registry
			.as_ref()
			.and_then(|r| match EthBlockDataCacheMetrics::register(r) {
				Ok(m) => Some(m),
				Err(e) => {
					log::error!(target: "eth-cache", "Failed to register eth block data cache metrics: {e:?}");
					None
				}
			});

		outer_spawn_handle.spawn("EthBlockDataCacheTask", None, async move {
			let mut blocks_cache = LRUCacheByteLimited::<B::Hash, EncodedSizeBlock>::new(
				"blocks_cache",
				blocks_cache_max_size as u64,
				prometheus_registry.clone(),
			);
			let mut statuses_cache = LRUCacheByteLimited::<B::Hash, EncodedSizeStatuses>::new(
				"statuses_cache",
				statuses_cache_max_size as u64,
				prometheus_registry,
			);

			let mut awaiting_blocks = BlockWaitList::<B>::new();
			let mut awaiting_statuses = StatusesWaitList::<B>::new();

			// Handle all incoming messages.
			// Exits when there are no more senders.
			// Any long computation should be spawned in a separate task
			// to keep this task handle messages as soon as possible.
			while let Some(message) = task_rx.recv().await {
				use EthBlockDataCacheMessage::*;
				match message {
					RequestCurrentBlock {
						block_hash,
						response_tx,
					} => Self::request_current_block(
						&spawn_handle,
						&mut blocks_cache,
						&mut awaiting_blocks,
						storage_override.clone(),
						block_hash,
						response_tx,
						task_tx.clone(),
						cache_metrics.as_ref(),
					),
					FetchedCurrentBlock { block_hash, block } => {
						if let Some(wait_list) = awaiting_blocks.remove(&block_hash) {
							let n = wait_list.len();
							if let Some(m) = cache_metrics.as_ref() {
								m.observe_coalesced("block", n);
							}
							let shared = block.map(Arc::new);
							for sender in wait_list {
								let _ = sender.send(shared.clone());
							}
							if let Some(block) = shared {
								blocks_cache.put(block_hash, EncodedSizeBlock(block));
							}
						}
					}

					RequestCurrentTransactionStatuses {
						block_hash,
						response_tx,
					} => Self::request_current_statuses(
						&spawn_handle,
						&mut statuses_cache,
						&mut awaiting_statuses,
						storage_override.clone(),
						block_hash,
						response_tx,
						task_tx.clone(),
						cache_metrics.as_ref(),
					),
					FetchedCurrentTransactionStatuses {
						block_hash,
						statuses,
					} => {
						if let Some(wait_list) = awaiting_statuses.remove(&block_hash) {
							let n = wait_list.len();
							if let Some(m) = cache_metrics.as_ref() {
								m.observe_coalesced("statuses", n);
							}
							let shared = statuses.map(|s| Arc::new(s));
							for sender in wait_list {
								let _ = sender.send(shared.clone());
							}
							if let Some(statuses) = shared {
								statuses_cache.put(block_hash, EncodedSizeStatuses(statuses));
							}
						}
					}
				}
			}
		});

		Self(outer_task_tx)
	}

	fn request_current_block(
		spawn_handle: &SpawnTaskHandle,
		cache: &mut LRUCacheByteLimited<B::Hash, EncodedSizeBlock>,
		wait_list: &mut BlockWaitList<B>,
		storage_override: Arc<dyn StorageOverride<B>>,
		block_hash: B::Hash,
		response_tx: oneshot::Sender<Option<Arc<EthereumBlock>>>,
		task_tx: mpsc::Sender<EthBlockDataCacheMessage<B>>,
		metrics: Option<&EthBlockDataCacheMetrics>,
	) {
		if let Some(w) = cache.get(&block_hash).cloned() {
			let _ = response_tx.send(Some(w.0));
			return;
		}

		if let Some(waiting) = wait_list.get_mut(&block_hash) {
			waiting.push(response_tx);
			if let Some(m) = metrics {
				m.inc_waiter_joined("block");
			}
			return;
		}

		wait_list.insert(block_hash, vec![response_tx]);
		if let Some(m) = metrics {
			m.inc_fetch_started("block");
		}

		spawn_handle.spawn("EthBlockDataCacheTask Worker", None, async move {
			let message = EthBlockDataCacheMessage::FetchedCurrentBlock {
				block_hash,
				block: storage_override.current_block(block_hash),
			};
			let _ = task_tx.send(message).await;
		});
	}

	fn request_current_statuses(
		spawn_handle: &SpawnTaskHandle,
		cache: &mut LRUCacheByteLimited<B::Hash, EncodedSizeStatuses>,
		wait_list: &mut StatusesWaitList<B>,
		storage_override: Arc<dyn StorageOverride<B>>,
		block_hash: B::Hash,
		response_tx: oneshot::Sender<Option<Arc<Vec<TransactionStatus>>>>,
		task_tx: mpsc::Sender<EthBlockDataCacheMessage<B>>,
		metrics: Option<&EthBlockDataCacheMetrics>,
	) {
		if let Some(w) = cache.get(&block_hash).cloned() {
			let _ = response_tx.send(Some(w.0));
			return;
		}

		if let Some(waiting) = wait_list.get_mut(&block_hash) {
			waiting.push(response_tx);
			if let Some(m) = metrics {
				m.inc_waiter_joined("statuses");
			}
			return;
		}

		wait_list.insert(block_hash, vec![response_tx]);
		if let Some(m) = metrics {
			m.inc_fetch_started("statuses");
		}

		spawn_handle.spawn("EthBlockDataCacheTask Worker", None, async move {
			let message = EthBlockDataCacheMessage::FetchedCurrentTransactionStatuses {
				block_hash,
				statuses: storage_override.current_transaction_statuses(block_hash),
			};
			let _ = task_tx.send(message).await;
		});
	}

	/// Cache for `handler.current_block`.
	pub async fn current_block(&self, block_hash: B::Hash) -> Option<EthereumBlock> {
		let (response_tx, response_rx) = oneshot::channel();

		self.0
			.send(EthBlockDataCacheMessage::RequestCurrentBlock {
				block_hash,
				response_tx,
			})
			.await
			.ok()?;

		response_rx
			.await
			.ok()?
			.map(|arc| (*arc).clone())
	}

	/// Cache for `handler.current_transaction_statuses`.
	pub async fn current_transaction_statuses(
		&self,
		block_hash: B::Hash,
	) -> Option<Vec<TransactionStatus>> {
		let (response_tx, response_rx) = oneshot::channel();

		self.0
			.send(
				EthBlockDataCacheMessage::RequestCurrentTransactionStatuses {
					block_hash,
					response_tx,
				},
			)
			.await
			.ok()?;

		response_rx
			.await
			.ok()?
			.map(|arc| (*arc).clone())
	}
}

pub struct EthTask<B, C, BE>(PhantomData<(B, C, BE)>);

impl<B, C, BE> EthTask<B, C, BE>
where
	B: BlockT,
	C: ProvideRuntimeApi<B>,
	C::Api: EthereumRuntimeRPCApi<B>,
	C: BlockchainEvents<B> + 'static,
	C: HeaderBackend<B> + StorageProvider<B, BE>,
	BE: Backend<B> + 'static,
{
	pub async fn filter_pool_task(
		client: Arc<C>,
		filter_pool: Arc<Mutex<BTreeMap<U256, FilterPoolItem>>>,
		retain_threshold: u64,
	) {
		let mut notification_st = client.import_notification_stream();

		while let Some(notification) = notification_st.next().await {
			if let Ok(filter_pool) = &mut filter_pool.lock() {
				let imported_number: u64 = UniqueSaturatedInto::<u64>::unique_saturated_into(
					*notification.header.number(),
				);

				filter_pool.retain(|_, v| v.at_block + retain_threshold > imported_number);
			}
		}
	}

	pub async fn fee_history_task(
		client: Arc<C>,
		storage_override: Arc<dyn StorageOverride<B>>,
		fee_history_cache: FeeHistoryCache,
		block_limit: u64,
	) {
		struct TransactionHelper {
			gas_used: u64,
			effective_reward: u64,
		}
		// Calculates the cache for a single block
		#[rustfmt::skip]
			let fee_history_cache_item = |hash: B::Hash| -> (
			FeeHistoryCacheItem,
			Option<u64>
		) {
			// Evenly spaced percentile list from 0.0 to 100.0 with a 0.5 resolution.
			// This means we cache 200 percentile points.
			// Later in request handling we will approximate by rounding percentiles that
			// fall in between with `(round(n*2)/2)`.
			let reward_percentiles: Vec<f64> = {
				let mut percentile: f64 = 0.0;
				(0..201)
					.map(|_| {
						let val = percentile;
						percentile += 0.5;
						val
					})
					.collect()
			};

			let block = storage_override.current_block(hash);
			let mut block_number: Option<u64> = None;
			let base_fee = client.runtime_api().gas_price(hash).unwrap_or_default();
			let receipts = storage_override.current_receipts(hash);
			let mut result = FeeHistoryCacheItem {
				base_fee: UniqueSaturatedInto::<u64>::unique_saturated_into(base_fee),
				gas_used_ratio: 0f64,
				rewards: Vec::new(),
			};
			if let (Some(block), Some(receipts)) = (block, receipts) {
				block_number = Some(UniqueSaturatedInto::<u64>::unique_saturated_into(block.header.number));
				let gas_used = UniqueSaturatedInto::<u64>::unique_saturated_into(block.header.gas_used) as f64;
				let gas_limit = UniqueSaturatedInto::<u64>::unique_saturated_into(block.header.gas_limit) as f64;
				result.gas_used_ratio = gas_used / gas_limit;

				let mut previous_cumulative_gas = U256::zero();
				let used_gas = |current: U256, previous: &mut U256| -> u64 {
					let r = UniqueSaturatedInto::<u64>::unique_saturated_into(current.saturating_sub(*previous));
					*previous = current;
					r
				};
				// Build a list of relevant transaction information.
				let mut transactions: Vec<TransactionHelper> = receipts
					.iter()
					.enumerate()
					.map(|(i, receipt)| TransactionHelper {
						gas_used: match receipt {
							ethereum::ReceiptV4::Legacy(d) | ethereum::ReceiptV4::EIP2930(d) | ethereum::ReceiptV4::EIP1559(d) | ethereum::ReceiptV4::EIP7702(d) => used_gas(d.used_gas, &mut previous_cumulative_gas),
						},
						effective_reward: match block.transactions.get(i) {
							Some(ethereum::TransactionV3::Legacy(t)) => {
								UniqueSaturatedInto::<u64>::unique_saturated_into(t.gas_price.saturating_sub(base_fee))
							}
							Some(ethereum::TransactionV3::EIP2930(t)) => {
								UniqueSaturatedInto::<u64>::unique_saturated_into(t.gas_price.saturating_sub(base_fee))
							}
							Some(ethereum::TransactionV3::EIP1559(t)) => UniqueSaturatedInto::<u64>::unique_saturated_into(
									t
										.max_priority_fee_per_gas
										.min(t.max_fee_per_gas.saturating_sub(base_fee))
							),
							Some(ethereum::TransactionV3::EIP7702(t)) => UniqueSaturatedInto::<u64>::unique_saturated_into(
									t
										.max_priority_fee_per_gas
										.min(t.max_fee_per_gas.saturating_sub(base_fee))
							),
							None => 0,
						},
					})
					.collect();
				// Sort ASC by effective reward.
				transactions.sort_by(|a, b| a.effective_reward.cmp(&b.effective_reward));

				// Calculate percentile rewards.
				result.rewards = reward_percentiles
					.into_iter()
					.filter_map(|p| {
						let target_gas = (p * gas_used / 100f64) as u64;
						let mut sum_gas = 0;
						for tx in &transactions {
							sum_gas += tx.gas_used;
							if target_gas <= sum_gas {
								return Some(tx.effective_reward);
							}
						}
						None
					})
					.collect();
			} else {
				result.rewards = reward_percentiles.iter().map(|_| 0).collect();
			}
			(result, block_number)
		};

		// Commits the result to cache
		let commit_if_any = |item: FeeHistoryCacheItem, key: Option<u64>| {
			if let (Some(block_number), Ok(fee_history_cache)) =
				(key, &mut fee_history_cache.lock())
			{
				fee_history_cache.insert(block_number, item);
				// We want to remain within the configured cache bounds.
				// The first key out of bounds.
				let first_out = block_number.saturating_sub(block_limit);
				// Out of bounds size.
				let to_remove = (fee_history_cache.len() as u64).saturating_sub(block_limit);
				// Remove all cache data before `block_limit`.
				for i in 0..to_remove {
					// Cannot overflow.
					let key = first_out - i;
					fee_history_cache.remove(&key);
				}
			}
		};

		let mut notification_st = client.import_notification_stream();

		while let Some(notification) = notification_st.next().await {
			if notification.is_new_best {
				// In case a re-org happened on import.
				if let Some(tree_route) = notification.tree_route {
					if let Ok(fee_history_cache) = &mut fee_history_cache.lock() {
						// Remove retracted.
						let _lock = tree_route.retracted().iter().map(|hash_and_number| {
							let n = UniqueSaturatedInto::<u64>::unique_saturated_into(
								hash_and_number.number,
							);
							fee_history_cache.remove(&n);
						});
						// Insert enacted.
						let _ = tree_route.enacted().iter().map(|hash_and_number| {
							let (result, block_number) =
								fee_history_cache_item(hash_and_number.hash);
							commit_if_any(result, block_number);
						});
					}
				}
				// Cache the imported block.
				let (result, block_number) = fee_history_cache_item(notification.hash);
				commit_if_any(result, block_number);
			}
		}
	}
}
