#![feature(prelude_import)]
#![allow(
    clippy::too_many_arguments,
    clippy::large_enum_variant,
    clippy::manual_range_contains,
    clippy::explicit_counter_loop,
    clippy::len_zero,
    clippy::new_without_default
)]
#[prelude_import]
use std::prelude::rust_2021::*;
#[macro_use]
extern crate std;
mod eth {
    mod block {
        use std::sync::Arc;
        use ethereum_types::{H256, U256};
        use jsonrpsee::core::RpcResult as Result;
        use sc_client_api::backend::{Backend, StateBackend, StorageProvider};
        use sc_network_common::ExHashT;
        use sc_transaction_pool::ChainApi;
        use sp_api::{BlockId, HeaderT, ProvideRuntimeApi};
        use sp_blockchain::HeaderBackend;
        use sp_core::hashing::keccak_256;
        use sp_runtime::traits::{BlakeTwo256, Block as BlockT};
        use fc_rpc_core::types::*;
        use fp_rpc::EthereumRuntimeRPCApi;
        use crate::{
            eth::{rich_block_build, Eth},
            frontier_backend_client, internal_err,
        };
        impl<B, C, P, CT, BE, H: ExHashT, A: ChainApi, EGA> Eth<B, C, P, CT, BE, H, A, EGA>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: StorageProvider<B, BE> + HeaderBackend<B> + Send + Sync + 'static,
            C: ProvideRuntimeApi<B>,
            C::Api: EthereumRuntimeRPCApi<B>,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
        {
            pub async fn block_by_hash(&self, hash: H256, full: bool) -> Result<Option<RichBlock>> {
                let client = Arc::clone(&self.client);
                let block_data_cache = Arc::clone(&self.block_data_cache);
                let backend = Arc::clone(&self.backend);
                let id = match frontier_backend_client::load_hash::<B, C>(
                    client.as_ref(),
                    backend.as_ref(),
                    hash,
                )
                .map_err(|err| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &[""],
                            &[::core::fmt::ArgumentV1::new_debug(&err)],
                        ));
                        res
                    })
                })? {
                    Some(hash) => hash,
                    _ => return Ok(None),
                };
                let substrate_hash = client.expect_block_hash_from_id(&id).map_err(|_| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &["Expect block number from id: "],
                            &[::core::fmt::ArgumentV1::new_display(&id)],
                        ));
                        res
                    })
                })?;
                let schema = frontier_backend_client::onchain_storage_schema::<B, C, BE>(
                    client.as_ref(),
                    id,
                );
                let block = block_data_cache.current_block(schema, substrate_hash).await;
                let statuses = block_data_cache
                    .current_transaction_statuses(schema, substrate_hash)
                    .await;
                let base_fee = client.runtime_api().gas_price(&id).ok();
                match (block, statuses) {
                    (Some(block), Some(statuses)) => {
                        let mut rich_block = rich_block_build(
                            block,
                            statuses.into_iter().map(Option::Some).collect(),
                            Some(hash),
                            full,
                            base_fee,
                        );
                        let number = rich_block.inner.header.number.unwrap_or_default();
                        if rich_block.inner.header.parent_hash == H256::default()
                            && number > U256::zero()
                        {
                            if let Ok(Some(header)) = client.header(substrate_hash) {
                                let parent_hash = *header.parent_hash();
                                let parent_id = BlockId::Hash(parent_hash);
                                let schema =
                                    frontier_backend_client::onchain_storage_schema::<B, C, BE>(
                                        client.as_ref(),
                                        parent_id,
                                    );
                                if let Some(block) =
                                    block_data_cache.current_block(schema, parent_hash).await
                                {
                                    rich_block.inner.header.parent_hash = H256::from_slice(
                                        keccak_256(&rlp::encode(&block.header)).as_slice(),
                                    );
                                }
                            }
                        }
                        Ok(Some(rich_block))
                    }
                    _ => Ok(None),
                }
            }
            pub async fn block_by_number(
                &self,
                number: BlockNumber,
                full: bool,
            ) -> Result<Option<RichBlock>> {
                let client = Arc::clone(&self.client);
                let block_data_cache = Arc::clone(&self.block_data_cache);
                let backend = Arc::clone(&self.backend);
                let id = match frontier_backend_client::native_block_id::<B, C>(
                    client.as_ref(),
                    backend.as_ref(),
                    Some(number),
                )? {
                    Some(id) => id,
                    None => return Ok(None),
                };
                let substrate_hash = client.expect_block_hash_from_id(&id).map_err(|_| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &["Expect block number from id: "],
                            &[::core::fmt::ArgumentV1::new_display(&id)],
                        ));
                        res
                    })
                })?;
                let schema = frontier_backend_client::onchain_storage_schema::<B, C, BE>(
                    client.as_ref(),
                    id,
                );
                let block = block_data_cache.current_block(schema, substrate_hash).await;
                let statuses = block_data_cache
                    .current_transaction_statuses(schema, substrate_hash)
                    .await;
                let base_fee = client.runtime_api().gas_price(&id).ok();
                match (block, statuses) {
                    (Some(block), Some(statuses)) => {
                        let hash = H256::from(keccak_256(&rlp::encode(&block.header)));
                        let mut rich_block = rich_block_build(
                            block,
                            statuses.into_iter().map(Option::Some).collect(),
                            Some(hash),
                            full,
                            base_fee,
                        );
                        let number = rich_block.inner.header.number.unwrap_or_default();
                        if rich_block.inner.header.parent_hash == H256::default()
                            && number > U256::zero()
                        {
                            if let Ok(Some(header)) = client.header(substrate_hash) {
                                let parent_hash = *header.parent_hash();
                                let parent_id = BlockId::Hash(parent_hash);
                                let schema =
                                    frontier_backend_client::onchain_storage_schema::<B, C, BE>(
                                        client.as_ref(),
                                        parent_id,
                                    );
                                if let Some(block) =
                                    block_data_cache.current_block(schema, parent_hash).await
                                {
                                    rich_block.inner.header.parent_hash = H256::from_slice(
                                        keccak_256(&rlp::encode(&block.header)).as_slice(),
                                    );
                                }
                            }
                        }
                        Ok(Some(rich_block))
                    }
                    _ => Ok(None),
                }
            }
            pub fn block_transaction_count_by_hash(&self, hash: H256) -> Result<Option<U256>> {
                let id = match frontier_backend_client::load_hash::<B, C>(
                    self.client.as_ref(),
                    self.backend.as_ref(),
                    hash,
                )
                .map_err(|err| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &[""],
                            &[::core::fmt::ArgumentV1::new_debug(&err)],
                        ));
                        res
                    })
                })? {
                    Some(hash) => hash,
                    _ => return Ok(None),
                };
                let schema = frontier_backend_client::onchain_storage_schema::<B, C, BE>(
                    self.client.as_ref(),
                    id,
                );
                let block = self
                    .overrides
                    .schemas
                    .get(&schema)
                    .unwrap_or(&self.overrides.fallback)
                    .current_block(&id);
                match block {
                    Some(block) => Ok(Some(U256::from(block.transactions.len()))),
                    None => Ok(None),
                }
            }
            pub fn block_transaction_count_by_number(
                &self,
                number: BlockNumber,
            ) -> Result<Option<U256>> {
                if let BlockNumber::Pending = number {
                    return Ok(Some(U256::from(
                        self.graph.validated_pool().ready().count(),
                    )));
                }
                let id = match frontier_backend_client::native_block_id::<B, C>(
                    self.client.as_ref(),
                    self.backend.as_ref(),
                    Some(number),
                )? {
                    Some(id) => id,
                    None => return Ok(None),
                };
                let schema = frontier_backend_client::onchain_storage_schema::<B, C, BE>(
                    self.client.as_ref(),
                    id,
                );
                let block = self
                    .overrides
                    .schemas
                    .get(&schema)
                    .unwrap_or(&self.overrides.fallback)
                    .current_block(&id);
                match block {
                    Some(block) => Ok(Some(U256::from(block.transactions.len()))),
                    None => Ok(None),
                }
            }
            pub fn block_uncles_count_by_hash(&self, _: H256) -> Result<U256> {
                Ok(U256::zero())
            }
            pub fn block_uncles_count_by_number(&self, _: BlockNumber) -> Result<U256> {
                Ok(U256::zero())
            }
            pub fn uncle_by_block_hash_and_index(
                &self,
                _: H256,
                _: Index,
            ) -> Result<Option<RichBlock>> {
                Ok(None)
            }
            pub fn uncle_by_block_number_and_index(
                &self,
                _: BlockNumber,
                _: Index,
            ) -> Result<Option<RichBlock>> {
                Ok(None)
            }
        }
    }
    mod cache {
        mod lru_cache {
            use lru::LruCache;
            use scale_codec::Encode;
            pub struct LRUCacheByteLimited<K, V> {
                cache: LruCache<K, V>,
                max_size: u64,
                metrics: Option<LRUCacheByteLimitedMetrics>,
                size: u64,
            }
            impl<K: Eq + core::hash::Hash, V: Encode> LRUCacheByteLimited<K, V> {
                pub fn new(
                    cache_name: &'static str,
                    max_size: u64,
                    prometheus_registry: Option<prometheus_endpoint::Registry>,
                ) -> Self {
                    let metrics = match prometheus_registry {
                        Some(registry) => {
                            match LRUCacheByteLimitedMetrics::register(cache_name, &registry) {
                                Ok(metrics) => Some(metrics),
                                Err(e) => {
                                    {
                                        let lvl = ::log::Level::Error;
                                        if lvl <= ::log::STATIC_MAX_LEVEL
                                            && lvl <= ::log::max_level()
                                        {
                                            ::log::__private_api_log(
                                                ::core::fmt::Arguments::new_v1(
                                                    &["Failed to register metrics: "],
                                                    &[::core::fmt::ArgumentV1::new_debug(&e)],
                                                ),
                                                lvl,
                                                &(
                                                    "eth-cache",
                                                    "fc_rpc::eth::cache::lru_cache",
                                                    "client/rpc/src/eth/cache/lru_cache.rs",
                                                    39u32,
                                                ),
                                                ::log::__private_api::Option::None,
                                            );
                                        }
                                    };
                                    None
                                }
                            }
                        }
                        None => None,
                    };
                    Self {
                        cache: LruCache::unbounded(),
                        max_size,
                        metrics,
                        size: 0,
                    }
                }
                pub fn get(&mut self, k: &K) -> Option<&V> {
                    if let Some(v) = self.cache.get(k) {
                        if let Some(metrics) = &self.metrics {
                            metrics.hits.inc();
                        }
                        Some(v)
                    } else {
                        if let Some(metrics) = &self.metrics {
                            metrics.miss.inc();
                        }
                        None
                    }
                }
                pub fn put(&mut self, k: K, v: V) {
                    self.size += v.encoded_size() as u64;
                    while self.size > self.max_size {
                        if let Some((_, v)) = self.cache.pop_lru() {
                            let v_size = v.encoded_size() as u64;
                            self.size -= v_size;
                        } else {
                            break;
                        }
                    }
                    self.cache.put(k, v);
                    if let Some(metrics) = &self.metrics {
                        metrics.size.set(self.size);
                    }
                }
            }
            struct LRUCacheByteLimitedMetrics {
                hits: prometheus::IntCounter,
                miss: prometheus::IntCounter,
                size: prometheus_endpoint::Gauge<prometheus_endpoint::U64>,
            }
            impl LRUCacheByteLimitedMetrics {
                pub(crate) fn register(
                    cache_name: &'static str,
                    registry: &prometheus_endpoint::Registry,
                ) -> std::result::Result<Self, prometheus_endpoint::PrometheusError>
                {
                    Ok(Self {
                        hits: prometheus_endpoint::register(
                            prometheus::IntCounter::new(
                                {
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["frontier_eth_", "_hits"],
                                        &[::core::fmt::ArgumentV1::new_display(&cache_name)],
                                    ));
                                    res
                                },
                                {
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["Hits of eth ", " cache."],
                                        &[::core::fmt::ArgumentV1::new_display(&cache_name)],
                                    ));
                                    res
                                },
                            )?,
                            registry,
                        )?,
                        miss: prometheus_endpoint::register(
                            prometheus::IntCounter::new(
                                {
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["frontier_eth_", "_miss"],
                                        &[::core::fmt::ArgumentV1::new_display(&cache_name)],
                                    ));
                                    res
                                },
                                {
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["Misses of eth ", " cache."],
                                        &[::core::fmt::ArgumentV1::new_display(&cache_name)],
                                    ));
                                    res
                                },
                            )?,
                            registry,
                        )?,
                        size: prometheus_endpoint::register(
                            prometheus_endpoint::Gauge::new(
                                {
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["frontier_eth_", "_size"],
                                        &[::core::fmt::ArgumentV1::new_display(&cache_name)],
                                    ));
                                    res
                                },
                                {
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["Size of eth ", " data cache."],
                                        &[::core::fmt::ArgumentV1::new_display(&cache_name)],
                                    ));
                                    res
                                },
                            )?,
                            registry,
                        )?,
                    })
                }
            }
        }
        use std::{
            collections::{BTreeMap, HashMap},
            marker::PhantomData,
            sync::{Arc, Mutex},
        };
        use ethereum::BlockV2 as EthereumBlock;
        use ethereum_types::{H256, U256};
        use futures::StreamExt;
        use tokio::sync::{mpsc, oneshot};
        use sc_client_api::{
            backend::{Backend, StateBackend, StorageProvider},
            client::BlockchainEvents,
        };
        use sc_service::SpawnTaskHandle;
        use sp_api::ProvideRuntimeApi;
        use sp_blockchain::HeaderBackend;
        use sp_runtime::{
            generic::BlockId,
            traits::{BlakeTwo256, Block as BlockT, Header as HeaderT, UniqueSaturatedInto},
        };
        use fc_rpc_core::types::*;
        use fp_rpc::{EthereumRuntimeRPCApi, TransactionStatus};
        use fp_storage::EthereumStorageSchema;
        use self::lru_cache::LRUCacheByteLimited;
        use crate::{
            frontier_backend_client,
            overrides::{OverrideHandle, StorageOverride},
        };
        type WaitList<Hash, T> = HashMap<Hash, Vec<oneshot::Sender<Option<T>>>>;
        enum EthBlockDataCacheMessage<B: BlockT> {
            RequestCurrentBlock {
                block_hash: B::Hash,
                schema: EthereumStorageSchema,
                response_tx: oneshot::Sender<Option<EthereumBlock>>,
            },
            FetchedCurrentBlock {
                block_hash: B::Hash,
                block: Option<EthereumBlock>,
            },
            RequestCurrentTransactionStatuses {
                block_hash: B::Hash,
                schema: EthereumStorageSchema,
                response_tx: oneshot::Sender<Option<Vec<TransactionStatus>>>,
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
                overrides: Arc<OverrideHandle<B>>,
                blocks_cache_max_size: usize,
                statuses_cache_max_size: usize,
                prometheus_registry: Option<prometheus_endpoint::Registry>,
            ) -> Self {
                let (task_tx, mut task_rx) = mpsc::channel(100);
                let outer_task_tx = task_tx.clone();
                let outer_spawn_handle = spawn_handle.clone();
                outer_spawn_handle.spawn("EthBlockDataCacheTask", None, async move {
                    let mut blocks_cache = LRUCacheByteLimited::<B::Hash, EthereumBlock>::new(
                        "blocks_cache",
                        blocks_cache_max_size as u64,
                        prometheus_registry.clone(),
                    );
                    let mut statuses_cache =
                        LRUCacheByteLimited::<B::Hash, Vec<TransactionStatus>>::new(
                            "statuses_cache",
                            statuses_cache_max_size as u64,
                            prometheus_registry,
                        );
                    let mut awaiting_blocks =
                        HashMap::<B::Hash, Vec<oneshot::Sender<Option<EthereumBlock>>>>::new();
                    let mut awaiting_statuses = HashMap::<
                        B::Hash,
                        Vec<oneshot::Sender<Option<Vec<TransactionStatus>>>>,
                    >::new();
                    while let Some(message) = task_rx.recv().await {
                        use EthBlockDataCacheMessage::*;
                        match message {
                            RequestCurrentBlock {
                                block_hash,
                                schema,
                                response_tx,
                            } => Self::request_current(
                                &spawn_handle,
                                &mut blocks_cache,
                                &mut awaiting_blocks,
                                Arc::clone(&overrides),
                                block_hash,
                                schema,
                                response_tx,
                                task_tx.clone(),
                                move |handler| FetchedCurrentBlock {
                                    block_hash,
                                    block: handler.current_block(&BlockId::Hash(block_hash)),
                                },
                            ),
                            FetchedCurrentBlock { block_hash, block } => {
                                if let Some(wait_list) = awaiting_blocks.remove(&block_hash) {
                                    for sender in wait_list {
                                        let _ = sender.send(block.clone());
                                    }
                                }
                                if let Some(block) = block {
                                    blocks_cache.put(block_hash, block);
                                }
                            }
                            RequestCurrentTransactionStatuses {
                                block_hash,
                                schema,
                                response_tx,
                            } => Self::request_current(
                                &spawn_handle,
                                &mut statuses_cache,
                                &mut awaiting_statuses,
                                Arc::clone(&overrides),
                                block_hash,
                                schema,
                                response_tx,
                                task_tx.clone(),
                                move |handler| FetchedCurrentTransactionStatuses {
                                    block_hash,
                                    statuses: handler
                                        .current_transaction_statuses(&BlockId::Hash(block_hash)),
                                },
                            ),
                            FetchedCurrentTransactionStatuses {
                                block_hash,
                                statuses,
                            } => {
                                if let Some(wait_list) = awaiting_statuses.remove(&block_hash) {
                                    for sender in wait_list {
                                        let _ = sender.send(statuses.clone());
                                    }
                                }
                                if let Some(statuses) = statuses {
                                    statuses_cache.put(block_hash, statuses);
                                }
                            }
                        }
                    }
                });
                Self(outer_task_tx)
            }
            fn request_current<T, F>(
                spawn_handle: &SpawnTaskHandle,
                cache: &mut LRUCacheByteLimited<B::Hash, T>,
                wait_list: &mut WaitList<B::Hash, T>,
                overrides: Arc<OverrideHandle<B>>,
                block_hash: B::Hash,
                schema: EthereumStorageSchema,
                response_tx: oneshot::Sender<Option<T>>,
                task_tx: mpsc::Sender<EthBlockDataCacheMessage<B>>,
                handler_call: F,
            ) where
                T: Clone + scale_codec::Encode,
                F: FnOnce(
                    &Box<dyn StorageOverride<B> + Send + Sync>,
                ) -> EthBlockDataCacheMessage<B>,
                F: Send + 'static,
            {
                if let Some(data) = cache.get(&block_hash).cloned() {
                    let _ = response_tx.send(Some(data));
                    return;
                }
                if let Some(waiting) = wait_list.get_mut(&block_hash) {
                    waiting.push(response_tx);
                    return;
                }
                wait_list.insert(
                    block_hash,
                    <[_]>::into_vec(
                        #[rustc_box]
                        ::alloc::boxed::Box::new([response_tx]),
                    ),
                );
                spawn_handle.spawn("EthBlockDataCacheTask Worker", None, async move {
                    let handler = overrides
                        .schemas
                        .get(&schema)
                        .unwrap_or(&overrides.fallback);
                    let message = handler_call(handler);
                    let _ = task_tx.send(message).await;
                });
            }
            /// Cache for `handler.current_block`.
            pub async fn current_block(
                &self,
                schema: EthereumStorageSchema,
                block_hash: B::Hash,
            ) -> Option<EthereumBlock> {
                let (response_tx, response_rx) = oneshot::channel();
                self.0
                    .send(EthBlockDataCacheMessage::RequestCurrentBlock {
                        block_hash,
                        schema,
                        response_tx,
                    })
                    .await
                    .ok()?;
                response_rx.await.ok()?
            }
            /// Cache for `handler.current_transaction_statuses`.
            pub async fn current_transaction_statuses(
                &self,
                schema: EthereumStorageSchema,
                block_hash: B::Hash,
            ) -> Option<Vec<TransactionStatus>> {
                let (response_tx, response_rx) = oneshot::channel();
                self.0
                    .send(
                        EthBlockDataCacheMessage::RequestCurrentTransactionStatuses {
                            block_hash,
                            schema,
                            response_tx,
                        },
                    )
                    .await
                    .ok()?;
                response_rx.await.ok()?
            }
        }
        pub struct EthTask<B, C, BE>(PhantomData<(B, C, BE)>);
        impl<B, C, BE> EthTask<B, C, BE>
        where
            B: BlockT<Hash = H256>,
            C: ProvideRuntimeApi<B> + StorageProvider<B, BE> + BlockchainEvents<B>,
            C: HeaderBackend<B> + Send + Sync + 'static,
            C::Api: EthereumRuntimeRPCApi<B>,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
        {
            pub async fn filter_pool_task(
                client: Arc<C>,
                filter_pool: Arc<Mutex<BTreeMap<U256, FilterPoolItem>>>,
                retain_threshold: u64,
            ) {
                let mut notification_st = client.import_notification_stream();
                while let Some(notification) = notification_st.next().await {
                    if let Ok(filter_pool) = &mut filter_pool.lock() {
                        let imported_number: u64 =
                            UniqueSaturatedInto::<u64>::unique_saturated_into(
                                *notification.header.number(),
                            );
                        filter_pool.retain(|_, v| v.at_block + retain_threshold > imported_number);
                    }
                }
            }
            pub async fn fee_history_task(
                client: Arc<C>,
                overrides: Arc<OverrideHandle<B>>,
                fee_history_cache: FeeHistoryCache,
                block_limit: u64,
            ) {
                struct TransactionHelper {
                    gas_used: u64,
                    effective_reward: u64,
                }
                # [rustfmt :: skip] let fee_history_cache_item = | hash : H256 | -> (FeeHistoryCacheItem , Option < u64 >) { let id = BlockId :: Hash (hash) ; let schema = frontier_backend_client :: onchain_storage_schema :: < B , C , BE > (client . as_ref () , id) ; let handler = overrides . schemas . get (& schema) . unwrap_or (& overrides . fallback) ; let reward_percentiles : Vec < f64 > = { let mut percentile : f64 = 0.0 ; (0 .. 201) . into_iter () . map (| _ | { let val = percentile ; percentile += 0.5 ; val }) . collect () } ; let block = handler . current_block (& id) ; let mut block_number : Option < u64 > = None ; let base_fee = client . runtime_api () . gas_price (& id) . unwrap_or_default () ; let receipts = handler . current_receipts (& id) ; let mut result = FeeHistoryCacheItem { base_fee : base_fee . as_u64 () , gas_used_ratio : 0f64 , rewards : Vec :: new () , } ; if let (Some (block) , Some (receipts)) = (block , receipts) { block_number = Some (block . header . number . as_u64 ()) ; let gas_used = block . header . gas_used . as_u64 () as f64 ; let gas_limit = block . header . gas_limit . as_u64 () as f64 ; result . gas_used_ratio = gas_used / gas_limit ; let mut previous_cumulative_gas = U256 :: zero () ; let used_gas = | current : U256 , previous : & mut U256 | -> u64 { let r = current . saturating_sub (* previous) . as_u64 () ; * previous = current ; r } ; let mut transactions : Vec < TransactionHelper > = receipts . iter () . enumerate () . map (| (i , receipt) | TransactionHelper { gas_used : match receipt { ethereum :: ReceiptV3 :: Legacy (d) | ethereum :: ReceiptV3 :: EIP2930 (d) | ethereum :: ReceiptV3 :: EIP1559 (d) => used_gas (d . used_gas , & mut previous_cumulative_gas) , } , effective_reward : match block . transactions . get (i) { Some (& ethereum :: TransactionV2 :: Legacy (ref t)) => { t . gas_price . saturating_sub (base_fee) . as_u64 () } Some (& ethereum :: TransactionV2 :: EIP2930 (ref t)) => { t . gas_price . saturating_sub (base_fee) . as_u64 () } Some (& ethereum :: TransactionV2 :: EIP1559 (ref t)) => t . max_priority_fee_per_gas . min (t . max_fee_per_gas . saturating_sub (base_fee)) . as_u64 () , None => 0 , } , }) . collect () ; transactions . sort_by (| a , b | a . effective_reward . cmp (& b . effective_reward)) ; result . rewards = reward_percentiles . into_iter () . filter_map (| p | { let target_gas = (p * gas_used / 100f64) as u64 ; let mut sum_gas = 0 ; for tx in & transactions { sum_gas += tx . gas_used ; if target_gas <= sum_gas { return Some (tx . effective_reward) ; } } None }) . collect () ; } else { result . rewards = reward_percentiles . iter () . map (| _ | 0) . collect () ; } (result , block_number) } ;
                let commit_if_any = |item: FeeHistoryCacheItem, key: Option<u64>| {
                    if let (Some(block_number), Ok(fee_history_cache)) =
                        (key, &mut fee_history_cache.lock())
                    {
                        fee_history_cache.insert(block_number, item);
                        let first_out = block_number.saturating_sub(block_limit);
                        let to_remove =
                            (fee_history_cache.len() as u64).saturating_sub(block_limit);
                        for i in 0..to_remove {
                            let key = first_out - i;
                            fee_history_cache.remove(&key);
                        }
                    }
                };
                let mut notification_st = client.import_notification_stream();
                while let Some(notification) = notification_st.next().await {
                    if notification.is_new_best {
                        if let Some(tree_route) = notification.tree_route {
                            if let Ok(fee_history_cache) = &mut fee_history_cache.lock() {
                                let _lock = tree_route.retracted().iter().map(|hash_and_number| {
                                    let n = UniqueSaturatedInto::<u64>::unique_saturated_into(
                                        hash_and_number.number,
                                    );
                                    fee_history_cache.remove(&n);
                                });
                                let _ = tree_route.enacted().iter().map(|hash_and_number| {
                                    let (result, block_number) =
                                        fee_history_cache_item(hash_and_number.hash);
                                    commit_if_any(result, block_number);
                                });
                            }
                        }
                        let (result, block_number) = fee_history_cache_item(notification.hash);
                        commit_if_any(result, block_number);
                    }
                }
            }
        }
    }
    mod client {
        use ethereum_types::{H160, H256, U256, U64};
        use jsonrpsee::core::RpcResult as Result;
        use sc_client_api::backend::{Backend, StateBackend, StorageProvider};
        use sc_network_common::ExHashT;
        use sc_transaction_pool::ChainApi;
        use sp_api::ProvideRuntimeApi;
        use sp_blockchain::HeaderBackend;
        use sp_consensus::SyncOracle;
        use sp_runtime::{
            generic::BlockId,
            traits::{BlakeTwo256, Block as BlockT, UniqueSaturatedInto},
        };
        use fc_rpc_core::types::*;
        use fp_rpc::EthereumRuntimeRPCApi;
        use crate::{eth::Eth, frontier_backend_client, internal_err};
        impl<B, C, P, CT, BE, H: ExHashT, A: ChainApi, EGA> Eth<B, C, P, CT, BE, H, A, EGA>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: ProvideRuntimeApi<B> + StorageProvider<B, BE>,
            C: HeaderBackend<B> + Send + Sync + 'static,
            C::Api: EthereumRuntimeRPCApi<B>,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
        {
            pub fn protocol_version(&self) -> Result<u64> {
                Ok(1)
            }
            pub fn syncing(&self) -> Result<SyncStatus> {
                if self.network.is_major_syncing() {
                    let block_number =
                        U256::from(UniqueSaturatedInto::<u128>::unique_saturated_into(
                            self.client.info().best_number,
                        ));
                    Ok(SyncStatus::Info(SyncInfo {
                        starting_block: U256::zero(),
                        current_block: block_number,
                        highest_block: block_number,
                        warp_chunks_amount: None,
                        warp_chunks_processed: None,
                    }))
                } else {
                    Ok(SyncStatus::None)
                }
            }
            pub fn author(&self) -> Result<H160> {
                let block = BlockId::Hash(self.client.info().best_hash);
                let schema = frontier_backend_client::onchain_storage_schema::<B, C, BE>(
                    self.client.as_ref(),
                    block,
                );
                Ok(self
                    .overrides
                    .schemas
                    .get(&schema)
                    .unwrap_or(&self.overrides.fallback)
                    .current_block(&block)
                    .ok_or_else(|| internal_err("fetching author through override failed"))?
                    .header
                    .beneficiary)
            }
            pub fn accounts(&self) -> Result<Vec<H160>> {
                let mut accounts = Vec::new();
                for signer in &*self.signers {
                    accounts.append(&mut signer.accounts());
                }
                Ok(accounts)
            }
            pub fn block_number(&self) -> Result<U256> {
                Ok(U256::from(
                    UniqueSaturatedInto::<u128>::unique_saturated_into(
                        self.client.info().best_number,
                    ),
                ))
            }
            pub fn chain_id(&self) -> Result<Option<U64>> {
                let hash = self.client.info().best_hash;
                Ok(Some(
                    self.client
                        .runtime_api()
                        .chain_id(&BlockId::Hash(hash))
                        .map_err(|err| {
                            internal_err({
                                let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                    &["fetch runtime chain id failed: "],
                                    &[::core::fmt::ArgumentV1::new_debug(&err)],
                                ));
                                res
                            })
                        })?
                        .into(),
                ))
            }
        }
    }
    mod execute {
        use std::sync::Arc;
        use ethereum_types::{H256, U256};
        use evm::{ExitError, ExitReason};
        use jsonrpsee::core::RpcResult as Result;
        use sc_client_api::backend::{Backend, StateBackend, StorageProvider};
        use sc_network_common::ExHashT;
        use sc_transaction_pool::ChainApi;
        use sp_api::{ApiExt, ProvideRuntimeApi, CallApiAt};
        use sp_block_builder::BlockBuilder as BlockBuilderApi;
        use sp_blockchain::HeaderBackend;
        use sp_runtime::{
            generic::BlockId,
            traits::{BlakeTwo256, Block as BlockT},
            SaturatedConversion,
        };
        use fc_rpc_core::types::*;
        use fp_rpc::EthereumRuntimeRPCApi;
        use crate::{
            eth::{pending_runtime_api, Eth},
            frontier_backend_client, internal_err,
        };
        /// Default JSONRPC error code return by geth
        pub const JSON_RPC_ERROR_DEFAULT: i32 = -32000;
        /// Allow to adapt a request for `estimate_gas`.
        /// Can be used to estimate gas of some contracts using a different function
        /// in the case the normal gas estimation doesn't work.
        ///
        /// Exemple: a precompile that tries to do a subcall but succeeds regardless of the
        /// success of the subcall. The gas estimation will thus optimize the gas limit down
        /// to the minimum, while we want to estimate a gas limit that will allow the subcall to
        /// have enough gas to succeed.
        pub trait EstimateGasAdapter {
            fn adapt_request(request: CallRequest) -> CallRequest;
        }
        impl EstimateGasAdapter for () {
            fn adapt_request(request: CallRequest) -> CallRequest {
                request
            }
        }
        impl<B, C, P, CT, BE, H: ExHashT, A: ChainApi, EGA> Eth<B, C, P, CT, BE, H, A, EGA>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: ProvideRuntimeApi<B> + StorageProvider<B, BE>,
            C: HeaderBackend<B> + Send + Sync + 'static,
            C::Api: BlockBuilderApi<B> + EthereumRuntimeRPCApi<B>,
            C::Api: CallApiAt<B>,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
            A: ChainApi<Block = B> + 'static,
            EGA: EstimateGasAdapter,
        {
            pub fn call(
                &self,
                request: CallRequest,
                number: Option<BlockNumber>,
                state_override: Option<CallStateOverride>,
            ) -> Result<Bytes> {
                let CallRequest {
                    from,
                    to,
                    gas_price,
                    max_fee_per_gas,
                    max_priority_fee_per_gas,
                    gas,
                    value,
                    data,
                    nonce,
                    access_list,
                    ..
                } = request;
                let (gas_price, max_fee_per_gas, max_priority_fee_per_gas) = {
                    let details =
                        fee_details(gas_price, max_fee_per_gas, max_priority_fee_per_gas)?;
                    (
                        details.gas_price,
                        details.max_fee_per_gas,
                        details.max_priority_fee_per_gas,
                    )
                };
                let (id, api) = match frontier_backend_client::native_block_id::<B, C>(
                    self.client.as_ref(),
                    self.backend.as_ref(),
                    number,
                )? {
                    Some(id) => (id, self.client.runtime_api()),
                    None => {
                        let id = BlockId::Hash(self.client.info().best_hash);
                        let api = pending_runtime_api(self.client.as_ref(), self.graph.as_ref())?;
                        (id, api)
                    }
                };
                if let Err(sp_blockchain::Error::UnknownBlock(_)) =
                    self.client.expect_block_hash_from_id(&id)
                {
                    return Err(crate::err(JSON_RPC_ERROR_DEFAULT, "header not found", None));
                }
                let api_version = if let Ok(Some(api_version)) =
                    api.api_version::<dyn EthereumRuntimeRPCApi<B>>(&id)
                {
                    api_version
                } else {
                    return Err(internal_err("failed to retrieve Runtime Api version"));
                };
                let block = if api_version > 1 {
                    api.current_block(&id).map_err(|err| {
                        internal_err({
                            let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                &["runtime error: "],
                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                            ));
                            res
                        })
                    })?
                } else {
                    #[allow(deprecated)]
                    let legacy_block = api.current_block_before_version_2(&id).map_err(|err| {
                        internal_err({
                            let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                &["runtime error: "],
                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                            ));
                            res
                        })
                    })?;
                    legacy_block.map(|block| block.into())
                };
                let block_gas_limit = block
                    .ok_or_else(|| internal_err("block unavailable, cannot query gas limit"))?
                    .header
                    .gas_limit;
                let max_gas_limit = block_gas_limit * self.execute_gas_limit_multiplier;
                let gas_limit = match gas {
                    Some(amount) => {
                        if amount > max_gas_limit {
                            return Err(internal_err({
                                let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                    &[
                                        "provided gas limit is too high (can be up to ",
                                        "x the block gas limit)",
                                    ],
                                    &[::core::fmt::ArgumentV1::new_display(
                                        &self.execute_gas_limit_multiplier,
                                    )],
                                ));
                                res
                            }));
                        }
                        amount
                    }
                    None => match api.gas_limit_multiplier_support(&id) {
                        Ok(_) => max_gas_limit,
                        _ => block_gas_limit,
                    },
                };
                let data = data.map(|d| d.0).unwrap_or_default();
                match to {
                    Some(to) => {
                        if api_version == 1 {
                            #[allow(deprecated)]
                            let info = api
                                .call_before_version_2(
                                    &id,
                                    from.unwrap_or_default(),
                                    to,
                                    data,
                                    value.unwrap_or_default(),
                                    gas_limit,
                                    gas_price,
                                    nonce,
                                    false,
                                )
                                .map_err(|err| {
                                    internal_err({
                                        let res =
                                            ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                                &["runtime error: "],
                                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                                            ));
                                        res
                                    })
                                })?
                                .map_err(|err| {
                                    internal_err({
                                        let res =
                                            ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                                &["execution fatal: "],
                                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                                            ));
                                        res
                                    })
                                })?;
                            error_on_execution_failure(&info.exit_reason, &info.value)?;
                            Ok(Bytes(info.value))
                        } else if api_version >= 2 && api_version < 4 {
                            #[allow(deprecated)]
                            let info = api
                                .call_before_version_4(
                                    &id,
                                    from.unwrap_or_default(),
                                    to,
                                    data,
                                    value.unwrap_or_default(),
                                    gas_limit,
                                    max_fee_per_gas,
                                    max_priority_fee_per_gas,
                                    nonce,
                                    false,
                                )
                                .map_err(|err| {
                                    internal_err({
                                        let res =
                                            ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                                &["runtime error: "],
                                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                                            ));
                                        res
                                    })
                                })?
                                .map_err(|err| {
                                    internal_err({
                                        let res =
                                            ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                                &["execution fatal: "],
                                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                                            ));
                                        res
                                    })
                                })?;
                            error_on_execution_failure(&info.exit_reason, &info.value)?;
                            Ok(Bytes(info.value))
                        } else if api_version == 4 {
                            let access_list = access_list.unwrap_or_default();
                            let info = api
                                .call(
                                    &id,
                                    from.unwrap_or_default(),
                                    to,
                                    data,
                                    value.unwrap_or_default(),
                                    gas_limit,
                                    max_fee_per_gas,
                                    max_priority_fee_per_gas,
                                    nonce,
                                    false,
                                    Some(
                                        access_list
                                            .into_iter()
                                            .map(|item| (item.address, item.storage_keys))
                                            .collect(),
                                    ),
                                )
                                .map_err(|err| {
                                    internal_err({
                                        let res =
                                            ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                                &["runtime error: "],
                                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                                            ));
                                        res
                                    })
                                })?
                                .map_err(|err| {
                                    internal_err({
                                        let res =
                                            ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                                &["execution fatal: "],
                                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                                            ));
                                        res
                                    })
                                })?;
                            error_on_execution_failure(&info.exit_reason, &info.value)?;
                            Ok(Bytes(info.value))
                        } else {
                            Err(internal_err("failed to retrieve Runtime Api version"))
                        }
                    }
                    None => {
                        if api_version == 1 {
                            #[allow(deprecated)]
                            let info = api
                                .create_before_version_2(
                                    &id,
                                    from.unwrap_or_default(),
                                    data,
                                    value.unwrap_or_default(),
                                    gas_limit,
                                    gas_price,
                                    nonce,
                                    false,
                                )
                                .map_err(|err| {
                                    internal_err({
                                        let res =
                                            ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                                &["runtime error: "],
                                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                                            ));
                                        res
                                    })
                                })?
                                .map_err(|err| {
                                    internal_err({
                                        let res =
                                            ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                                &["execution fatal: "],
                                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                                            ));
                                        res
                                    })
                                })?;
                            error_on_execution_failure(&info.exit_reason, &[])?;
                            let code = api.account_code_at(&id, info.value).map_err(|err| {
                                internal_err({
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["runtime error: "],
                                        &[::core::fmt::ArgumentV1::new_debug(&err)],
                                    ));
                                    res
                                })
                            })?;
                            Ok(Bytes(code))
                        } else if api_version >= 2 && api_version < 4 {
                            #[allow(deprecated)]
                            let info = api
                                .create_before_version_4(
                                    &id,
                                    from.unwrap_or_default(),
                                    data,
                                    value.unwrap_or_default(),
                                    gas_limit,
                                    max_fee_per_gas,
                                    max_priority_fee_per_gas,
                                    nonce,
                                    false,
                                )
                                .map_err(|err| {
                                    internal_err({
                                        let res =
                                            ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                                &["runtime error: "],
                                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                                            ));
                                        res
                                    })
                                })?
                                .map_err(|err| {
                                    internal_err({
                                        let res =
                                            ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                                &["execution fatal: "],
                                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                                            ));
                                        res
                                    })
                                })?;
                            error_on_execution_failure(&info.exit_reason, &[])?;
                            let code = api.account_code_at(&id, info.value).map_err(|err| {
                                internal_err({
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["runtime error: "],
                                        &[::core::fmt::ArgumentV1::new_debug(&err)],
                                    ));
                                    res
                                })
                            })?;
                            Ok(Bytes(code))
                        } else if api_version == 4 {
                            let access_list = access_list.unwrap_or_default();
                            let info = api
                                .create(
                                    &id,
                                    from.unwrap_or_default(),
                                    data,
                                    value.unwrap_or_default(),
                                    gas_limit,
                                    max_fee_per_gas,
                                    max_priority_fee_per_gas,
                                    nonce,
                                    false,
                                    Some(
                                        access_list
                                            .into_iter()
                                            .map(|item| (item.address, item.storage_keys))
                                            .collect(),
                                    ),
                                )
                                .map_err(|err| {
                                    internal_err({
                                        let res =
                                            ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                                &["runtime error: "],
                                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                                            ));
                                        res
                                    })
                                })?
                                .map_err(|err| {
                                    internal_err({
                                        let res =
                                            ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                                &["execution fatal: "],
                                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                                            ));
                                        res
                                    })
                                })?;
                            error_on_execution_failure(&info.exit_reason, &[])?;
                            let code = api.account_code_at(&id, info.value).map_err(|err| {
                                internal_err({
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["runtime error: "],
                                        &[::core::fmt::ArgumentV1::new_debug(&err)],
                                    ));
                                    res
                                })
                            })?;
                            Ok(Bytes(code))
                        } else {
                            Err(internal_err("failed to retrieve Runtime Api version"))
                        }
                    }
                }
            }
            pub async fn estimate_gas(
                &self,
                request: CallRequest,
                _: Option<BlockNumber>,
            ) -> Result<U256> {
                let client = Arc::clone(&self.client);
                let block_data_cache = Arc::clone(&self.block_data_cache);
                const MIN_GAS_PER_TX: U256 = U256([21_000, 0, 0, 0]);
                let best_hash = client.info().best_hash;
                let request = EGA::adapt_request(request);
                let is_simple_transfer = match &request.data {
                    None => true,
                    Some(vec) => vec.0.is_empty(),
                };
                if is_simple_transfer {
                    if let Some(to) = request.to {
                        let to_code = client
                            .runtime_api()
                            .account_code_at(&BlockId::Hash(best_hash), to)
                            .map_err(|err| {
                                internal_err({
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["runtime error: "],
                                        &[::core::fmt::ArgumentV1::new_debug(&err)],
                                    ));
                                    res
                                })
                            })?;
                        if to_code.is_empty() {
                            return Ok(MIN_GAS_PER_TX);
                        }
                    }
                }
                let (gas_price, max_fee_per_gas, max_priority_fee_per_gas) = {
                    let details = fee_details(
                        request.gas_price,
                        request.max_fee_per_gas,
                        request.max_priority_fee_per_gas,
                    )?;
                    (
                        details.gas_price,
                        details.max_fee_per_gas,
                        details.max_priority_fee_per_gas,
                    )
                };
                let block_gas_limit = {
                    let substrate_hash = client.info().best_hash;
                    let id = BlockId::Hash(substrate_hash);
                    let schema =
                        frontier_backend_client::onchain_storage_schema::<B, C, BE>(&client, id);
                    let block = block_data_cache.current_block(schema, substrate_hash).await;
                    block
                        .ok_or_else(|| internal_err("block unavailable, cannot query gas limit"))?
                        .header
                        .gas_limit
                };
                let max_gas_limit = block_gas_limit * self.execute_gas_limit_multiplier;
                let api = client.runtime_api();
                let mut highest = match request.gas {
                    Some(amount) => {
                        if amount > max_gas_limit {
                            return Err(internal_err({
                                let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                    &[
                                        "provided gas limit is too high (can be up to ",
                                        "x the block gas limit)",
                                    ],
                                    &[::core::fmt::ArgumentV1::new_display(
                                        &self.execute_gas_limit_multiplier,
                                    )],
                                ));
                                res
                            }));
                        }
                        amount
                    }
                    None => match api.gas_limit_multiplier_support(&BlockId::Hash(best_hash)) {
                        Ok(_) => max_gas_limit,
                        _ => block_gas_limit,
                    },
                };
                if let Some(from) = request.from {
                    let gas_price = gas_price.unwrap_or_default();
                    if gas_price > U256::zero() {
                        let balance = api
                            .account_basic(&BlockId::Hash(best_hash), from)
                            .map_err(|err| {
                                internal_err({
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["runtime error: "],
                                        &[::core::fmt::ArgumentV1::new_debug(&err)],
                                    ));
                                    res
                                })
                            })?
                            .balance;
                        let mut available = balance;
                        if let Some(value) = request.value {
                            if value > available {
                                return Err(internal_err("insufficient funds for transfer"));
                            }
                            available -= value;
                        }
                        let allowance = available / gas_price;
                        if highest > allowance {
                            {
                                let lvl = ::log::Level::Warn;
                                if lvl <= ::log::STATIC_MAX_LEVEL && lvl <= ::log::max_level() {
                                    ::log::__private_api_log(
                                        ::core::fmt::Arguments::new_v1(
                                            &[
                                                "Gas estimation capped by limited funds original ",
                                                " balance ",
                                                " sent ",
                                                " feecap ",
                                                " fundable ",
                                            ],
                                            &[
                                                ::core::fmt::ArgumentV1::new_display(&highest),
                                                ::core::fmt::ArgumentV1::new_display(&balance),
                                                ::core::fmt::ArgumentV1::new_display(
                                                    &request.value.unwrap_or_default(),
                                                ),
                                                ::core::fmt::ArgumentV1::new_display(&gas_price),
                                                ::core::fmt::ArgumentV1::new_display(&allowance),
                                            ],
                                        ),
                                        lvl,
                                        &(
                                            "fc_rpc::eth::execute",
                                            "fc_rpc::eth::execute",
                                            "client/rpc/src/eth/execute.rs",
                                            423u32,
                                        ),
                                        ::log::__private_api::Option::None,
                                    );
                                }
                            };
                            highest = allowance;
                        }
                    }
                }
                struct ExecutableResult {
                    data: Vec<u8>,
                    exit_reason: ExitReason,
                    used_gas: U256,
                }
                # [rustfmt :: skip] let executable = move | request , gas_limit , api_version , api : sp_api :: ApiRef < '_ , C :: Api > , estimate_mode | -> Result < ExecutableResult > { let CallRequest { from , to , gas , value , data , nonce , access_list , .. } = request ; let gas_limit = core :: cmp :: min (gas . unwrap_or (gas_limit) , gas_limit) ; let data = data . map (| d | d . 0) . unwrap_or_default () ; let (exit_reason , data , used_gas) = match to { Some (to) => { let info = if api_version == 1 { # [allow (deprecated)] api . call_before_version_2 (& BlockId :: Hash (best_hash) , from . unwrap_or_default () , to , data , value . unwrap_or_default () , gas_limit , gas_price , nonce , estimate_mode) . map_err (| err | internal_err ({ let res = :: alloc :: fmt :: format (:: core :: fmt :: Arguments :: new_v1 (& ["runtime error: "] , & [:: core :: fmt :: ArgumentV1 :: new_debug (& err)])) ; res })) ? . map_err (| err | internal_err ({ let res = :: alloc :: fmt :: format (:: core :: fmt :: Arguments :: new_v1 (& ["execution fatal: "] , & [:: core :: fmt :: ArgumentV1 :: new_debug (& err)])) ; res })) ? } else if api_version < 4 { # [allow (deprecated)] api . call_before_version_4 (& BlockId :: Hash (best_hash) , from . unwrap_or_default () , to , data , value . unwrap_or_default () , gas_limit , max_fee_per_gas , max_priority_fee_per_gas , nonce , estimate_mode) . map_err (| err | internal_err ({ let res = :: alloc :: fmt :: format (:: core :: fmt :: Arguments :: new_v1 (& ["runtime error: "] , & [:: core :: fmt :: ArgumentV1 :: new_debug (& err)])) ; res })) ? . map_err (| err | internal_err ({ let res = :: alloc :: fmt :: format (:: core :: fmt :: Arguments :: new_v1 (& ["execution fatal: "] , & [:: core :: fmt :: ArgumentV1 :: new_debug (& err)])) ; res })) ? } else { let access_list = access_list . unwrap_or_default () ; api . call (& BlockId :: Hash (best_hash) , from . unwrap_or_default () , to , data , value . unwrap_or_default () , gas_limit , max_fee_per_gas , max_priority_fee_per_gas , nonce , estimate_mode , Some (access_list . into_iter () . map (| item | (item . address , item . storage_keys)) . collect ())) . map_err (| err | internal_err ({ let res = :: alloc :: fmt :: format (:: core :: fmt :: Arguments :: new_v1 (& ["runtime error: "] , & [:: core :: fmt :: ArgumentV1 :: new_debug (& err)])) ; res })) ? . map_err (| err | internal_err ({ let res = :: alloc :: fmt :: format (:: core :: fmt :: Arguments :: new_v1 (& ["execution fatal: "] , & [:: core :: fmt :: ArgumentV1 :: new_debug (& err)])) ; res })) ? } ; (info . exit_reason , info . value , info . used_gas) } None => { let info = if api_version == 1 { # [allow (deprecated)] api . create_before_version_2 (& BlockId :: Hash (best_hash) , from . unwrap_or_default () , data , value . unwrap_or_default () , gas_limit , gas_price , nonce , estimate_mode) . map_err (| err | internal_err ({ let res = :: alloc :: fmt :: format (:: core :: fmt :: Arguments :: new_v1 (& ["runtime error: "] , & [:: core :: fmt :: ArgumentV1 :: new_debug (& err)])) ; res })) ? . map_err (| err | internal_err ({ let res = :: alloc :: fmt :: format (:: core :: fmt :: Arguments :: new_v1 (& ["execution fatal: "] , & [:: core :: fmt :: ArgumentV1 :: new_debug (& err)])) ; res })) ? } else if api_version < 4 { # [allow (deprecated)] api . create_before_version_4 (& BlockId :: Hash (best_hash) , from . unwrap_or_default () , data , value . unwrap_or_default () , gas_limit , max_fee_per_gas , max_priority_fee_per_gas , nonce , estimate_mode) . map_err (| err | internal_err ({ let res = :: alloc :: fmt :: format (:: core :: fmt :: Arguments :: new_v1 (& ["runtime error: "] , & [:: core :: fmt :: ArgumentV1 :: new_debug (& err)])) ; res })) ? . map_err (| err | internal_err ({ let res = :: alloc :: fmt :: format (:: core :: fmt :: Arguments :: new_v1 (& ["execution fatal: "] , & [:: core :: fmt :: ArgumentV1 :: new_debug (& err)])) ; res })) ? } else { let access_list = access_list . unwrap_or_default () ; api . create (& BlockId :: Hash (best_hash) , from . unwrap_or_default () , data , value . unwrap_or_default () , gas_limit , max_fee_per_gas , max_priority_fee_per_gas , nonce , estimate_mode , Some (access_list . into_iter () . map (| item | (item . address , item . storage_keys)) . collect ())) . map_err (| err | internal_err ({ let res = :: alloc :: fmt :: format (:: core :: fmt :: Arguments :: new_v1 (& ["runtime error: "] , & [:: core :: fmt :: ArgumentV1 :: new_debug (& err)])) ; res })) ? . map_err (| err | internal_err ({ let res = :: alloc :: fmt :: format (:: core :: fmt :: Arguments :: new_v1 (& ["execution fatal: "] , & [:: core :: fmt :: ArgumentV1 :: new_debug (& err)])) ; res })) ? } ; (info . exit_reason , Vec :: new () , info . used_gas) } } ; Ok (ExecutableResult { exit_reason , data , used_gas }) } ;
                let api_version = if let Ok(Some(api_version)) =
                    client
                        .runtime_api()
                        .api_version::<dyn EthereumRuntimeRPCApi<B>>(&BlockId::Hash(best_hash))
                {
                    api_version
                } else {
                    return Err(internal_err("failed to retrieve Runtime Api version"));
                };
                let cap = highest;
                let estimate_mode = !false;
                let ExecutableResult {
                    data,
                    exit_reason,
                    used_gas,
                } = executable(
                    request.clone(),
                    highest,
                    api_version,
                    client.runtime_api(),
                    estimate_mode,
                )?;
                match exit_reason {
                    ExitReason::Succeed(_) => (),
                    ExitReason::Error(ExitError::OutOfGas) => {
                        return Err(internal_err({
                            let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                &["gas required exceeds allowance "],
                                &[::core::fmt::ArgumentV1::new_display(&cap)],
                            ));
                            res
                        }))
                    }
                    ExitReason::Revert(revert) => {
                        if request.gas.is_some() || request.gas_price.is_some() {
                            let ExecutableResult {
                                data,
                                exit_reason,
                                used_gas: _,
                            } = executable(
                                request.clone(),
                                max_gas_limit,
                                api_version,
                                client.runtime_api(),
                                estimate_mode,
                            )?;
                            match exit_reason {
                                ExitReason::Succeed(_) => {
                                    return Err(internal_err({
                                        let res =
                                            ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                                &["gas required exceeds allowance "],
                                                &[::core::fmt::ArgumentV1::new_display(&cap)],
                                            ));
                                        res
                                    }))
                                }
                                other => error_on_execution_failure(&other, &data)?,
                            }
                        } else {
                            error_on_execution_failure(&ExitReason::Revert(revert), &data)?
                        }
                    }
                    other => error_on_execution_failure(&other, &data)?,
                };
                #[cfg(not(feature = "rpc-binary-search-estimate"))]
                {
                    Ok(used_gas)
                }
            }
        }
        pub fn error_on_execution_failure(reason: &ExitReason, data: &[u8]) -> Result<()> {
            match reason {
                ExitReason::Succeed(_) => Ok(()),
                ExitReason::Error(e) => {
                    if *e == ExitError::OutOfGas {
                        return Err(internal_err("out of gas"));
                    }
                    Err(crate::internal_err_with_data(
                        {
                            let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                &["evm error: "],
                                &[::core::fmt::ArgumentV1::new_debug(&e)],
                            ));
                            res
                        },
                        &[],
                    ))
                }
                ExitReason::Revert(_) => {
                    const LEN_START: usize = 36;
                    const MESSAGE_START: usize = 68;
                    let mut message =
                        "VM Exception while processing transaction: revert".to_string();
                    if data.len() > MESSAGE_START {
                        let message_len =
                            U256::from(&data[LEN_START..MESSAGE_START]).saturated_into::<usize>();
                        let message_end = MESSAGE_START.saturating_add(message_len);
                        if data.len() >= message_end {
                            let body: &[u8] = &data[MESSAGE_START..message_end];
                            if let Ok(reason) = std::str::from_utf8(body) {
                                message = {
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["", " "],
                                        &[
                                            ::core::fmt::ArgumentV1::new_display(&message),
                                            ::core::fmt::ArgumentV1::new_display(&reason),
                                        ],
                                    ));
                                    res
                                };
                            }
                        }
                    }
                    Err(crate::internal_err_with_data(message, data))
                }
                ExitReason::Fatal(e) => Err(crate::internal_err_with_data(
                    {
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &["evm fatal: "],
                            &[::core::fmt::ArgumentV1::new_debug(&e)],
                        ));
                        res
                    },
                    &[],
                )),
            }
        }
        struct FeeDetails {
            gas_price: Option<U256>,
            max_fee_per_gas: Option<U256>,
            max_priority_fee_per_gas: Option<U256>,
        }
        fn fee_details(
            request_gas_price: Option<U256>,
            request_max_fee: Option<U256>,
            request_priority: Option<U256>,
        ) -> Result<FeeDetails> {
            match (request_gas_price, request_max_fee, request_priority) {
                (gas_price, None, None) => {
                    let gas_price = if gas_price.unwrap_or_default().is_zero() {
                        None
                    } else {
                        gas_price
                    };
                    Ok(FeeDetails {
                        gas_price,
                        max_fee_per_gas: gas_price,
                        max_priority_fee_per_gas: gas_price,
                    })
                }
                (_, max_fee, max_priority) => {
                    let max_fee = if max_fee.unwrap_or_default().is_zero() {
                        None
                    } else {
                        max_fee
                    };
                    if let Some(max_priority) = max_priority {
                        let max_fee = max_fee.unwrap_or_default();
                        if max_priority > max_fee {
                            return Err (internal_err ("Invalid input: `max_priority_fee_per_gas` greater than `max_fee_per_gas`")) ;
                        }
                    }
                    Ok(FeeDetails {
                        gas_price: max_fee,
                        max_fee_per_gas: max_fee,
                        max_priority_fee_per_gas: max_priority,
                    })
                }
            }
        }
    }
    mod fee {
        use ethereum_types::{H256, U256};
        use jsonrpsee::core::RpcResult as Result;
        use sc_client_api::backend::{Backend, StateBackend, StorageProvider};
        use sc_network_common::ExHashT;
        use sc_transaction_pool::ChainApi;
        use sp_api::ProvideRuntimeApi;
        use sp_blockchain::HeaderBackend;
        use sp_runtime::{
            generic::BlockId,
            traits::{BlakeTwo256, Block as BlockT, UniqueSaturatedInto},
        };
        use fc_rpc_core::types::*;
        use fp_rpc::EthereumRuntimeRPCApi;
        use crate::{eth::Eth, frontier_backend_client, internal_err};
        impl<B, C, P, CT, BE, H: ExHashT, A: ChainApi, EGA> Eth<B, C, P, CT, BE, H, A, EGA>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: ProvideRuntimeApi<B> + StorageProvider<B, BE>,
            C: HeaderBackend<B> + Send + Sync + 'static,
            C::Api: EthereumRuntimeRPCApi<B>,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
        {
            pub fn gas_price(&self) -> Result<U256> {
                let block = BlockId::Hash(self.client.info().best_hash);
                self.client.runtime_api().gas_price(&block).map_err(|err| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &["fetch runtime chain id failed: "],
                            &[::core::fmt::ArgumentV1::new_debug(&err)],
                        ));
                        res
                    })
                })
            }
            pub fn fee_history(
                &self,
                block_count: U256,
                newest_block: BlockNumber,
                reward_percentiles: Option<Vec<f64>>,
            ) -> Result<FeeHistory> {
                let range_limit = U256::from(1024);
                let block_count = if block_count > range_limit {
                    range_limit.as_u64()
                } else {
                    block_count.as_u64()
                };
                if let Ok(Some(id)) = frontier_backend_client::native_block_id::<B, C>(
                    self.client.as_ref(),
                    self.backend.as_ref(),
                    Some(newest_block),
                ) {
                    let Ok (number) = self . client . expect_block_number_from_id (& id) else { return Err (internal_err ({ let res = :: alloc :: fmt :: format (:: core :: fmt :: Arguments :: new_v1 (& ["Failed to retrieve block number at "] , & [:: core :: fmt :: ArgumentV1 :: new_display (& id)])) ; res })) ; } ;
                    let highest = UniqueSaturatedInto::<u64>::unique_saturated_into(number);
                    let lowest = highest.saturating_sub(block_count.saturating_sub(1));
                    let best_number = UniqueSaturatedInto::<u64>::unique_saturated_into(
                        self.client.info().best_number,
                    );
                    if lowest < best_number.saturating_sub(self.fee_history_cache_limit) {
                        return Err(internal_err("Block range out of bounds."));
                    }
                    if let Ok(fee_history_cache) = &self.fee_history_cache.lock() {
                        let mut response = FeeHistory {
                            oldest_block: U256::from(lowest),
                            base_fee_per_gas: Vec::new(),
                            gas_used_ratio: Vec::new(),
                            reward: None,
                        };
                        let mut rewards = Vec::new();
                        for n in lowest..highest + 1 {
                            if let Some(block) = fee_history_cache.get(&n) {
                                response.base_fee_per_gas.push(U256::from(block.base_fee));
                                response.gas_used_ratio.push(block.gas_used_ratio);
                                if let Some(ref requested_percentiles) = reward_percentiles {
                                    let mut block_rewards = Vec::new();
                                    let resolution_per_percentile: f64 = 2.0;
                                    for p in requested_percentiles {
                                        let p = p.clamp(0.0, 100.0);
                                        let index =
                                            ((p.round() / 2f64) * 2f64) * resolution_per_percentile;
                                        let reward =
                                            if let Some(r) = block.rewards.get(index as usize) {
                                                U256::from(*r)
                                            } else {
                                                U256::zero()
                                            };
                                        block_rewards.push(reward);
                                    }
                                    if !block_rewards.is_empty() {
                                        rewards.push(block_rewards);
                                    }
                                }
                            }
                        }
                        if rewards.len() > 0 {
                            response.reward = Some(rewards);
                        }
                        if let (Some(last_gas_used), Some(last_fee_per_gas)) = (
                            response.gas_used_ratio.last(),
                            response.base_fee_per_gas.last(),
                        ) {
                            let schema = frontier_backend_client::onchain_storage_schema::<B, C, BE>(
                                self.client.as_ref(),
                                id,
                            );
                            let handler = self
                                .overrides
                                .schemas
                                .get(&schema)
                                .unwrap_or(&self.overrides.fallback);
                            let default_elasticity = sp_runtime::Permill::from_parts(125_000);
                            let elasticity = handler
                                .elasticity(&id)
                                .unwrap_or(default_elasticity)
                                .deconstruct();
                            let elasticity = elasticity as f64 / 1_000_000f64;
                            let last_fee_per_gas = last_fee_per_gas.as_u64() as f64;
                            if last_gas_used > &0.5 {
                                let increase = ((last_gas_used - 0.5) * 2f64) * elasticity;
                                let new_base_fee =
                                    (last_fee_per_gas + (last_fee_per_gas * increase)) as u64;
                                response.base_fee_per_gas.push(U256::from(new_base_fee));
                            } else if last_gas_used < &0.5 {
                                let increase = ((0.5 - last_gas_used) * 2f64) * elasticity;
                                let new_base_fee =
                                    (last_fee_per_gas - (last_fee_per_gas * increase)) as u64;
                                response.base_fee_per_gas.push(U256::from(new_base_fee));
                            } else {
                                response
                                    .base_fee_per_gas
                                    .push(U256::from(last_fee_per_gas as u64));
                            }
                        }
                        return Ok(response);
                    } else {
                        return Err(internal_err("Failed to read fee history cache."));
                    }
                }
                Err(internal_err({
                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                        &["Failed to retrieve requested block ", "."],
                        &[::core::fmt::ArgumentV1::new_debug(&newest_block)],
                    ));
                    res
                }))
            }
            pub fn max_priority_fee_per_gas(&self) -> Result<U256> {
                let at_percentile = 60;
                let block_count = 20;
                let index = (at_percentile * 2) as usize;
                let highest = UniqueSaturatedInto::<u64>::unique_saturated_into(
                    self.client.info().best_number,
                );
                let lowest = highest.saturating_sub(block_count - 1);
                let mut rewards = Vec::new();
                if let Ok(fee_history_cache) = &self.fee_history_cache.lock() {
                    for n in lowest..highest + 1 {
                        if let Some(block) = fee_history_cache.get(&n) {
                            let reward = if let Some(r) = block.rewards.get(index) {
                                U256::from(*r)
                            } else {
                                U256::zero()
                            };
                            rewards.push(reward);
                        }
                    }
                } else {
                    return Err(internal_err("Failed to read fee oracle cache."));
                }
                Ok(*rewards.iter().min().unwrap_or(&U256::zero()))
            }
        }
    }
    mod filter {
        use std::{marker::PhantomData, sync::Arc, time};
        use ethereum::BlockV2 as EthereumBlock;
        use ethereum_types::{H256, U256};
        use jsonrpsee::core::{async_trait, RpcResult as Result};
        use sc_client_api::backend::{Backend, StateBackend, StorageProvider};
        use sp_api::ProvideRuntimeApi;
        use sp_blockchain::HeaderBackend;
        use sp_core::hashing::keccak_256;
        use sp_runtime::{
            generic::BlockId,
            traits::{
                BlakeTwo256, Block as BlockT, NumberFor, One, Saturating, UniqueSaturatedInto,
            },
        };
        use fc_rpc_core::{types::*, EthFilterApiServer};
        use fp_rpc::{EthereumRuntimeRPCApi, TransactionStatus};
        use crate::{eth::cache::EthBlockDataCacheTask, frontier_backend_client, internal_err};
        pub struct EthFilter<B: BlockT, C, BE> {
            client: Arc<C>,
            backend: Arc<fc_db::Backend<B>>,
            filter_pool: FilterPool,
            max_stored_filters: usize,
            max_past_logs: u32,
            block_data_cache: Arc<EthBlockDataCacheTask<B>>,
            _marker: PhantomData<BE>,
        }
        impl<B: BlockT, C, BE> EthFilter<B, C, BE> {
            pub fn new(
                client: Arc<C>,
                backend: Arc<fc_db::Backend<B>>,
                filter_pool: FilterPool,
                max_stored_filters: usize,
                max_past_logs: u32,
                block_data_cache: Arc<EthBlockDataCacheTask<B>>,
            ) -> Self {
                Self {
                    client,
                    backend,
                    filter_pool,
                    max_stored_filters,
                    max_past_logs,
                    block_data_cache,
                    _marker: PhantomData,
                }
            }
        }
        impl<B, C, BE> EthFilter<B, C, BE>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: HeaderBackend<B> + Send + Sync + 'static,
        {
            fn create_filter(&self, filter_type: FilterType) -> Result<U256> {
                let block_number = UniqueSaturatedInto::<u64>::unique_saturated_into(
                    self.client.info().best_number,
                );
                let pool = self.filter_pool.clone();
                let response = if let Ok(locked) = &mut pool.lock() {
                    if locked.len() >= self.max_stored_filters {
                        return Err(internal_err({
                            let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                &["Filter pool is full (limit ", ")."],
                                &[::core::fmt::ArgumentV1::new_debug(&self.max_stored_filters)],
                            ));
                            res
                        }));
                    }
                    let last_key = match {
                        let mut iter = locked.iter();
                        iter.next_back()
                    } {
                        Some((k, _)) => *k,
                        None => U256::zero(),
                    };
                    let key = last_key.checked_add(U256::one()).unwrap();
                    locked.insert(
                        key,
                        FilterPoolItem {
                            last_poll: BlockNumber::Num(block_number),
                            filter_type,
                            at_block: block_number,
                        },
                    );
                    Ok(key)
                } else {
                    Err(internal_err("Filter pool is not available."))
                };
                response
            }
        }
        impl<B, C, BE> EthFilterApiServer for EthFilter<B, C, BE>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: ProvideRuntimeApi<B> + StorageProvider<B, BE>,
            C: HeaderBackend<B> + Send + Sync + 'static,
            C::Api: EthereumRuntimeRPCApi<B>,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
        {
            fn new_filter(&self, filter: Filter) -> Result<U256> {
                self.create_filter(FilterType::Log(filter))
            }
            fn new_block_filter(&self) -> Result<U256> {
                self.create_filter(FilterType::Block)
            }
            fn new_pending_transaction_filter(&self) -> Result<U256> {
                Err(internal_err("Method not available."))
            }
            #[allow(
                clippy::let_unit_value,
                clippy::no_effect_underscore_binding,
                clippy::shadow_same,
                clippy::type_complexity,
                clippy::type_repetition_in_bounds,
                clippy::used_underscore_binding
            )]
            fn filter_changes<'life0, 'async_trait>(
                &'life0 self,
                index: Index,
            ) -> ::core::pin::Pin<
                Box<
                    dyn ::core::future::Future<Output = Result<FilterChanges>>
                        + ::core::marker::Send
                        + 'async_trait,
                >,
            >
            where
                'life0: 'async_trait,
                Self: 'async_trait,
            {
                Box::pin(async move {
                    if let ::core::option::Option::Some(__ret) =
                        ::core::option::Option::None::<Result<FilterChanges>>
                    {
                        return __ret;
                    }
                    let __self = self;
                    let index = index;
                    let __ret: Result<FilterChanges> = {
                        enum FuturePath<B: BlockT> {
                            Block {
                                last: u64,
                                next: u64,
                            },
                            Log {
                                filter: Filter,
                                from_number: NumberFor<B>,
                                current_number: NumberFor<B>,
                            },
                            Error(jsonrpsee::core::Error),
                        }
                        let key = U256::from(index.value());
                        let block_number = UniqueSaturatedInto::<u64>::unique_saturated_into(
                            __self.client.info().best_number,
                        );
                        let pool = __self.filter_pool.clone();
                        let path = if let Ok(locked) = &mut pool.lock() {
                            if let Some(pool_item) = locked.get(&key).cloned() {
                                match &pool_item.filter_type {
                                    FilterType::Block => {
                                        let last = pool_item.last_poll.to_min_block_num().unwrap();
                                        let next = block_number + 1;
                                        locked.insert(
                                            key,
                                            FilterPoolItem {
                                                last_poll: BlockNumber::Num(next),
                                                filter_type: pool_item.filter_type.clone(),
                                                at_block: pool_item.at_block,
                                            },
                                        );
                                        FuturePath::<B>::Block { last, next }
                                    }
                                    FilterType::Log(filter) => {
                                        locked.insert(
                                            key,
                                            FilterPoolItem {
                                                last_poll: BlockNumber::Num(block_number + 1),
                                                filter_type: pool_item.filter_type.clone(),
                                                at_block: pool_item.at_block,
                                            },
                                        );
                                        let best_number = __self.client.info().best_number;
                                        let mut current_number = filter
                                            .to_block
                                            .and_then(|v| v.to_min_block_num())
                                            .map(|s| s.unique_saturated_into())
                                            .unwrap_or(best_number);
                                        if current_number > best_number {
                                            current_number = best_number;
                                        }
                                        let last_poll = pool_item
                                            .last_poll
                                            .to_min_block_num()
                                            .unwrap()
                                            .unique_saturated_into();
                                        let filter_from = filter
                                            .from_block
                                            .and_then(|v| v.to_min_block_num())
                                            .map(|s| s.unique_saturated_into())
                                            .unwrap_or(last_poll);
                                        let from_number = std::cmp::max(last_poll, filter_from);
                                        FuturePath::Log {
                                            filter: filter.clone(),
                                            from_number,
                                            current_number,
                                        }
                                    }
                                    _ => FuturePath::Error(internal_err("Method not available.")),
                                }
                            } else {
                                FuturePath::Error(internal_err({
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["Filter id ", " does not exist."],
                                        &[::core::fmt::ArgumentV1::new_debug(&key)],
                                    ));
                                    res
                                }))
                            }
                        } else {
                            FuturePath::Error(internal_err("Filter pool is not available."))
                        };
                        let client = Arc::clone(&__self.client);
                        let block_data_cache = Arc::clone(&__self.block_data_cache);
                        let max_past_logs = __self.max_past_logs;
                        match path {
                            FuturePath::Error(err) => Err(err),
                            FuturePath::Block { last, next } => {
                                let mut ethereum_hashes: Vec<H256> = Vec::new();
                                for n in last..next {
                                    let id = BlockId::Number(n.unique_saturated_into());
                                    let substrate_hash =
                                        client.expect_block_hash_from_id(&id).map_err(|_| {
                                            internal_err({
                                                let res = ::alloc::fmt::format(
                                                    ::core::fmt::Arguments::new_v1(
                                                        &["Expect block number from id: "],
                                                        &[::core::fmt::ArgumentV1::new_display(
                                                            &id,
                                                        )],
                                                    ),
                                                );
                                                res
                                            })
                                        })?;
                                    let schema = frontier_backend_client::onchain_storage_schema::<
                                        B,
                                        C,
                                        BE,
                                    >(
                                        client.as_ref(), id
                                    );
                                    let block = block_data_cache
                                        .current_block(schema, substrate_hash)
                                        .await;
                                    if let Some(block) = block {
                                        ethereum_hashes.push(block.header.hash())
                                    }
                                }
                                Ok(FilterChanges::Hashes(ethereum_hashes))
                            }
                            FuturePath::Log {
                                filter,
                                from_number,
                                current_number,
                            } => {
                                let mut ret: Vec<Log> = Vec::new();
                                let _ = filter_range_logs(
                                    client.as_ref(),
                                    &block_data_cache,
                                    &mut ret,
                                    max_past_logs,
                                    &filter,
                                    from_number,
                                    current_number,
                                )
                                .await?;
                                Ok(FilterChanges::Logs(ret))
                            }
                        }
                    };
                    #[allow(unreachable_code)]
                    __ret
                })
            }
            #[allow(
                clippy::let_unit_value,
                clippy::no_effect_underscore_binding,
                clippy::shadow_same,
                clippy::type_complexity,
                clippy::type_repetition_in_bounds,
                clippy::used_underscore_binding
            )]
            fn filter_logs<'life0, 'async_trait>(
                &'life0 self,
                index: Index,
            ) -> ::core::pin::Pin<
                Box<
                    dyn ::core::future::Future<Output = Result<Vec<Log>>>
                        + ::core::marker::Send
                        + 'async_trait,
                >,
            >
            where
                'life0: 'async_trait,
                Self: 'async_trait,
            {
                Box::pin(async move {
                    if let ::core::option::Option::Some(__ret) =
                        ::core::option::Option::None::<Result<Vec<Log>>>
                    {
                        return __ret;
                    }
                    let __self = self;
                    let index = index;
                    let __ret: Result<Vec<Log>> = {
                        let key = U256::from(index.value());
                        let pool = __self.filter_pool.clone();
                        let filter_result: Result<Filter> = (|| {
                            let pool = pool
                                .lock()
                                .map_err(|_| internal_err("Filter pool is not available."))?;
                            let pool_item = pool.get(&key).ok_or_else(|| {
                                internal_err({
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["Filter id ", " does not exist."],
                                        &[::core::fmt::ArgumentV1::new_debug(&key)],
                                    ));
                                    res
                                })
                            })?;
                            match &pool_item.filter_type {
                                FilterType::Log(filter) => Ok(filter.clone()),
                                _ => Err(internal_err({
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["Filter id ", " is not a Log filter."],
                                        &[::core::fmt::ArgumentV1::new_debug(&key)],
                                    ));
                                    res
                                })),
                            }
                        })();
                        let client = Arc::clone(&__self.client);
                        let block_data_cache = Arc::clone(&__self.block_data_cache);
                        let max_past_logs = __self.max_past_logs;
                        let filter = filter_result?;
                        let best_number = client.info().best_number;
                        let mut current_number = filter
                            .to_block
                            .and_then(|v| v.to_min_block_num())
                            .map(|s| s.unique_saturated_into())
                            .unwrap_or(best_number);
                        if current_number > best_number {
                            current_number = best_number;
                        }
                        let from_number = filter
                            .from_block
                            .and_then(|v| v.to_min_block_num())
                            .map(|s| s.unique_saturated_into())
                            .unwrap_or(best_number);
                        let mut ret: Vec<Log> = Vec::new();
                        let _ = filter_range_logs(
                            client.as_ref(),
                            &block_data_cache,
                            &mut ret,
                            max_past_logs,
                            &filter,
                            from_number,
                            current_number,
                        )
                        .await?;
                        Ok(ret)
                    };
                    #[allow(unreachable_code)]
                    __ret
                })
            }
            fn uninstall_filter(&self, index: Index) -> Result<bool> {
                let key = U256::from(index.value());
                let pool = self.filter_pool.clone();
                let response = if let Ok(locked) = &mut pool.lock() {
                    if locked.remove(&key).is_some() {
                        Ok(true)
                    } else {
                        Err(internal_err({
                            let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                &["Filter id ", " does not exist."],
                                &[::core::fmt::ArgumentV1::new_debug(&key)],
                            ));
                            res
                        }))
                    }
                } else {
                    Err(internal_err("Filter pool is not available."))
                };
                response
            }
            #[allow(
                clippy::let_unit_value,
                clippy::no_effect_underscore_binding,
                clippy::shadow_same,
                clippy::type_complexity,
                clippy::type_repetition_in_bounds,
                clippy::used_underscore_binding
            )]
            fn logs<'life0, 'async_trait>(
                &'life0 self,
                filter: Filter,
            ) -> ::core::pin::Pin<
                Box<
                    dyn ::core::future::Future<Output = Result<Vec<Log>>>
                        + ::core::marker::Send
                        + 'async_trait,
                >,
            >
            where
                'life0: 'async_trait,
                Self: 'async_trait,
            {
                Box::pin(async move {
                    if let ::core::option::Option::Some(__ret) =
                        ::core::option::Option::None::<Result<Vec<Log>>>
                    {
                        return __ret;
                    }
                    let __self = self;
                    let filter = filter;
                    let __ret: Result<Vec<Log>> = {
                        let client = Arc::clone(&__self.client);
                        let block_data_cache = Arc::clone(&__self.block_data_cache);
                        let backend = Arc::clone(&__self.backend);
                        let max_past_logs = __self.max_past_logs;
                        let mut ret: Vec<Log> = Vec::new();
                        if let Some(hash) = filter.block_hash {
                            let id = match frontier_backend_client::load_hash::<B, C>(
                                client.as_ref(),
                                backend.as_ref(),
                                hash,
                            )
                            .map_err(|err| {
                                internal_err({
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &[""],
                                        &[::core::fmt::ArgumentV1::new_debug(&err)],
                                    ));
                                    res
                                })
                            })? {
                                Some(hash) => hash,
                                _ => return Ok(Vec::new()),
                            };
                            let substrate_hash =
                                client.expect_block_hash_from_id(&id).map_err(|_| {
                                    internal_err({
                                        let res =
                                            ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                                &["Expect block number from id: "],
                                                &[::core::fmt::ArgumentV1::new_display(&id)],
                                            ));
                                        res
                                    })
                                })?;
                            let schema = frontier_backend_client::onchain_storage_schema::<B, C, BE>(
                                client.as_ref(),
                                id,
                            );
                            let block =
                                block_data_cache.current_block(schema, substrate_hash).await;
                            let statuses = block_data_cache
                                .current_transaction_statuses(schema, substrate_hash)
                                .await;
                            if let (Some(block), Some(statuses)) = (block, statuses) {
                                filter_block_logs(&mut ret, &filter, block, statuses);
                            }
                        } else {
                            let best_number = client.info().best_number;
                            let mut current_number = filter
                                .to_block
                                .and_then(|v| v.to_min_block_num())
                                .map(|s| s.unique_saturated_into())
                                .unwrap_or(best_number);
                            if current_number > best_number {
                                current_number = best_number;
                            }
                            let from_number = filter
                                .from_block
                                .and_then(|v| v.to_min_block_num())
                                .map(|s| s.unique_saturated_into())
                                .unwrap_or(best_number);
                            let _ = filter_range_logs(
                                client.as_ref(),
                                &block_data_cache,
                                &mut ret,
                                max_past_logs,
                                &filter,
                                from_number,
                                current_number,
                            )
                            .await?;
                        }
                        Ok(ret)
                    };
                    #[allow(unreachable_code)]
                    __ret
                })
            }
        }
        async fn filter_range_logs<B: BlockT, C, BE>(
            client: &C,
            block_data_cache: &EthBlockDataCacheTask<B>,
            ret: &mut Vec<Log>,
            max_past_logs: u32,
            filter: &Filter,
            from: NumberFor<B>,
            to: NumberFor<B>,
        ) -> Result<()>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: ProvideRuntimeApi<B> + StorageProvider<B, BE>,
            C: HeaderBackend<B> + Send + Sync + 'static,
            C::Api: EthereumRuntimeRPCApi<B>,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
        {
            let max_duration = time::Duration::from_secs(10);
            let begin_request = time::Instant::now();
            let mut current_number = from;
            let topics_input = if filter.topics.is_some() {
                let filtered_params = FilteredParams::new(Some(filter.clone()));
                Some(filtered_params.flat_topics)
            } else {
                None
            };
            let address_bloom_filter = FilteredParams::adresses_bloom_filter(&filter.address);
            let topics_bloom_filter = FilteredParams::topics_bloom_filter(&topics_input);
            while current_number <= to {
                let id = BlockId::Number(current_number);
                let substrate_hash = client.expect_block_hash_from_id(&id).map_err(|_| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &["Expect block number from id: "],
                            &[::core::fmt::ArgumentV1::new_display(&id)],
                        ));
                        res
                    })
                })?;
                let schema =
                    frontier_backend_client::onchain_storage_schema::<B, C, BE>(client, id);
                let block = block_data_cache.current_block(schema, substrate_hash).await;
                if let Some(block) = block {
                    if FilteredParams::address_in_bloom(
                        block.header.logs_bloom,
                        &address_bloom_filter,
                    ) && FilteredParams::topics_in_bloom(
                        block.header.logs_bloom,
                        &topics_bloom_filter,
                    ) {
                        let statuses = block_data_cache
                            .current_transaction_statuses(schema, substrate_hash)
                            .await;
                        if let Some(statuses) = statuses {
                            filter_block_logs(ret, filter, block, statuses);
                        }
                    }
                }
                if ret.len() as u32 > max_past_logs {
                    return Err(internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &["query returned more than ", " results"],
                            &[::core::fmt::ArgumentV1::new_display(&max_past_logs)],
                        ));
                        res
                    }));
                }
                if begin_request.elapsed() > max_duration {
                    return Err(internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &["query timeout of ", " seconds exceeded"],
                            &[::core::fmt::ArgumentV1::new_display(
                                &max_duration.as_secs(),
                            )],
                        ));
                        res
                    }));
                }
                if current_number == to {
                    break;
                } else {
                    current_number = current_number.saturating_add(One::one());
                }
            }
            Ok(())
        }
        fn filter_block_logs<'a>(
            ret: &'a mut Vec<Log>,
            filter: &'a Filter,
            block: EthereumBlock,
            transaction_statuses: Vec<TransactionStatus>,
        ) -> &'a Vec<Log> {
            let params = FilteredParams::new(Some(filter.clone()));
            let mut block_log_index: u32 = 0;
            let block_hash = H256::from(keccak_256(&rlp::encode(&block.header)));
            for status in transaction_statuses.iter() {
                let logs = status.logs.clone();
                let mut transaction_log_index: u32 = 0;
                let transaction_hash = status.transaction_hash;
                for ethereum_log in logs {
                    let mut log = Log {
                        address: ethereum_log.address,
                        topics: ethereum_log.topics.clone(),
                        data: Bytes(ethereum_log.data.clone()),
                        block_hash: None,
                        block_number: None,
                        transaction_hash: None,
                        transaction_index: None,
                        log_index: None,
                        transaction_log_index: None,
                        removed: false,
                    };
                    let mut add: bool = true;
                    match (filter.address.clone(), filter.topics.clone()) {
                        (Some(_), Some(_)) => {
                            if !params.filter_address(&log) || !params.filter_topics(&log) {
                                add = false;
                            }
                        }
                        (Some(_), _) => {
                            if !params.filter_address(&log) {
                                add = false;
                            }
                        }
                        (_, Some(_)) => {
                            if !params.filter_topics(&log) {
                                add = false;
                            }
                        }
                        _ => {}
                    }
                    if add {
                        log.block_hash = Some(block_hash);
                        log.block_number = Some(block.header.number);
                        log.transaction_hash = Some(transaction_hash);
                        log.transaction_index = Some(U256::from(status.transaction_index));
                        log.log_index = Some(U256::from(block_log_index));
                        log.transaction_log_index = Some(U256::from(transaction_log_index));
                        ret.push(log);
                    }
                    transaction_log_index += 1;
                    block_log_index += 1;
                }
            }
            ret
        }
    }
    pub mod format {
        use sc_transaction_pool_api::error::{Error as PError, IntoPoolError};
        use sp_runtime::transaction_validity::InvalidTransaction;
        use fp_ethereum::TransactionValidationError as VError;
        pub struct Geth;
        impl Geth {
            pub fn pool_error(err: impl IntoPoolError) -> String {
                match err.into_pool_error() {
                    Ok(PError::AlreadyImported(_)) => "already known".to_string(),
                    Ok(PError::TemporarilyBanned) => "already known".into(),
                    Ok(PError::TooLowPriority { .. }) => {
                        "replacement transaction underpriced".into()
                    }
                    Ok(PError::InvalidTransaction(inner)) => match inner {
                        InvalidTransaction::Stale => "nonce too low".into(),
                        InvalidTransaction::Payment => {
                            "insufficient funds for gas * price + value".into()
                        }
                        InvalidTransaction::ExhaustsResources => "exceeds block gas limit".into(),
                        InvalidTransaction::Custom(inner) => match inner.into() {
                            VError::UnknownError => "unknown error".into(),
                            VError::InvalidChainId => "invalid chain id".into(),
                            VError::InvalidSignature => "invalid sender".into(),
                            VError::GasLimitTooLow => "intrinsic gas too low".into(),
                            VError::GasLimitTooHigh => "exceeds block gas limit".into(),
                            VError::MaxFeePerGasTooLow => {
                                "max priority fee per gas higher than max fee per gas".into()
                            }
                        },
                        _ => "unknown error".into(),
                    },
                    err => {
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &["submit transaction to pool failed: "],
                            &[::core::fmt::ArgumentV1::new_debug(&err)],
                        ));
                        res
                    }
                }
            }
        }
    }
    mod mining {
        use ethereum_types::{H256, H64, U256};
        use jsonrpsee::core::RpcResult as Result;
        use sc_network_common::ExHashT;
        use sc_transaction_pool::ChainApi;
        use sp_runtime::traits::Block as BlockT;
        use fc_rpc_core::types::*;
        use crate::eth::Eth;
        impl<B: BlockT, C, P, CT, BE, H: ExHashT, A: ChainApi, EGA> Eth<B, C, P, CT, BE, H, A, EGA> {
            pub fn is_mining(&self) -> Result<bool> {
                Ok(self.is_authority)
            }
            pub fn hashrate(&self) -> Result<U256> {
                Ok(U256::zero())
            }
            pub fn work(&self) -> Result<Work> {
                Ok(Work::default())
            }
            pub fn submit_hashrate(&self, _: U256, _: H256) -> Result<bool> {
                Ok(false)
            }
            pub fn submit_work(&self, _: H64, _: H256, _: H256) -> Result<bool> {
                Ok(false)
            }
        }
    }
    mod state {
        use ethereum_types::{H160, H256, U256};
        use jsonrpsee::core::RpcResult as Result;
        use scale_codec::Encode;
        use sc_client_api::backend::{Backend, StateBackend, StorageProvider};
        use sc_network_common::ExHashT;
        use sc_transaction_pool::ChainApi;
        use sc_transaction_pool_api::{InPoolTransaction, TransactionPool};
        use sp_api::ProvideRuntimeApi;
        use sp_block_builder::BlockBuilder as BlockBuilderApi;
        use sp_blockchain::HeaderBackend;
        use sp_runtime::{
            generic::BlockId,
            traits::{BlakeTwo256, Block as BlockT},
        };
        use fc_rpc_core::types::*;
        use fp_rpc::EthereumRuntimeRPCApi;
        use crate::{
            eth::{pending_runtime_api, Eth},
            frontier_backend_client, internal_err,
        };
        impl<B, C, P, CT, BE, H: ExHashT, A: ChainApi, EGA> Eth<B, C, P, CT, BE, H, A, EGA>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: ProvideRuntimeApi<B> + StorageProvider<B, BE>,
            C: HeaderBackend<B> + Send + Sync + 'static,
            C::Api: BlockBuilderApi<B> + EthereumRuntimeRPCApi<B>,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
            P: TransactionPool<Block = B> + Send + Sync + 'static,
            A: ChainApi<Block = B> + 'static,
        {
            pub fn balance(&self, address: H160, number: Option<BlockNumber>) -> Result<U256> {
                let number = number.unwrap_or(BlockNumber::Latest);
                if number == BlockNumber::Pending {
                    let api = pending_runtime_api(self.client.as_ref(), self.graph.as_ref())?;
                    Ok(api
                        .account_basic(&BlockId::Hash(self.client.info().best_hash), address)
                        .map_err(|err| {
                            internal_err({
                                let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                    &["fetch runtime chain id failed: "],
                                    &[::core::fmt::ArgumentV1::new_debug(&err)],
                                ));
                                res
                            })
                        })?
                        .balance)
                } else if let Ok(Some(id)) = frontier_backend_client::native_block_id::<B, C>(
                    self.client.as_ref(),
                    self.backend.as_ref(),
                    Some(number),
                ) {
                    Ok(self
                        .client
                        .runtime_api()
                        .account_basic(&id, address)
                        .map_err(|err| {
                            internal_err({
                                let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                    &["fetch runtime chain id failed: "],
                                    &[::core::fmt::ArgumentV1::new_debug(&err)],
                                ));
                                res
                            })
                        })?
                        .balance)
                } else {
                    Ok(U256::zero())
                }
            }
            pub fn storage_at(
                &self,
                address: H160,
                index: U256,
                number: Option<BlockNumber>,
            ) -> Result<H256> {
                let number = number.unwrap_or(BlockNumber::Latest);
                if number == BlockNumber::Pending {
                    let api = pending_runtime_api(self.client.as_ref(), self.graph.as_ref())?;
                    Ok(api
                        .storage_at(&BlockId::Hash(self.client.info().best_hash), address, index)
                        .unwrap_or_default())
                } else if let Ok(Some(id)) = frontier_backend_client::native_block_id::<B, C>(
                    self.client.as_ref(),
                    self.backend.as_ref(),
                    Some(number),
                ) {
                    let schema = frontier_backend_client::onchain_storage_schema::<B, C, BE>(
                        self.client.as_ref(),
                        id,
                    );
                    Ok(self
                        .overrides
                        .schemas
                        .get(&schema)
                        .unwrap_or(&self.overrides.fallback)
                        .storage_at(&id, address, index)
                        .unwrap_or_default())
                } else {
                    Ok(H256::default())
                }
            }
            pub fn transaction_count(
                &self,
                address: H160,
                number: Option<BlockNumber>,
            ) -> Result<U256> {
                if let Some(BlockNumber::Pending) = number {
                    let block = BlockId::Hash(self.client.info().best_hash);
                    let nonce = self
                        .client
                        .runtime_api()
                        .account_basic(&block, address)
                        .map_err(|err| {
                            internal_err({
                                let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                    &["fetch runtime account basic failed: "],
                                    &[::core::fmt::ArgumentV1::new_debug(&err)],
                                ));
                                res
                            })
                        })?
                        .nonce;
                    let mut current_nonce = nonce;
                    let mut current_tag = (address, nonce).encode();
                    for tx in self.pool.ready() {
                        if tx.provides().get(0) == Some(&current_tag) {
                            current_nonce = current_nonce.saturating_add(1.into());
                            current_tag = (address, current_nonce).encode();
                        }
                    }
                    return Ok(current_nonce);
                }
                let id = match frontier_backend_client::native_block_id::<B, C>(
                    self.client.as_ref(),
                    self.backend.as_ref(),
                    number,
                )? {
                    Some(id) => id,
                    None => return Ok(U256::zero()),
                };
                Ok(self
                    .client
                    .runtime_api()
                    .account_basic(&id, address)
                    .map_err(|err| {
                        internal_err({
                            let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                &["fetch runtime account basic failed: "],
                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                            ));
                            res
                        })
                    })?
                    .nonce)
            }
            pub fn code_at(&self, address: H160, number: Option<BlockNumber>) -> Result<Bytes> {
                let number = number.unwrap_or(BlockNumber::Latest);
                if number == BlockNumber::Pending {
                    let api = pending_runtime_api(self.client.as_ref(), self.graph.as_ref())?;
                    Ok(api
                        .account_code_at(&BlockId::Hash(self.client.info().best_hash), address)
                        .unwrap_or_default()
                        .into())
                } else if let Ok(Some(id)) = frontier_backend_client::native_block_id::<B, C>(
                    self.client.as_ref(),
                    self.backend.as_ref(),
                    Some(number),
                ) {
                    let schema = frontier_backend_client::onchain_storage_schema::<B, C, BE>(
                        self.client.as_ref(),
                        id,
                    );
                    Ok(self
                        .overrides
                        .schemas
                        .get(&schema)
                        .unwrap_or(&self.overrides.fallback)
                        .account_code_at(&id, address)
                        .unwrap_or_default()
                        .into())
                } else {
                    Ok(Bytes(::alloc::vec::Vec::new()))
                }
            }
        }
    }
    mod submit {
        use ethereum_types::H256;
        use futures::future::TryFutureExt;
        use jsonrpsee::core::RpcResult as Result;
        use sc_client_api::backend::{Backend, StateBackend, StorageProvider};
        use sc_network_common::ExHashT;
        use sc_transaction_pool::ChainApi;
        use sc_transaction_pool_api::TransactionPool;
        use sp_api::{ApiExt, ProvideRuntimeApi};
        use sp_block_builder::BlockBuilder as BlockBuilderApi;
        use sp_blockchain::HeaderBackend;
        use sp_runtime::{
            generic::BlockId,
            traits::{BlakeTwo256, Block as BlockT},
            transaction_validity::TransactionSource,
        };
        use fc_rpc_core::types::*;
        use fp_rpc::{ConvertTransaction, ConvertTransactionRuntimeApi, EthereumRuntimeRPCApi};
        use crate::{
            eth::{format, Eth},
            internal_err,
        };
        impl<B, C, P, CT, BE, H: ExHashT, A: ChainApi, EGA> Eth<B, C, P, CT, BE, H, A, EGA>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: ProvideRuntimeApi<B> + StorageProvider<B, BE>,
            C: HeaderBackend<B> + Send + Sync + 'static,
            C::Api: BlockBuilderApi<B> + ConvertTransactionRuntimeApi<B> + EthereumRuntimeRPCApi<B>,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
            P: TransactionPool<Block = B> + Send + Sync + 'static,
            CT: ConvertTransaction<<B as BlockT>::Extrinsic> + Send + Sync + 'static,
            A: ChainApi<Block = B> + 'static,
        {
            pub async fn send_transaction(&self, request: TransactionRequest) -> Result<H256> {
                let from = match request.from {
                    Some(from) => from,
                    None => {
                        let accounts = match self.accounts() {
                            Ok(accounts) => accounts,
                            Err(e) => return Err(e),
                        };
                        match accounts.get(0) {
                            Some(account) => *account,
                            None => return Err(internal_err("no signer available")),
                        }
                    }
                };
                let nonce = match request.nonce {
                    Some(nonce) => nonce,
                    None => match self.transaction_count(from, None) {
                        Ok(nonce) => nonce,
                        Err(e) => return Err(e),
                    },
                };
                let chain_id = match self.chain_id() {
                    Ok(Some(chain_id)) => chain_id.as_u64(),
                    Ok(None) => return Err(internal_err("chain id not available")),
                    Err(e) => return Err(e),
                };
                let hash = self.client.info().best_hash;
                let gas_price = request.gas_price;
                let gas_limit = match request.gas {
                    Some(gas_limit) => gas_limit,
                    None => {
                        let block = self
                            .client
                            .runtime_api()
                            .current_block(&BlockId::Hash(hash));
                        if let Ok(Some(block)) = block {
                            block.header.gas_limit
                        } else {
                            return Err(internal_err("block unavailable, cannot query gas limit"));
                        }
                    }
                };
                let max_fee_per_gas = request.max_fee_per_gas;
                let message: Option<TransactionMessage> = request.into();
                let message = match message {
                    Some(TransactionMessage::Legacy(mut m)) => {
                        m.nonce = nonce;
                        m.chain_id = Some(chain_id);
                        m.gas_limit = gas_limit;
                        if gas_price.is_none() {
                            m.gas_price = self.gas_price().unwrap_or_default();
                        }
                        TransactionMessage::Legacy(m)
                    }
                    Some(TransactionMessage::EIP2930(mut m)) => {
                        m.nonce = nonce;
                        m.chain_id = chain_id;
                        m.gas_limit = gas_limit;
                        if gas_price.is_none() {
                            m.gas_price = self.gas_price().unwrap_or_default();
                        }
                        TransactionMessage::EIP2930(m)
                    }
                    Some(TransactionMessage::EIP1559(mut m)) => {
                        m.nonce = nonce;
                        m.chain_id = chain_id;
                        m.gas_limit = gas_limit;
                        if max_fee_per_gas.is_none() {
                            m.max_fee_per_gas = self.gas_price().unwrap_or_default();
                        }
                        TransactionMessage::EIP1559(m)
                    }
                    _ => return Err(internal_err("invalid transaction parameters")),
                };
                let mut transaction = None;
                for signer in &self.signers {
                    if signer.accounts().contains(&from) {
                        match signer.sign(message, &from) {
                            Ok(t) => transaction = Some(t),
                            Err(e) => return Err(e),
                        }
                        break;
                    }
                }
                let transaction = match transaction {
                    Some(transaction) => transaction,
                    None => return Err(internal_err("no signer available")),
                };
                let transaction_hash = transaction.hash();
                let block_hash = BlockId::hash(self.client.info().best_hash);
                let api_version = match self
                    .client
                    .runtime_api()
                    .api_version::<dyn ConvertTransactionRuntimeApi<B>>(&block_hash)
                {
                    Ok(api_version) => api_version,
                    _ => return Err(internal_err("cannot access runtime api")),
                };
                let extrinsic = match api_version {
                    Some(2) => match self
                        .client
                        .runtime_api()
                        .convert_transaction(&block_hash, transaction)
                    {
                        Ok(extrinsic) => extrinsic,
                        Err(_) => return Err(internal_err("cannot access runtime api")),
                    },
                    Some(1) => {
                        if let ethereum::TransactionV2::Legacy(legacy_transaction) = transaction {
                            #[allow(deprecated)]
                            match self
                                .client
                                .runtime_api()
                                .convert_transaction_before_version_2(
                                    &block_hash,
                                    legacy_transaction,
                                ) {
                                Ok(extrinsic) => extrinsic,
                                Err(_) => return Err(internal_err("cannot access runtime api")),
                            }
                        } else {
                            return Err(internal_err(
                                "This runtime not support eth transactions v2",
                            ));
                        }
                    }
                    None => {
                        if let Some(ref convert_transaction) = self.convert_transaction {
                            convert_transaction.convert_transaction(transaction.clone())
                        } else {
                            return Err (internal_err ("No TransactionConverter is provided and the runtime api ConvertTransactionRuntimeApi is not found")) ;
                        }
                    }
                    _ => {
                        return Err(internal_err(
                            "ConvertTransactionRuntimeApi version not supported",
                        ))
                    }
                };
                self.pool
                    .submit_one(&block_hash, TransactionSource::Local, extrinsic)
                    .map_ok(move |_| transaction_hash)
                    .map_err(|err| internal_err(format::Geth::pool_error(err)))
                    .await
            }
            pub async fn send_raw_transaction(&self, bytes: Bytes) -> Result<H256> {
                let slice = &bytes.0[..];
                if slice.is_empty() {
                    return Err(internal_err("transaction data is empty"));
                }
                let transaction: ethereum::TransactionV2 =
                    match ethereum::EnvelopedDecodable::decode(slice) {
                        Ok(transaction) => transaction,
                        Err(_) => return Err(internal_err("decode transaction failed")),
                    };
                let transaction_hash = transaction.hash();
                let block_hash = BlockId::hash(self.client.info().best_hash);
                let api_version = match self
                    .client
                    .runtime_api()
                    .api_version::<dyn ConvertTransactionRuntimeApi<B>>(&block_hash)
                {
                    Ok(api_version) => api_version,
                    _ => return Err(internal_err("cannot access runtime api")),
                };
                let extrinsic = match api_version {
                    Some(2) => match self
                        .client
                        .runtime_api()
                        .convert_transaction(&block_hash, transaction)
                    {
                        Ok(extrinsic) => extrinsic,
                        Err(_) => return Err(internal_err("cannot access runtime api")),
                    },
                    Some(1) => {
                        if let ethereum::TransactionV2::Legacy(legacy_transaction) = transaction {
                            #[allow(deprecated)]
                            match self
                                .client
                                .runtime_api()
                                .convert_transaction_before_version_2(
                                    &block_hash,
                                    legacy_transaction,
                                ) {
                                Ok(extrinsic) => extrinsic,
                                Err(_) => {
                                    return Err(internal_err("cannot access runtime api"));
                                }
                            }
                        } else {
                            return Err(internal_err(
                                "This runtime not support eth transactions v2",
                            ));
                        }
                    }
                    None => {
                        if let Some(ref convert_transaction) = self.convert_transaction {
                            convert_transaction.convert_transaction(transaction.clone())
                        } else {
                            return Err (internal_err ("No TransactionConverter is provided and the runtime api ConvertTransactionRuntimeApi is not found")) ;
                        }
                    }
                    _ => {
                        return Err(internal_err(
                            "ConvertTransactionRuntimeApi version not supported",
                        ))
                    }
                };
                self.pool
                    .submit_one(&block_hash, TransactionSource::Local, extrinsic)
                    .map_ok(move |_| transaction_hash)
                    .map_err(|err| internal_err(format::Geth::pool_error(err)))
                    .await
            }
        }
    }
    mod transaction {
        use std::sync::Arc;
        use ethereum::TransactionV2 as EthereumTransaction;
        use ethereum_types::{H256, U256, U64};
        use jsonrpsee::core::RpcResult as Result;
        use sc_client_api::backend::{Backend, StateBackend, StorageProvider};
        use sc_network_common::ExHashT;
        use sc_transaction_pool::ChainApi;
        use sc_transaction_pool_api::InPoolTransaction;
        use sp_api::{ApiExt, ProvideRuntimeApi};
        use sp_blockchain::HeaderBackend;
        use sp_core::hashing::keccak_256;
        use sp_runtime::{
            generic::BlockId,
            traits::{BlakeTwo256, Block as BlockT},
        };
        use fc_rpc_core::types::*;
        use fp_rpc::EthereumRuntimeRPCApi;
        use crate::{
            eth::{transaction_build, Eth},
            frontier_backend_client, internal_err,
        };
        impl<B, C, P, CT, BE, H: ExHashT, A: ChainApi, EGA> Eth<B, C, P, CT, BE, H, A, EGA>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: ProvideRuntimeApi<B>
                + StorageProvider<B, BE>
                + HeaderBackend<B>
                + Send
                + Sync
                + 'static,
            C::Api: EthereumRuntimeRPCApi<B>,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
            A: ChainApi<Block = B> + 'static,
        {
            pub async fn transaction_by_hash(&self, hash: H256) -> Result<Option<Transaction>> {
                let client = Arc::clone(&self.client);
                let block_data_cache = Arc::clone(&self.block_data_cache);
                let backend = Arc::clone(&self.backend);
                let graph = Arc::clone(&self.graph);
                let (hash, index) = match frontier_backend_client::load_transactions::<B, C>(
                    client.as_ref(),
                    backend.as_ref(),
                    hash,
                    true,
                )
                .map_err(|err| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &[""],
                            &[::core::fmt::ArgumentV1::new_debug(&err)],
                        ));
                        res
                    })
                })? {
                    Some((hash, index)) => (hash, index as usize),
                    None => {
                        let api = client.runtime_api();
                        let best_block: BlockId<B> = BlockId::Hash(client.info().best_hash);
                        let api_version = if let Ok(Some(api_version)) =
                            api.api_version::<dyn EthereumRuntimeRPCApi<B>>(&best_block)
                        {
                            api_version
                        } else {
                            return Err(internal_err("failed to retrieve Runtime Api version"));
                        };
                        let mut xts: Vec<<B as BlockT>::Extrinsic> = Vec::new();
                        xts.extend(
                            graph
                                .validated_pool()
                                .ready()
                                .map(|in_pool_tx| in_pool_tx.data().clone())
                                .collect::<Vec<<B as BlockT>::Extrinsic>>(),
                        );
                        xts.extend(
                            graph
                                .validated_pool()
                                .futures()
                                .iter()
                                .map(|(_hash, extrinsic)| extrinsic.clone())
                                .collect::<Vec<<B as BlockT>::Extrinsic>>(),
                        );
                        let ethereum_transactions: Vec<EthereumTransaction> = if api_version > 1 {
                            api.extrinsic_filter(&best_block, xts).map_err(|err| {
                                internal_err({
                                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                        &["fetch runtime extrinsic filter failed: "],
                                        &[::core::fmt::ArgumentV1::new_debug(&err)],
                                    ));
                                    res
                                })
                            })?
                        } else {
                            #[allow(deprecated)]
                            let legacy = api
                                .extrinsic_filter_before_version_2(&best_block, xts)
                                .map_err(|err| {
                                    internal_err({
                                        let res =
                                            ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                                &["fetch runtime extrinsic filter failed: "],
                                                &[::core::fmt::ArgumentV1::new_debug(&err)],
                                            ));
                                        res
                                    })
                                })?;
                            legacy.into_iter().map(|tx| tx.into()).collect()
                        };
                        for txn in ethereum_transactions {
                            let inner_hash = txn.hash();
                            if hash == inner_hash {
                                return Ok(Some(transaction_build(txn, None, None, None)));
                            }
                        }
                        return Ok(None);
                    }
                };
                let id = match frontier_backend_client::load_hash::<B, C>(
                    client.as_ref(),
                    backend.as_ref(),
                    hash,
                )
                .map_err(|err| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &[""],
                            &[::core::fmt::ArgumentV1::new_debug(&err)],
                        ));
                        res
                    })
                })? {
                    Some(hash) => hash,
                    _ => return Ok(None),
                };
                let substrate_hash = client.expect_block_hash_from_id(&id).map_err(|_| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &["Expect block number from id: "],
                            &[::core::fmt::ArgumentV1::new_display(&id)],
                        ));
                        res
                    })
                })?;
                let schema = frontier_backend_client::onchain_storage_schema::<B, C, BE>(
                    client.as_ref(),
                    id,
                );
                let block = block_data_cache.current_block(schema, substrate_hash).await;
                let statuses = block_data_cache
                    .current_transaction_statuses(schema, substrate_hash)
                    .await;
                let base_fee = client.runtime_api().gas_price(&id).unwrap_or_default();
                match (block, statuses) {
                    (Some(block), Some(statuses)) => Ok(Some(transaction_build(
                        block.transactions[index].clone(),
                        Some(block),
                        Some(statuses[index].clone()),
                        Some(base_fee),
                    ))),
                    _ => Ok(None),
                }
            }
            pub async fn transaction_by_block_hash_and_index(
                &self,
                hash: H256,
                index: Index,
            ) -> Result<Option<Transaction>> {
                let client = Arc::clone(&self.client);
                let block_data_cache = Arc::clone(&self.block_data_cache);
                let backend = Arc::clone(&self.backend);
                let id = match frontier_backend_client::load_hash::<B, C>(
                    client.as_ref(),
                    backend.as_ref(),
                    hash,
                )
                .map_err(|err| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &[""],
                            &[::core::fmt::ArgumentV1::new_debug(&err)],
                        ));
                        res
                    })
                })? {
                    Some(hash) => hash,
                    _ => return Ok(None),
                };
                let substrate_hash = client.expect_block_hash_from_id(&id).map_err(|_| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &["Expect block number from id: "],
                            &[::core::fmt::ArgumentV1::new_display(&id)],
                        ));
                        res
                    })
                })?;
                let index = index.value();
                let schema = frontier_backend_client::onchain_storage_schema::<B, C, BE>(
                    client.as_ref(),
                    id,
                );
                let block = block_data_cache.current_block(schema, substrate_hash).await;
                let statuses = block_data_cache
                    .current_transaction_statuses(schema, substrate_hash)
                    .await;
                let base_fee = client.runtime_api().gas_price(&id).unwrap_or_default();
                match (block, statuses) {
                    (Some(block), Some(statuses)) => {
                        if let (Some(transaction), Some(status)) =
                            (block.transactions.get(index), statuses.get(index))
                        {
                            Ok(Some(transaction_build(
                                transaction.clone(),
                                Some(block),
                                Some(status.clone()),
                                Some(base_fee),
                            )))
                        } else {
                            Err(internal_err({
                                let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                    &["", " is out of bounds"],
                                    &[::core::fmt::ArgumentV1::new_debug(&index)],
                                ));
                                res
                            }))
                        }
                    }
                    _ => Ok(None),
                }
            }
            pub async fn transaction_by_block_number_and_index(
                &self,
                number: BlockNumber,
                index: Index,
            ) -> Result<Option<Transaction>> {
                let client = Arc::clone(&self.client);
                let block_data_cache = Arc::clone(&self.block_data_cache);
                let backend = Arc::clone(&self.backend);
                let id = match frontier_backend_client::native_block_id::<B, C>(
                    client.as_ref(),
                    backend.as_ref(),
                    Some(number),
                )? {
                    Some(id) => id,
                    None => return Ok(None),
                };
                let substrate_hash = client.expect_block_hash_from_id(&id).map_err(|_| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &["Expect block number from id: "],
                            &[::core::fmt::ArgumentV1::new_display(&id)],
                        ));
                        res
                    })
                })?;
                let index = index.value();
                let schema = frontier_backend_client::onchain_storage_schema::<B, C, BE>(
                    client.as_ref(),
                    id,
                );
                let block = block_data_cache.current_block(schema, substrate_hash).await;
                let statuses = block_data_cache
                    .current_transaction_statuses(schema, substrate_hash)
                    .await;
                let base_fee = client.runtime_api().gas_price(&id).unwrap_or_default();
                match (block, statuses) {
                    (Some(block), Some(statuses)) => {
                        if let (Some(transaction), Some(status)) =
                            (block.transactions.get(index), statuses.get(index))
                        {
                            Ok(Some(transaction_build(
                                transaction.clone(),
                                Some(block),
                                Some(status.clone()),
                                Some(base_fee),
                            )))
                        } else {
                            Err(internal_err({
                                let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                                    &["", " is out of bounds"],
                                    &[::core::fmt::ArgumentV1::new_debug(&index)],
                                ));
                                res
                            }))
                        }
                    }
                    _ => Ok(None),
                }
            }
            pub async fn transaction_receipt(&self, hash: H256) -> Result<Option<Receipt>> {
                let client = Arc::clone(&self.client);
                let overrides = Arc::clone(&self.overrides);
                let block_data_cache = Arc::clone(&self.block_data_cache);
                let backend = Arc::clone(&self.backend);
                let (hash, index) = match frontier_backend_client::load_transactions::<B, C>(
                    client.as_ref(),
                    backend.as_ref(),
                    hash,
                    true,
                )
                .map_err(|err| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &[""],
                            &[::core::fmt::ArgumentV1::new_debug(&err)],
                        ));
                        res
                    })
                })? {
                    Some((hash, index)) => (hash, index as usize),
                    None => return Ok(None),
                };
                let id = match frontier_backend_client::load_hash::<B, C>(
                    client.as_ref(),
                    backend.as_ref(),
                    hash,
                )
                .map_err(|err| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &[""],
                            &[::core::fmt::ArgumentV1::new_debug(&err)],
                        ));
                        res
                    })
                })? {
                    Some(hash) => hash,
                    _ => return Ok(None),
                };
                let substrate_hash = client.expect_block_hash_from_id(&id).map_err(|_| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &["Expect block number from id: "],
                            &[::core::fmt::ArgumentV1::new_display(&id)],
                        ));
                        res
                    })
                })?;
                let schema = frontier_backend_client::onchain_storage_schema::<B, C, BE>(
                    client.as_ref(),
                    id,
                );
                let handler = overrides
                    .schemas
                    .get(&schema)
                    .unwrap_or(&overrides.fallback);
                let block = block_data_cache.current_block(schema, substrate_hash).await;
                let statuses = block_data_cache
                    .current_transaction_statuses(schema, substrate_hash)
                    .await;
                let receipts = handler.current_receipts(&id);
                let is_eip1559 = handler.is_eip1559(&id);
                match (block, statuses, receipts) {
                    (Some(block), Some(statuses), Some(receipts)) => {
                        let block_hash = H256::from(keccak_256(&rlp::encode(&block.header)));
                        let receipt = receipts[index].clone();
                        let (logs, logs_bloom, status_code, cumulative_gas_used, gas_used) =
                            if !is_eip1559 {
                                match receipt {
                                    ethereum::ReceiptV3::Legacy(ref d) => {
                                        let index = core::cmp::min(receipts.len(), index + 1);
                                        let cumulative_gas : u32 = receipts [.. index] . iter () . map (| r | match r { ethereum :: ReceiptV3 :: Legacy (d) => Ok (d . used_gas . as_u32 ()) , _ => Err (internal_err ({ let res = :: alloc :: fmt :: format (:: core :: fmt :: Arguments :: new_v1 (& ["Unknown receipt for request "] , & [:: core :: fmt :: ArgumentV1 :: new_display (& hash)])) ; res })) , }) . sum :: < Result < u32 > > () ? ;
                                        (
                                            d.logs.clone(),
                                            d.logs_bloom,
                                            d.status_code,
                                            U256::from(cumulative_gas),
                                            d.used_gas,
                                        )
                                    }
                                    _ => {
                                        return Err(internal_err({
                                            let res = ::alloc::fmt::format(
                                                ::core::fmt::Arguments::new_v1(
                                                    &["Unknown receipt for request "],
                                                    &[::core::fmt::ArgumentV1::new_display(&hash)],
                                                ),
                                            );
                                            res
                                        }))
                                    }
                                }
                            } else {
                                match receipt {
                                    ethereum::ReceiptV3::Legacy(ref d)
                                    | ethereum::ReceiptV3::EIP2930(ref d)
                                    | ethereum::ReceiptV3::EIP1559(ref d) => {
                                        let cumulative_gas = d.used_gas;
                                        let gas_used = if index > 0 {
                                            let previous_receipt = receipts[index - 1].clone();
                                            let previous_gas_used = match previous_receipt {
                                                ethereum::ReceiptV3::Legacy(d)
                                                | ethereum::ReceiptV3::EIP2930(d)
                                                | ethereum::ReceiptV3::EIP1559(d) => d.used_gas,
                                            };
                                            cumulative_gas.saturating_sub(previous_gas_used)
                                        } else {
                                            cumulative_gas
                                        };
                                        (
                                            d.logs.clone(),
                                            d.logs_bloom,
                                            d.status_code,
                                            cumulative_gas,
                                            gas_used,
                                        )
                                    }
                                }
                            };
                        let status = statuses[index].clone();
                        let mut cumulative_receipts = receipts;
                        cumulative_receipts.truncate((status.transaction_index + 1) as usize);
                        let transaction = block.transactions[index].clone();
                        let effective_gas_price = match transaction {
                            EthereumTransaction::Legacy(t) => t.gas_price,
                            EthereumTransaction::EIP2930(t) => t.gas_price,
                            EthereumTransaction::EIP1559(t) => client
                                .runtime_api()
                                .gas_price(&id)
                                .unwrap_or_default()
                                .checked_add(t.max_priority_fee_per_gas)
                                .unwrap_or_else(U256::max_value)
                                .min(t.max_fee_per_gas),
                        };
                        return Ok(Some(Receipt {
                            transaction_hash: Some(status.transaction_hash),
                            transaction_index: Some(status.transaction_index.into()),
                            block_hash: Some(block_hash),
                            from: Some(status.from),
                            to: status.to,
                            block_number: Some(block.header.number),
                            cumulative_gas_used,
                            gas_used: Some(gas_used),
                            contract_address: status.contract_address,
                            logs: {
                                let mut pre_receipts_log_index = None;
                                if cumulative_receipts.len() > 0 {
                                    cumulative_receipts.truncate(cumulative_receipts.len() - 1);
                                    pre_receipts_log_index = Some(
                                        cumulative_receipts
                                            .iter()
                                            .map(|r| match r {
                                                ethereum::ReceiptV3::Legacy(d)
                                                | ethereum::ReceiptV3::EIP2930(d)
                                                | ethereum::ReceiptV3::EIP1559(d) => {
                                                    d.logs.len() as u32
                                                }
                                            })
                                            .sum::<u32>(),
                                    );
                                }
                                logs.iter()
                                    .enumerate()
                                    .map(|(i, log)| Log {
                                        address: log.address,
                                        topics: log.topics.clone(),
                                        data: Bytes(log.data.clone()),
                                        block_hash: Some(block_hash),
                                        block_number: Some(block.header.number),
                                        transaction_hash: Some(status.transaction_hash),
                                        transaction_index: Some(status.transaction_index.into()),
                                        log_index: Some(U256::from(
                                            (pre_receipts_log_index.unwrap_or(0)) + i as u32,
                                        )),
                                        transaction_log_index: Some(U256::from(i)),
                                        removed: false,
                                    })
                                    .collect()
                            },
                            status_code: Some(U64::from(status_code)),
                            logs_bloom,
                            state_root: None,
                            effective_gas_price,
                            transaction_type: match receipt {
                                ethereum::ReceiptV3::Legacy(_) => U256::from(0),
                                ethereum::ReceiptV3::EIP2930(_) => U256::from(1),
                                ethereum::ReceiptV3::EIP1559(_) => U256::from(2),
                            },
                        }));
                    }
                    _ => Ok(None),
                }
            }
        }
    }
    use std::{collections::BTreeMap, marker::PhantomData, sync::Arc};
    use ethereum::{BlockV2 as EthereumBlock, TransactionV2 as EthereumTransaction};
    use ethereum_types::{H160, H256, H512, H64, U256, U64};
    use jsonrpsee::core::{async_trait, RpcResult as Result};
    use sc_client_api::backend::{Backend, StateBackend, StorageProvider};
    use sc_network::NetworkService;
    use sc_network_common::ExHashT;
    use sc_transaction_pool::{ChainApi, Pool};
    use sc_transaction_pool_api::{InPoolTransaction, TransactionPool};
    use sp_api::{Core, HeaderT, ProvideRuntimeApi};
    use sp_block_builder::BlockBuilder as BlockBuilderApi;
    use sp_blockchain::HeaderBackend;
    use sp_core::hashing::keccak_256;
    use sp_runtime::{
        generic::BlockId,
        traits::{BlakeTwo256, Block as BlockT, UniqueSaturatedInto},
    };
    use fc_rpc_core::{types::*, EthApiServer};
    use fp_rpc::{ConvertTransactionRuntimeApi, EthereumRuntimeRPCApi, TransactionStatus};
    use crate::{internal_err, overrides::OverrideHandle, public_key, signer::EthSigner};
    pub use self::{
        cache::{EthBlockDataCacheTask, EthTask},
        execute::EstimateGasAdapter,
        filter::EthFilter,
    };
    /// Eth API implementation.
    pub struct Eth<B: BlockT, C, P, CT, BE, H: ExHashT, A: ChainApi, EGA = ()> {
        pool: Arc<P>,
        graph: Arc<Pool<A>>,
        client: Arc<C>,
        convert_transaction: Option<CT>,
        network: Arc<NetworkService<B, H>>,
        is_authority: bool,
        signers: Vec<Box<dyn EthSigner>>,
        overrides: Arc<OverrideHandle<B>>,
        backend: Arc<fc_db::Backend<B>>,
        block_data_cache: Arc<EthBlockDataCacheTask<B>>,
        fee_history_cache: FeeHistoryCache,
        fee_history_cache_limit: FeeHistoryCacheLimit,
        /// When using eth_call/eth_estimateGas, the maximum allowed gas limit will be
        /// block.gas_limit * execute_gas_limit_multiplier
        execute_gas_limit_multiplier: u64,
        _marker: PhantomData<(B, BE, EGA)>,
    }
    impl<B: BlockT, C, P, CT, BE, H: ExHashT, A: ChainApi> Eth<B, C, P, CT, BE, H, A, ()> {
        pub fn new(
            client: Arc<C>,
            pool: Arc<P>,
            graph: Arc<Pool<A>>,
            convert_transaction: Option<CT>,
            network: Arc<NetworkService<B, H>>,
            signers: Vec<Box<dyn EthSigner>>,
            overrides: Arc<OverrideHandle<B>>,
            backend: Arc<fc_db::Backend<B>>,
            is_authority: bool,
            block_data_cache: Arc<EthBlockDataCacheTask<B>>,
            fee_history_cache: FeeHistoryCache,
            fee_history_cache_limit: FeeHistoryCacheLimit,
            execute_gas_limit_multiplier: u64,
        ) -> Self {
            Self {
                client,
                pool,
                graph,
                convert_transaction,
                network,
                is_authority,
                signers,
                overrides,
                backend,
                block_data_cache,
                fee_history_cache,
                fee_history_cache_limit,
                execute_gas_limit_multiplier,
                _marker: PhantomData,
            }
        }
    }
    impl<B: BlockT, C, P, CT, BE, H: ExHashT, A: ChainApi, EGA> Eth<B, C, P, CT, BE, H, A, EGA> {
        pub fn with_estimate_gas_adapter<EGA2: EstimateGasAdapter>(
            self,
        ) -> Eth<B, C, P, CT, BE, H, A, EGA2> {
            let Self {
                client,
                pool,
                graph,
                convert_transaction,
                network,
                is_authority,
                signers,
                overrides,
                backend,
                block_data_cache,
                fee_history_cache,
                fee_history_cache_limit,
                execute_gas_limit_multiplier,
                _marker: _,
            } = self;
            Eth {
                client,
                pool,
                graph,
                convert_transaction,
                network,
                is_authority,
                signers,
                overrides,
                backend,
                block_data_cache,
                fee_history_cache,
                fee_history_cache_limit,
                execute_gas_limit_multiplier,
                _marker: PhantomData,
            }
        }
    }
    impl<B, C, P, CT, BE, H: ExHashT, A, EGA> EthApiServer for Eth<B, C, P, CT, BE, H, A, EGA>
    where
        B: BlockT<Hash = H256> + Send + Sync + 'static,
        C: ProvideRuntimeApi<B> + StorageProvider<B, BE>,
        C: HeaderBackend<B> + Send + Sync + 'static,
        C::Api: BlockBuilderApi<B> + ConvertTransactionRuntimeApi<B> + EthereumRuntimeRPCApi<B>,
        P: TransactionPool<Block = B> + Send + Sync + 'static,
        CT: fp_rpc::ConvertTransaction<<B as BlockT>::Extrinsic> + Send + Sync + 'static,
        BE: Backend<B> + 'static,
        BE::State: StateBackend<BlakeTwo256>,
        A: ChainApi<Block = B> + 'static,
        EGA: EstimateGasAdapter + Send + Sync + 'static,
    {
        fn protocol_version(&self) -> Result<u64> {
            self.protocol_version()
        }
        fn syncing(&self) -> Result<SyncStatus> {
            self.syncing()
        }
        fn author(&self) -> Result<H160> {
            self.author()
        }
        fn accounts(&self) -> Result<Vec<H160>> {
            self.accounts()
        }
        fn block_number(&self) -> Result<U256> {
            self.block_number()
        }
        fn chain_id(&self) -> Result<Option<U64>> {
            self.chain_id()
        }
        #[allow(
            clippy::let_unit_value,
            clippy::no_effect_underscore_binding,
            clippy::shadow_same,
            clippy::type_complexity,
            clippy::type_repetition_in_bounds,
            clippy::used_underscore_binding
        )]
        fn block_by_hash<'life0, 'async_trait>(
            &'life0 self,
            hash: H256,
            full: bool,
        ) -> ::core::pin::Pin<
            Box<
                dyn ::core::future::Future<Output = Result<Option<RichBlock>>>
                    + ::core::marker::Send
                    + 'async_trait,
            >,
        >
        where
            'life0: 'async_trait,
            Self: 'async_trait,
        {
            Box::pin(async move {
                if let ::core::option::Option::Some(__ret) =
                    ::core::option::Option::None::<Result<Option<RichBlock>>>
                {
                    return __ret;
                }
                let __self = self;
                let hash = hash;
                let full = full;
                let __ret: Result<Option<RichBlock>> = { __self.block_by_hash(hash, full).await };
                #[allow(unreachable_code)]
                __ret
            })
        }
        #[allow(
            clippy::let_unit_value,
            clippy::no_effect_underscore_binding,
            clippy::shadow_same,
            clippy::type_complexity,
            clippy::type_repetition_in_bounds,
            clippy::used_underscore_binding
        )]
        fn block_by_number<'life0, 'async_trait>(
            &'life0 self,
            number: BlockNumber,
            full: bool,
        ) -> ::core::pin::Pin<
            Box<
                dyn ::core::future::Future<Output = Result<Option<RichBlock>>>
                    + ::core::marker::Send
                    + 'async_trait,
            >,
        >
        where
            'life0: 'async_trait,
            Self: 'async_trait,
        {
            Box::pin(async move {
                if let ::core::option::Option::Some(__ret) =
                    ::core::option::Option::None::<Result<Option<RichBlock>>>
                {
                    return __ret;
                }
                let __self = self;
                let number = number;
                let full = full;
                let __ret: Result<Option<RichBlock>> =
                    { __self.block_by_number(number, full).await };
                #[allow(unreachable_code)]
                __ret
            })
        }
        fn block_transaction_count_by_hash(&self, hash: H256) -> Result<Option<U256>> {
            self.block_transaction_count_by_hash(hash)
        }
        fn block_transaction_count_by_number(&self, number: BlockNumber) -> Result<Option<U256>> {
            self.block_transaction_count_by_number(number)
        }
        fn block_uncles_count_by_hash(&self, hash: H256) -> Result<U256> {
            self.block_uncles_count_by_hash(hash)
        }
        fn block_uncles_count_by_number(&self, number: BlockNumber) -> Result<U256> {
            self.block_uncles_count_by_number(number)
        }
        fn uncle_by_block_hash_and_index(
            &self,
            hash: H256,
            index: Index,
        ) -> Result<Option<RichBlock>> {
            self.uncle_by_block_hash_and_index(hash, index)
        }
        fn uncle_by_block_number_and_index(
            &self,
            number: BlockNumber,
            index: Index,
        ) -> Result<Option<RichBlock>> {
            self.uncle_by_block_number_and_index(number, index)
        }
        #[allow(
            clippy::let_unit_value,
            clippy::no_effect_underscore_binding,
            clippy::shadow_same,
            clippy::type_complexity,
            clippy::type_repetition_in_bounds,
            clippy::used_underscore_binding
        )]
        fn transaction_by_hash<'life0, 'async_trait>(
            &'life0 self,
            hash: H256,
        ) -> ::core::pin::Pin<
            Box<
                dyn ::core::future::Future<Output = Result<Option<Transaction>>>
                    + ::core::marker::Send
                    + 'async_trait,
            >,
        >
        where
            'life0: 'async_trait,
            Self: 'async_trait,
        {
            Box::pin(async move {
                if let ::core::option::Option::Some(__ret) =
                    ::core::option::Option::None::<Result<Option<Transaction>>>
                {
                    return __ret;
                }
                let __self = self;
                let hash = hash;
                let __ret: Result<Option<Transaction>> = { __self.transaction_by_hash(hash).await };
                #[allow(unreachable_code)]
                __ret
            })
        }
        #[allow(
            clippy::let_unit_value,
            clippy::no_effect_underscore_binding,
            clippy::shadow_same,
            clippy::type_complexity,
            clippy::type_repetition_in_bounds,
            clippy::used_underscore_binding
        )]
        fn transaction_by_block_hash_and_index<'life0, 'async_trait>(
            &'life0 self,
            hash: H256,
            index: Index,
        ) -> ::core::pin::Pin<
            Box<
                dyn ::core::future::Future<Output = Result<Option<Transaction>>>
                    + ::core::marker::Send
                    + 'async_trait,
            >,
        >
        where
            'life0: 'async_trait,
            Self: 'async_trait,
        {
            Box::pin(async move {
                if let ::core::option::Option::Some(__ret) =
                    ::core::option::Option::None::<Result<Option<Transaction>>>
                {
                    return __ret;
                }
                let __self = self;
                let hash = hash;
                let index = index;
                let __ret: Result<Option<Transaction>> = {
                    __self
                        .transaction_by_block_hash_and_index(hash, index)
                        .await
                };
                #[allow(unreachable_code)]
                __ret
            })
        }
        #[allow(
            clippy::let_unit_value,
            clippy::no_effect_underscore_binding,
            clippy::shadow_same,
            clippy::type_complexity,
            clippy::type_repetition_in_bounds,
            clippy::used_underscore_binding
        )]
        fn transaction_by_block_number_and_index<'life0, 'async_trait>(
            &'life0 self,
            number: BlockNumber,
            index: Index,
        ) -> ::core::pin::Pin<
            Box<
                dyn ::core::future::Future<Output = Result<Option<Transaction>>>
                    + ::core::marker::Send
                    + 'async_trait,
            >,
        >
        where
            'life0: 'async_trait,
            Self: 'async_trait,
        {
            Box::pin(async move {
                if let ::core::option::Option::Some(__ret) =
                    ::core::option::Option::None::<Result<Option<Transaction>>>
                {
                    return __ret;
                }
                let __self = self;
                let number = number;
                let index = index;
                let __ret: Result<Option<Transaction>> = {
                    __self
                        .transaction_by_block_number_and_index(number, index)
                        .await
                };
                #[allow(unreachable_code)]
                __ret
            })
        }
        #[allow(
            clippy::let_unit_value,
            clippy::no_effect_underscore_binding,
            clippy::shadow_same,
            clippy::type_complexity,
            clippy::type_repetition_in_bounds,
            clippy::used_underscore_binding
        )]
        fn transaction_receipt<'life0, 'async_trait>(
            &'life0 self,
            hash: H256,
        ) -> ::core::pin::Pin<
            Box<
                dyn ::core::future::Future<Output = Result<Option<Receipt>>>
                    + ::core::marker::Send
                    + 'async_trait,
            >,
        >
        where
            'life0: 'async_trait,
            Self: 'async_trait,
        {
            Box::pin(async move {
                if let ::core::option::Option::Some(__ret) =
                    ::core::option::Option::None::<Result<Option<Receipt>>>
                {
                    return __ret;
                }
                let __self = self;
                let hash = hash;
                let __ret: Result<Option<Receipt>> = { __self.transaction_receipt(hash).await };
                #[allow(unreachable_code)]
                __ret
            })
        }
        fn balance(&self, address: H160, number: Option<BlockNumber>) -> Result<U256> {
            self.balance(address, number)
        }
        fn storage_at(
            &self,
            address: H160,
            index: U256,
            number: Option<BlockNumber>,
        ) -> Result<H256> {
            self.storage_at(address, index, number)
        }
        fn transaction_count(&self, address: H160, number: Option<BlockNumber>) -> Result<U256> {
            self.transaction_count(address, number)
        }
        fn code_at(&self, address: H160, number: Option<BlockNumber>) -> Result<Bytes> {
            self.code_at(address, number)
        }
        fn call(
            &self,
            request: CallRequest,
            number: Option<BlockNumber>,
            state_override: Option<CallStateOverride>,
        ) -> Result<Bytes> {
            self.call(request, number, state_override)
        }
        #[allow(
            clippy::let_unit_value,
            clippy::no_effect_underscore_binding,
            clippy::shadow_same,
            clippy::type_complexity,
            clippy::type_repetition_in_bounds,
            clippy::used_underscore_binding
        )]
        fn estimate_gas<'life0, 'async_trait>(
            &'life0 self,
            request: CallRequest,
            number: Option<BlockNumber>,
        ) -> ::core::pin::Pin<
            Box<
                dyn ::core::future::Future<Output = Result<U256>>
                    + ::core::marker::Send
                    + 'async_trait,
            >,
        >
        where
            'life0: 'async_trait,
            Self: 'async_trait,
        {
            Box::pin(async move {
                if let ::core::option::Option::Some(__ret) =
                    ::core::option::Option::None::<Result<U256>>
                {
                    return __ret;
                }
                let __self = self;
                let request = request;
                let number = number;
                let __ret: Result<U256> = { __self.estimate_gas(request, number).await };
                #[allow(unreachable_code)]
                __ret
            })
        }
        fn gas_price(&self) -> Result<U256> {
            self.gas_price()
        }
        fn fee_history(
            &self,
            block_count: U256,
            newest_block: BlockNumber,
            reward_percentiles: Option<Vec<f64>>,
        ) -> Result<FeeHistory> {
            self.fee_history(block_count, newest_block, reward_percentiles)
        }
        fn max_priority_fee_per_gas(&self) -> Result<U256> {
            self.max_priority_fee_per_gas()
        }
        fn is_mining(&self) -> Result<bool> {
            self.is_mining()
        }
        fn hashrate(&self) -> Result<U256> {
            self.hashrate()
        }
        fn work(&self) -> Result<Work> {
            self.work()
        }
        fn submit_hashrate(&self, hashrate: U256, id: H256) -> Result<bool> {
            self.submit_hashrate(hashrate, id)
        }
        fn submit_work(&self, nonce: H64, pow_hash: H256, mix_digest: H256) -> Result<bool> {
            self.submit_work(nonce, pow_hash, mix_digest)
        }
        #[allow(
            clippy::let_unit_value,
            clippy::no_effect_underscore_binding,
            clippy::shadow_same,
            clippy::type_complexity,
            clippy::type_repetition_in_bounds,
            clippy::used_underscore_binding
        )]
        fn send_transaction<'life0, 'async_trait>(
            &'life0 self,
            request: TransactionRequest,
        ) -> ::core::pin::Pin<
            Box<
                dyn ::core::future::Future<Output = Result<H256>>
                    + ::core::marker::Send
                    + 'async_trait,
            >,
        >
        where
            'life0: 'async_trait,
            Self: 'async_trait,
        {
            Box::pin(async move {
                if let ::core::option::Option::Some(__ret) =
                    ::core::option::Option::None::<Result<H256>>
                {
                    return __ret;
                }
                let __self = self;
                let request = request;
                let __ret: Result<H256> = { __self.send_transaction(request).await };
                #[allow(unreachable_code)]
                __ret
            })
        }
        #[allow(
            clippy::let_unit_value,
            clippy::no_effect_underscore_binding,
            clippy::shadow_same,
            clippy::type_complexity,
            clippy::type_repetition_in_bounds,
            clippy::used_underscore_binding
        )]
        fn send_raw_transaction<'life0, 'async_trait>(
            &'life0 self,
            bytes: Bytes,
        ) -> ::core::pin::Pin<
            Box<
                dyn ::core::future::Future<Output = Result<H256>>
                    + ::core::marker::Send
                    + 'async_trait,
            >,
        >
        where
            'life0: 'async_trait,
            Self: 'async_trait,
        {
            Box::pin(async move {
                if let ::core::option::Option::Some(__ret) =
                    ::core::option::Option::None::<Result<H256>>
                {
                    return __ret;
                }
                let __self = self;
                let bytes = bytes;
                let __ret: Result<H256> = { __self.send_raw_transaction(bytes).await };
                #[allow(unreachable_code)]
                __ret
            })
        }
    }
    fn rich_block_build(
        block: EthereumBlock,
        statuses: Vec<Option<TransactionStatus>>,
        hash: Option<H256>,
        full_transactions: bool,
        base_fee: Option<U256>,
    ) -> RichBlock {
        Rich {
            inner: Block {
                header: Header {
                    hash: Some(
                        hash.unwrap_or_else(|| H256::from(keccak_256(&rlp::encode(&block.header)))),
                    ),
                    parent_hash: block.header.parent_hash,
                    uncles_hash: block.header.ommers_hash,
                    author: block.header.beneficiary,
                    miner: block.header.beneficiary,
                    state_root: block.header.state_root,
                    transactions_root: block.header.transactions_root,
                    receipts_root: block.header.receipts_root,
                    number: Some(block.header.number),
                    gas_used: block.header.gas_used,
                    gas_limit: block.header.gas_limit,
                    extra_data: Bytes(block.header.extra_data.clone()),
                    logs_bloom: block.header.logs_bloom,
                    timestamp: U256::from(block.header.timestamp / 1000),
                    difficulty: block.header.difficulty,
                    nonce: Some(block.header.nonce),
                    size: Some(U256::from(rlp::encode(&block.header).len() as u32)),
                },
                total_difficulty: U256::zero(),
                uncles: ::alloc::vec::Vec::new(),
                transactions: {
                    if full_transactions {
                        BlockTransactions::Full(
                            block
                                .transactions
                                .iter()
                                .enumerate()
                                .map(|(index, transaction)| {
                                    transaction_build(
                                        transaction.clone(),
                                        Some(block.clone()),
                                        Some(statuses[index].clone().unwrap_or_default()),
                                        base_fee,
                                    )
                                })
                                .collect(),
                        )
                    } else {
                        BlockTransactions::Hashes(
                            block
                                .transactions
                                .iter()
                                .map(|transaction| transaction.hash())
                                .collect(),
                        )
                    }
                },
                size: Some(U256::from(rlp::encode(&block).len() as u32)),
                base_fee_per_gas: base_fee,
            },
            extra_info: BTreeMap::new(),
        }
    }
    fn transaction_build(
        ethereum_transaction: EthereumTransaction,
        block: Option<EthereumBlock>,
        status: Option<TransactionStatus>,
        base_fee: Option<U256>,
    ) -> Transaction {
        let mut transaction: Transaction = ethereum_transaction.clone().into();
        if let EthereumTransaction::EIP1559(_) = ethereum_transaction {
            if block.is_none() && status.is_none() {
                transaction.gas_price = transaction.max_fee_per_gas;
            } else {
                let base_fee = base_fee.unwrap_or_default();
                let max_priority_fee_per_gas =
                    transaction.max_priority_fee_per_gas.unwrap_or_default();
                let max_fee_per_gas = transaction.max_fee_per_gas.unwrap_or_default();
                transaction.gas_price = Some(
                    base_fee
                        .checked_add(max_priority_fee_per_gas)
                        .unwrap_or_else(U256::max_value)
                        .min(max_fee_per_gas),
                );
            }
        }
        let pubkey = match public_key(&ethereum_transaction) {
            Ok(p) => Some(p),
            Err(_e) => None,
        };
        transaction.block_hash = block
            .as_ref()
            .map(|block| H256::from(keccak_256(&rlp::encode(&block.header))));
        transaction.block_number = block.as_ref().map(|block| block.header.number);
        transaction.transaction_index = status.as_ref().map(|status| {
            U256::from(UniqueSaturatedInto::<u32>::unique_saturated_into(
                status.transaction_index,
            ))
        });
        transaction.from = status.as_ref().map_or(
            {
                match pubkey {
                    Some(pk) => H160::from(H256::from(keccak_256(&pk))),
                    _ => H160::default(),
                }
            },
            |status| status.from,
        );
        transaction.to = status.as_ref().map_or(
            {
                let action = match ethereum_transaction {
                    EthereumTransaction::Legacy(t) => t.action,
                    EthereumTransaction::EIP2930(t) => t.action,
                    EthereumTransaction::EIP1559(t) => t.action,
                };
                match action {
                    ethereum::TransactionAction::Call(to) => Some(to),
                    _ => None,
                }
            },
            |status| status.to,
        );
        transaction.creates = status.as_ref().and_then(|status| status.contract_address);
        transaction.public_key = pubkey.as_ref().map(H512::from);
        transaction
    }
    fn pending_runtime_api<'a, B: BlockT, C, BE, A: ChainApi>(
        client: &'a C,
        graph: &'a Pool<A>,
    ) -> Result<sp_api::ApiRef<'a, C::Api>>
    where
        B: BlockT<Hash = H256> + Send + Sync + 'static,
        C: ProvideRuntimeApi<B> + StorageProvider<B, BE>,
        C: HeaderBackend<B> + Send + Sync + 'static,
        C::Api: BlockBuilderApi<B> + EthereumRuntimeRPCApi<B>,
        BE: Backend<B> + 'static,
        BE::State: StateBackend<BlakeTwo256>,
        A: ChainApi<Block = B> + 'static,
    {
        let api = client.runtime_api();
        let best_hash = client.info().best_hash;
        let best = BlockId::Hash(client.info().best_hash);
        let xts: Vec<<B as BlockT>::Extrinsic> = graph
            .validated_pool()
            .ready()
            .map(|in_pool_tx| in_pool_tx.data().clone())
            .collect::<Vec<<B as BlockT>::Extrinsic>>();
        if let Ok(Some(header)) = client.header(best_hash) {
            let parent_hash = BlockId::Hash(*header.parent_hash());
            api.initialize_block(&parent_hash, &header).map_err(|e| {
                internal_err({
                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                        &["Runtime api access error: "],
                        &[::core::fmt::ArgumentV1::new_debug(&e)],
                    ));
                    res
                })
            })?;
            for xt in xts {
                let _ = api.apply_extrinsic(&best, xt);
            }
            Ok(api)
        } else {
            Err(internal_err({
                let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                    &["Cannot get header for block "],
                    &[::core::fmt::ArgumentV1::new_debug(&best)],
                ));
                res
            }))
        }
    }
}
mod eth_pubsub {
    use std::{collections::BTreeMap, marker::PhantomData, sync::Arc};
    use ethereum::{BlockV2 as EthereumBlock, TransactionV2 as EthereumTransaction};
    use ethereum_types::{H256, U256};
    use futures::{FutureExt as _, StreamExt as _};
    use jsonrpsee::{types::SubscriptionResult, SubscriptionSink};
    use sc_client_api::{
        backend::{Backend, StateBackend, StorageProvider},
        client::BlockchainEvents,
    };
    use sc_network::{NetworkService, NetworkStatusProvider};
    use sc_network_common::ExHashT;
    use sc_rpc::SubscriptionTaskExecutor;
    use sc_transaction_pool_api::TransactionPool;
    use sp_api::{ApiExt, BlockId, ProvideRuntimeApi};
    use sp_blockchain::HeaderBackend;
    use sp_consensus::SyncOracle;
    use sp_core::hashing::keccak_256;
    use sp_runtime::traits::{BlakeTwo256, Block as BlockT, UniqueSaturatedInto};
    use fc_rpc_core::{
        types::{
            pubsub::{Kind, Params, PubSubSyncStatus, Result as PubSubResult, SyncStatusMetadata},
            Bytes, FilteredParams, Header, Log, Rich,
        },
        EthPubSubApiServer,
    };
    use fp_rpc::EthereumRuntimeRPCApi;
    use crate::{frontier_backend_client, overrides::OverrideHandle};
    pub struct EthereumSubIdProvider;
    #[automatically_derived]
    impl ::core::fmt::Debug for EthereumSubIdProvider {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::write_str(f, "EthereumSubIdProvider")
        }
    }
    impl jsonrpsee::core::traits::IdProvider for EthereumSubIdProvider {
        fn next_id(&self) -> jsonrpsee::types::SubscriptionId<'static> {
            {
                let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                    &["0x"],
                    &[::core::fmt::ArgumentV1::new_display(&hex::encode(
                        rand::random::<u128>().to_le_bytes(),
                    ))],
                ));
                res
            }
            .into()
        }
    }
    /// Eth pub-sub API implementation.
    pub struct EthPubSub<B: BlockT, P, C, BE, H: ExHashT> {
        pool: Arc<P>,
        client: Arc<C>,
        network: Arc<NetworkService<B, H>>,
        subscriptions: SubscriptionTaskExecutor,
        overrides: Arc<OverrideHandle<B>>,
        starting_block: u64,
        _marker: PhantomData<BE>,
    }
    impl<B: BlockT, P, C, BE, H: ExHashT> EthPubSub<B, P, C, BE, H>
    where
        C: HeaderBackend<B> + Send + Sync + 'static,
    {
        pub fn new(
            pool: Arc<P>,
            client: Arc<C>,
            network: Arc<NetworkService<B, H>>,
            subscriptions: SubscriptionTaskExecutor,
            overrides: Arc<OverrideHandle<B>>,
        ) -> Self {
            let starting_block =
                UniqueSaturatedInto::<u64>::unique_saturated_into(client.info().best_number);
            Self {
                pool,
                client,
                network,
                subscriptions,
                overrides,
                starting_block,
                _marker: PhantomData,
            }
        }
    }
    struct EthSubscriptionResult;
    impl EthSubscriptionResult {
        pub fn new_heads(block: EthereumBlock) -> PubSubResult {
            PubSubResult::Header(Box::new(Rich {
                inner: Header {
                    hash: Some(H256::from(keccak_256(&rlp::encode(&block.header)))),
                    parent_hash: block.header.parent_hash,
                    uncles_hash: block.header.ommers_hash,
                    author: block.header.beneficiary,
                    miner: block.header.beneficiary,
                    state_root: block.header.state_root,
                    transactions_root: block.header.transactions_root,
                    receipts_root: block.header.receipts_root,
                    number: Some(block.header.number),
                    gas_used: block.header.gas_used,
                    gas_limit: block.header.gas_limit,
                    extra_data: Bytes(block.header.extra_data.clone()),
                    logs_bloom: block.header.logs_bloom,
                    timestamp: U256::from(block.header.timestamp),
                    difficulty: block.header.difficulty,
                    nonce: Some(block.header.nonce),
                    size: Some(U256::from(rlp::encode(&block.header).len() as u32)),
                },
                extra_info: BTreeMap::new(),
            }))
        }
        pub fn logs(
            block: EthereumBlock,
            receipts: Vec<ethereum::ReceiptV3>,
            params: &FilteredParams,
        ) -> Vec<Log> {
            let block_hash = Some(H256::from(keccak_256(&rlp::encode(&block.header))));
            let mut logs: Vec<Log> = ::alloc::vec::Vec::new();
            let mut log_index: u32 = 0;
            for (receipt_index, receipt) in receipts.into_iter().enumerate() {
                let receipt_logs = match receipt {
                    ethereum::ReceiptV3::Legacy(d)
                    | ethereum::ReceiptV3::EIP2930(d)
                    | ethereum::ReceiptV3::EIP1559(d) => d.logs,
                };
                let mut transaction_log_index: u32 = 0;
                let transaction_hash: Option<H256> = if receipt_logs.len() > 0 {
                    Some(block.transactions[receipt_index].hash())
                } else {
                    None
                };
                for log in receipt_logs {
                    if Self::add_log(block_hash.unwrap(), &log, &block, params) {
                        logs.push(Log {
                            address: log.address,
                            topics: log.topics,
                            data: Bytes(log.data),
                            block_hash,
                            block_number: Some(block.header.number),
                            transaction_hash,
                            transaction_index: Some(U256::from(receipt_index)),
                            log_index: Some(U256::from(log_index)),
                            transaction_log_index: Some(U256::from(transaction_log_index)),
                            removed: false,
                        });
                    }
                    log_index += 1;
                    transaction_log_index += 1;
                }
            }
            logs
        }
        fn add_log(
            block_hash: H256,
            ethereum_log: &ethereum::Log,
            block: &EthereumBlock,
            params: &FilteredParams,
        ) -> bool {
            let log = Log {
                address: ethereum_log.address,
                topics: ethereum_log.topics.clone(),
                data: Bytes(ethereum_log.data.clone()),
                block_hash: None,
                block_number: None,
                transaction_hash: None,
                transaction_index: None,
                log_index: None,
                transaction_log_index: None,
                removed: false,
            };
            if params.filter.is_some() {
                let block_number =
                    UniqueSaturatedInto::<u64>::unique_saturated_into(block.header.number);
                if !params.filter_block_range(block_number)
                    || !params.filter_block_hash(block_hash)
                    || !params.filter_address(&log)
                    || !params.filter_topics(&log)
                {
                    return false;
                }
            }
            true
        }
    }
    impl<B: BlockT, P, C, BE, H: ExHashT> EthPubSubApiServer for EthPubSub<B, P, C, BE, H>
    where
        B: BlockT<Hash = H256> + Send + Sync + 'static,
        P: TransactionPool<Block = B> + Send + Sync + 'static,
        C: ProvideRuntimeApi<B> + StorageProvider<B, BE> + BlockchainEvents<B>,
        C: HeaderBackend<B> + Send + Sync + 'static,
        C::Api: EthereumRuntimeRPCApi<B>,
        BE: Backend<B> + 'static,
        BE::State: StateBackend<BlakeTwo256>,
    {
        fn subscribe(
            &self,
            mut sink: SubscriptionSink,
            kind: Kind,
            params: Option<Params>,
        ) -> SubscriptionResult {
            sink.accept()?;
            let filtered_params = match params {
                Some(Params::Logs(filter)) => FilteredParams::new(Some(filter)),
                _ => FilteredParams::default(),
            };
            let client = self.client.clone();
            let pool = self.pool.clone();
            let network = self.network.clone();
            let overrides = self.overrides.clone();
            let starting_block = self.starting_block;
            let fut = async move {
                match kind {
                    Kind::Logs => {
                        let stream = client
                            .import_notification_stream()
                            .filter_map(move |notification| {
                                if notification.is_new_best {
                                    let id = BlockId::Hash(notification.hash);
                                    let schema = frontier_backend_client::onchain_storage_schema::<
                                        B,
                                        C,
                                        BE,
                                    >(
                                        client.as_ref(), id
                                    );
                                    let handler = overrides
                                        .schemas
                                        .get(&schema)
                                        .unwrap_or(&overrides.fallback);
                                    let block = handler.current_block(&id);
                                    let receipts = handler.current_receipts(&id);
                                    match (receipts, block) {
                                        (Some(receipts), Some(block)) => {
                                            futures::future::ready(Some((block, receipts)))
                                        }
                                        _ => futures::future::ready(None),
                                    }
                                } else {
                                    futures::future::ready(None)
                                }
                            })
                            .flat_map(move |(block, receipts)| {
                                futures::stream::iter(EthSubscriptionResult::logs(
                                    block,
                                    receipts,
                                    &filtered_params,
                                ))
                            })
                            .map(|x| PubSubResult::Log(Box::new(x)));
                        sink.pipe_from_stream(stream).await;
                    }
                    Kind::NewHeads => {
                        let stream = client
                            .import_notification_stream()
                            .filter_map(move |notification| {
                                if notification.is_new_best {
                                    let id = BlockId::Hash(notification.hash);
                                    let schema = frontier_backend_client::onchain_storage_schema::<
                                        B,
                                        C,
                                        BE,
                                    >(
                                        client.as_ref(), id
                                    );
                                    let handler = overrides
                                        .schemas
                                        .get(&schema)
                                        .unwrap_or(&overrides.fallback);
                                    let block = handler.current_block(&id);
                                    futures::future::ready(block)
                                } else {
                                    futures::future::ready(None)
                                }
                            })
                            .map(EthSubscriptionResult::new_heads);
                        sink.pipe_from_stream(stream).await;
                    }
                    Kind::NewPendingTransactions => {
                        use sc_transaction_pool_api::InPoolTransaction;
                        let stream = pool
                            .import_notification_stream()
                            .filter_map(move |txhash| {
                                if let Some(xt) = pool.ready_transaction(&txhash) {
                                    let best_block: BlockId<B> =
                                        BlockId::Hash(client.info().best_hash);
                                    let api = client.runtime_api();
                                    let api_version = if let Ok(Some(api_version)) =
                                        api.api_version::<dyn EthereumRuntimeRPCApi<B>>(&best_block)
                                    {
                                        api_version
                                    } else {
                                        return futures::future::ready(None);
                                    };
                                    let xts = <[_]>::into_vec(
                                        #[rustc_box]
                                        ::alloc::boxed::Box::new([xt.data().clone()]),
                                    );
                                    let txs: Option<Vec<EthereumTransaction>> = if api_version > 1 {
                                        api.extrinsic_filter(&best_block, xts).ok()
                                    } else {
                                        #[allow(deprecated)]
                                        if let Ok(legacy) =
                                            api.extrinsic_filter_before_version_2(&best_block, xts)
                                        {
                                            Some(legacy.into_iter().map(|tx| tx.into()).collect())
                                        } else {
                                            None
                                        }
                                    };
                                    let res = match txs {
                                        Some(txs) => {
                                            if txs.len() == 1 {
                                                Some(txs[0].clone())
                                            } else {
                                                None
                                            }
                                        }
                                        _ => None,
                                    };
                                    futures::future::ready(res)
                                } else {
                                    futures::future::ready(None)
                                }
                            })
                            .map(|transaction| PubSubResult::TransactionHash(transaction.hash()));
                        sink.pipe_from_stream(stream).await;
                    }
                    Kind::Syncing => {
                        let client = Arc::clone(&client);
                        let network = Arc::clone(&network);
                        async fn status<
                            C: HeaderBackend<B>,
                            B: BlockT,
                            H: ExHashT + Send + Sync,
                        >(
                            client: Arc<C>,
                            network: Arc<NetworkService<B, H>>,
                            starting_block: u64,
                        ) -> PubSubSyncStatus {
                            if network.is_major_syncing() {
                                let highest_block = network
                                    .status()
                                    .await
                                    .ok()
                                    .and_then(|res| res.best_seen_block)
                                    .map(UniqueSaturatedInto::<u64>::unique_saturated_into);
                                let current_block =
                                    UniqueSaturatedInto::<u64>::unique_saturated_into(
                                        client.info().best_number,
                                    );
                                PubSubSyncStatus::Detailed(SyncStatusMetadata {
                                    syncing: true,
                                    starting_block,
                                    current_block,
                                    highest_block,
                                })
                            } else {
                                PubSubSyncStatus::Simple(false)
                            }
                        }
                        let _ = sink.send(&PubSubResult::SyncState(
                            status(Arc::clone(&client), Arc::clone(&network), starting_block).await,
                        ));
                        let mut stream = client.clone().import_notification_stream();
                        let mut last_syncing_status = network.is_major_syncing();
                        while (stream.next().await).is_some() {
                            let syncing_status = network.is_major_syncing();
                            if syncing_status != last_syncing_status {
                                let _ = sink.send(&PubSubResult::SyncState(
                                    status(client.clone(), network.clone(), starting_block).await,
                                ));
                            }
                            last_syncing_status = syncing_status;
                        }
                    }
                }
            }
            .boxed();
            self.subscriptions.spawn(
                "frontier-rpc-subscription",
                Some("rpc"),
                fut.map(drop).boxed(),
            );
            Ok(())
        }
    }
}
mod net {
    use std::sync::Arc;
    use ethereum_types::H256;
    use jsonrpsee::core::RpcResult as Result;
    use sc_network::NetworkService;
    use sc_network_common::{service::NetworkPeers, ExHashT};
    use sp_api::ProvideRuntimeApi;
    use sp_blockchain::HeaderBackend;
    use sp_runtime::{generic::BlockId, traits::Block as BlockT};
    use fc_rpc_core::{types::PeerCount, NetApiServer};
    use fp_rpc::EthereumRuntimeRPCApi;
    use crate::internal_err;
    /// Net API implementation.
    pub struct Net<B: BlockT, C, H: ExHashT> {
        client: Arc<C>,
        network: Arc<NetworkService<B, H>>,
        peer_count_as_hex: bool,
    }
    impl<B: BlockT, C, H: ExHashT> Net<B, C, H> {
        pub fn new(
            client: Arc<C>,
            network: Arc<NetworkService<B, H>>,
            peer_count_as_hex: bool,
        ) -> Self {
            Self {
                client,
                network,
                peer_count_as_hex,
            }
        }
    }
    impl<B, C, H: ExHashT> NetApiServer for Net<B, C, H>
    where
        B: BlockT<Hash = H256> + Send + Sync + 'static,
        C: HeaderBackend<B> + ProvideRuntimeApi<B> + Send + Sync + 'static,
        C::Api: EthereumRuntimeRPCApi<B>,
    {
        fn version(&self) -> Result<String> {
            let hash = self.client.info().best_hash;
            Ok(self
                .client
                .runtime_api()
                .chain_id(&BlockId::Hash(hash))
                .map_err(|_| internal_err("fetch runtime chain id failed"))?
                .to_string())
        }
        fn peer_count(&self) -> Result<PeerCount> {
            let peer_count = self.network.sync_num_connected();
            Ok(match self.peer_count_as_hex {
                true => PeerCount::String({
                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                        &["0x"],
                        &[::core::fmt::ArgumentV1::new_lower_hex(&peer_count)],
                    ));
                    res
                }),
                false => PeerCount::U32(peer_count as u32),
            })
        }
        fn is_listening(&self) -> Result<bool> {
            Ok(true)
        }
    }
}
mod overrides {
    use std::{collections::BTreeMap, marker::PhantomData, sync::Arc};
    use ethereum::BlockV2 as EthereumBlock;
    use ethereum_types::{H160, H256, U256};
    use sp_api::{ApiExt, BlockId, ProvideRuntimeApi};
    use sp_io::hashing::{blake2_128, twox_128};
    use sp_runtime::{traits::Block as BlockT, Permill};
    use fp_rpc::{EthereumRuntimeRPCApi, TransactionStatus};
    use fp_storage::EthereumStorageSchema;
    mod schema_v1_override {
        use std::{marker::PhantomData, sync::Arc};
        use ethereum_types::{H160, H256, U256};
        use scale_codec::Decode;
        use sc_client_api::backend::{Backend, StateBackend, StorageProvider};
        use sp_api::BlockId;
        use sp_blockchain::HeaderBackend;
        use sp_runtime::{
            traits::{BlakeTwo256, Block as BlockT},
            Permill,
        };
        use sp_storage::StorageKey;
        use fp_rpc::TransactionStatus;
        use fp_storage::*;
        use super::{blake2_128_extend, storage_prefix_build, StorageOverride};
        /// An override for runtimes that use Schema V1
        pub struct SchemaV1Override<B: BlockT, C, BE> {
            client: Arc<C>,
            _marker: PhantomData<(B, BE)>,
        }
        impl<B: BlockT, C, BE> SchemaV1Override<B, C, BE> {
            pub fn new(client: Arc<C>) -> Self {
                Self {
                    client,
                    _marker: PhantomData,
                }
            }
        }
        impl<B, C, BE> SchemaV1Override<B, C, BE>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: StorageProvider<B, BE> + HeaderBackend<B> + Send + Sync + 'static,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
        {
            fn query_storage<T: Decode>(&self, id: &BlockId<B>, key: &StorageKey) -> Option<T> {
                if let Ok(Some(hash)) = self.client.block_hash_from_id(id) {
                    if let Ok(Some(data)) = self.client.storage(hash, key) {
                        if let Ok(result) = Decode::decode(&mut &data.0[..]) {
                            return Some(result);
                        }
                    }
                }
                None
            }
        }
        impl<B, C, BE> StorageOverride<B> for SchemaV1Override<B, C, BE>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: StorageProvider<B, BE> + HeaderBackend<B> + Send + Sync + 'static,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
        {
            /// For a given account address, returns pallet_evm::AccountCodes.
            fn account_code_at(&self, block: &BlockId<B>, address: H160) -> Option<Vec<u8>> {
                let mut key: Vec<u8> = storage_prefix_build(PALLET_EVM, EVM_ACCOUNT_CODES);
                key.extend(blake2_128_extend(address.as_bytes()));
                self.query_storage::<Vec<u8>>(block, &StorageKey(key))
            }
            /// For a given account address and index, returns pallet_evm::AccountStorages.
            fn storage_at(&self, block: &BlockId<B>, address: H160, index: U256) -> Option<H256> {
                let tmp: &mut [u8; 32] = &mut [0; 32];
                index.to_big_endian(tmp);
                let mut key: Vec<u8> = storage_prefix_build(PALLET_EVM, EVM_ACCOUNT_STORAGES);
                key.extend(blake2_128_extend(address.as_bytes()));
                key.extend(blake2_128_extend(tmp));
                self.query_storage::<H256>(block, &StorageKey(key))
            }
            /// Return the current block.
            fn current_block(&self, block: &BlockId<B>) -> Option<ethereum::BlockV2> {
                self.query_storage::<ethereum::BlockV0>(
                    block,
                    &StorageKey(storage_prefix_build(
                        PALLET_ETHEREUM,
                        ETHEREUM_CURRENT_BLOCK,
                    )),
                )
                .map(Into::into)
            }
            /// Return the current receipt.
            fn current_receipts(&self, block: &BlockId<B>) -> Option<Vec<ethereum::ReceiptV3>> {
                self.query_storage::<Vec<ethereum::ReceiptV0>>(
                    block,
                    &StorageKey(storage_prefix_build(
                        PALLET_ETHEREUM,
                        ETHEREUM_CURRENT_RECEIPTS,
                    )),
                )
                .map(|receipts| {
                    receipts
                        .into_iter()
                        .map(|r| {
                            ethereum::ReceiptV3::Legacy(ethereum::EIP658ReceiptData {
                                status_code: r.state_root.to_low_u64_be() as u8,
                                used_gas: r.used_gas,
                                logs_bloom: r.logs_bloom,
                                logs: r.logs,
                            })
                        })
                        .collect()
                })
            }
            /// Return the current transaction status.
            fn current_transaction_statuses(
                &self,
                block: &BlockId<B>,
            ) -> Option<Vec<TransactionStatus>> {
                self.query_storage::<Vec<TransactionStatus>>(
                    block,
                    &StorageKey(storage_prefix_build(
                        PALLET_ETHEREUM,
                        ETHEREUM_CURRENT_TRANSACTION_STATUS,
                    )),
                )
            }
            /// Prior to eip-1559 there is no elasticity.
            fn elasticity(&self, _block: &BlockId<B>) -> Option<Permill> {
                None
            }
            fn is_eip1559(&self, _block: &BlockId<B>) -> bool {
                false
            }
        }
    }
    mod schema_v2_override {
        use std::{marker::PhantomData, sync::Arc};
        use ethereum_types::{H160, H256, U256};
        use scale_codec::Decode;
        use sc_client_api::backend::{Backend, StateBackend, StorageProvider};
        use sp_api::BlockId;
        use sp_blockchain::HeaderBackend;
        use sp_runtime::{
            traits::{BlakeTwo256, Block as BlockT},
            Permill,
        };
        use sp_storage::StorageKey;
        use fp_rpc::TransactionStatus;
        use fp_storage::*;
        use super::{blake2_128_extend, storage_prefix_build, StorageOverride};
        /// An override for runtimes that use Schema V2
        pub struct SchemaV2Override<B: BlockT, C, BE> {
            client: Arc<C>,
            _marker: PhantomData<(B, BE)>,
        }
        impl<B: BlockT, C, BE> SchemaV2Override<B, C, BE> {
            pub fn new(client: Arc<C>) -> Self {
                Self {
                    client,
                    _marker: PhantomData,
                }
            }
        }
        impl<B, C, BE> SchemaV2Override<B, C, BE>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: StorageProvider<B, BE> + HeaderBackend<B> + Send + Sync + 'static,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
        {
            fn query_storage<T: Decode>(&self, id: &BlockId<B>, key: &StorageKey) -> Option<T> {
                if let Ok(Some(hash)) = self.client.block_hash_from_id(id) {
                    if let Ok(Some(data)) = self.client.storage(hash, key) {
                        if let Ok(result) = Decode::decode(&mut &data.0[..]) {
                            return Some(result);
                        }
                    }
                }
                None
            }
        }
        impl<B, C, BE> StorageOverride<B> for SchemaV2Override<B, C, BE>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: StorageProvider<B, BE> + HeaderBackend<B> + Send + Sync + 'static,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
        {
            /// For a given account address, returns pallet_evm::AccountCodes.
            fn account_code_at(&self, block: &BlockId<B>, address: H160) -> Option<Vec<u8>> {
                let mut key: Vec<u8> = storage_prefix_build(PALLET_EVM, EVM_ACCOUNT_CODES);
                key.extend(blake2_128_extend(address.as_bytes()));
                self.query_storage::<Vec<u8>>(block, &StorageKey(key))
            }
            /// For a given account address and index, returns pallet_evm::AccountStorages.
            fn storage_at(&self, block: &BlockId<B>, address: H160, index: U256) -> Option<H256> {
                let tmp: &mut [u8; 32] = &mut [0; 32];
                index.to_big_endian(tmp);
                let mut key: Vec<u8> = storage_prefix_build(PALLET_EVM, EVM_ACCOUNT_STORAGES);
                key.extend(blake2_128_extend(address.as_bytes()));
                key.extend(blake2_128_extend(tmp));
                self.query_storage::<H256>(block, &StorageKey(key))
            }
            /// Return the current block.
            fn current_block(&self, block: &BlockId<B>) -> Option<ethereum::BlockV2> {
                self.query_storage::<ethereum::BlockV2>(
                    block,
                    &StorageKey(storage_prefix_build(
                        PALLET_ETHEREUM,
                        ETHEREUM_CURRENT_BLOCK,
                    )),
                )
            }
            /// Return the current receipt.
            fn current_receipts(&self, block: &BlockId<B>) -> Option<Vec<ethereum::ReceiptV3>> {
                self.query_storage::<Vec<ethereum::ReceiptV0>>(
                    block,
                    &StorageKey(storage_prefix_build(
                        PALLET_ETHEREUM,
                        ETHEREUM_CURRENT_RECEIPTS,
                    )),
                )
                .map(|receipts| {
                    receipts
                        .into_iter()
                        .map(|r| {
                            ethereum::ReceiptV3::Legacy(ethereum::EIP658ReceiptData {
                                status_code: r.state_root.to_low_u64_be() as u8,
                                used_gas: r.used_gas,
                                logs_bloom: r.logs_bloom,
                                logs: r.logs,
                            })
                        })
                        .collect()
                })
            }
            /// Return the current transaction status.
            fn current_transaction_statuses(
                &self,
                block: &BlockId<B>,
            ) -> Option<Vec<TransactionStatus>> {
                self.query_storage::<Vec<TransactionStatus>>(
                    block,
                    &StorageKey(storage_prefix_build(
                        PALLET_ETHEREUM,
                        ETHEREUM_CURRENT_TRANSACTION_STATUS,
                    )),
                )
            }
            /// Return the elasticity at the given height.
            fn elasticity(&self, block: &BlockId<B>) -> Option<Permill> {
                let default_elasticity = Some(Permill::from_parts(125_000));
                let elasticity = self.query_storage::<Permill>(
                    block,
                    &StorageKey(storage_prefix_build(PALLET_BASE_FEE, BASE_FEE_ELASTICITY)),
                );
                if elasticity.is_some() {
                    elasticity
                } else {
                    default_elasticity
                }
            }
            fn is_eip1559(&self, _block: &BlockId<B>) -> bool {
                true
            }
        }
    }
    mod schema_v3_override {
        use std::{marker::PhantomData, sync::Arc};
        use ethereum_types::{H160, H256, U256};
        use scale_codec::Decode;
        use sc_client_api::backend::{Backend, StateBackend, StorageProvider};
        use sp_api::BlockId;
        use sp_blockchain::HeaderBackend;
        use sp_runtime::{
            traits::{BlakeTwo256, Block as BlockT},
            Permill,
        };
        use sp_storage::StorageKey;
        use fp_rpc::TransactionStatus;
        use fp_storage::*;
        use super::{blake2_128_extend, storage_prefix_build, StorageOverride};
        /// An override for runtimes that use Schema V3
        pub struct SchemaV3Override<B: BlockT, C, BE> {
            client: Arc<C>,
            _marker: PhantomData<(B, BE)>,
        }
        impl<B: BlockT, C, BE> SchemaV3Override<B, C, BE> {
            pub fn new(client: Arc<C>) -> Self {
                Self {
                    client,
                    _marker: PhantomData,
                }
            }
        }
        impl<B, C, BE> SchemaV3Override<B, C, BE>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: StorageProvider<B, BE> + HeaderBackend<B> + Send + Sync + 'static,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
        {
            fn query_storage<T: Decode>(&self, id: &BlockId<B>, key: &StorageKey) -> Option<T> {
                if let Ok(Some(hash)) = self.client.block_hash_from_id(id) {
                    if let Ok(Some(data)) = self.client.storage(hash, key) {
                        if let Ok(result) = Decode::decode(&mut &data.0[..]) {
                            return Some(result);
                        }
                    }
                }
                None
            }
        }
        impl<B, C, BE> StorageOverride<B> for SchemaV3Override<B, C, BE>
        where
            B: BlockT<Hash = H256> + Send + Sync + 'static,
            C: StorageProvider<B, BE> + HeaderBackend<B> + Send + Sync + 'static,
            BE: Backend<B> + 'static,
            BE::State: StateBackend<BlakeTwo256>,
        {
            /// For a given account address, returns pallet_evm::AccountCodes.
            fn account_code_at(&self, block: &BlockId<B>, address: H160) -> Option<Vec<u8>> {
                let mut key: Vec<u8> = storage_prefix_build(PALLET_EVM, EVM_ACCOUNT_CODES);
                key.extend(blake2_128_extend(address.as_bytes()));
                self.query_storage::<Vec<u8>>(block, &StorageKey(key))
            }
            /// For a given account address and index, returns pallet_evm::AccountStorages.
            fn storage_at(&self, block: &BlockId<B>, address: H160, index: U256) -> Option<H256> {
                let tmp: &mut [u8; 32] = &mut [0; 32];
                index.to_big_endian(tmp);
                let mut key: Vec<u8> = storage_prefix_build(PALLET_EVM, EVM_ACCOUNT_STORAGES);
                key.extend(blake2_128_extend(address.as_bytes()));
                key.extend(blake2_128_extend(tmp));
                self.query_storage::<H256>(block, &StorageKey(key))
            }
            /// Return the current block.
            fn current_block(&self, block: &BlockId<B>) -> Option<ethereum::BlockV2> {
                self.query_storage::<ethereum::BlockV2>(
                    block,
                    &StorageKey(storage_prefix_build(
                        PALLET_ETHEREUM,
                        ETHEREUM_CURRENT_BLOCK,
                    )),
                )
            }
            /// Return the current receipt.
            fn current_receipts(&self, block: &BlockId<B>) -> Option<Vec<ethereum::ReceiptV3>> {
                self.query_storage::<Vec<ethereum::ReceiptV3>>(
                    block,
                    &StorageKey(storage_prefix_build(
                        PALLET_ETHEREUM,
                        ETHEREUM_CURRENT_RECEIPTS,
                    )),
                )
            }
            /// Return the current transaction status.
            fn current_transaction_statuses(
                &self,
                block: &BlockId<B>,
            ) -> Option<Vec<TransactionStatus>> {
                self.query_storage::<Vec<TransactionStatus>>(
                    block,
                    &StorageKey(storage_prefix_build(
                        PALLET_ETHEREUM,
                        ETHEREUM_CURRENT_TRANSACTION_STATUS,
                    )),
                )
            }
            /// Return the elasticity at the given height.
            fn elasticity(&self, block: &BlockId<B>) -> Option<Permill> {
                let default_elasticity = Some(Permill::from_parts(125_000));
                let elasticity = self.query_storage::<Permill>(
                    block,
                    &StorageKey(storage_prefix_build(PALLET_BASE_FEE, BASE_FEE_ELASTICITY)),
                );
                if elasticity.is_some() {
                    elasticity
                } else {
                    default_elasticity
                }
            }
            fn is_eip1559(&self, _block: &BlockId<B>) -> bool {
                true
            }
        }
    }
    pub use schema_v1_override::SchemaV1Override;
    pub use schema_v2_override::SchemaV2Override;
    pub use schema_v3_override::SchemaV3Override;
    pub struct OverrideHandle<Block: BlockT> {
        pub schemas: BTreeMap<EthereumStorageSchema, Box<dyn StorageOverride<Block> + Send + Sync>>,
        pub fallback: Box<dyn StorageOverride<Block> + Send + Sync>,
    }
    /// Something that can fetch Ethereum-related data. This trait is quite similar to the runtime API,
    /// and indeed oe implementation of it uses the runtime API.
    /// Having this trait is useful because it allows optimized implementations that fetch data from a
    /// State Backend with some assumptions about pallet-ethereum's storage schema. Using such an
    /// optimized implementation avoids spawning a runtime and the overhead associated with it.
    pub trait StorageOverride<Block: BlockT> {
        /// For a given account address, returns pallet_evm::AccountCodes.
        fn account_code_at(&self, block: &BlockId<Block>, address: H160) -> Option<Vec<u8>>;
        /// For a given account address and index, returns pallet_evm::AccountStorages.
        fn storage_at(&self, block: &BlockId<Block>, address: H160, index: U256) -> Option<H256>;
        /// Return the current block.
        fn current_block(&self, block: &BlockId<Block>) -> Option<EthereumBlock>;
        /// Return the current receipt.
        fn current_receipts(&self, block: &BlockId<Block>) -> Option<Vec<ethereum::ReceiptV3>>;
        /// Return the current transaction status.
        fn current_transaction_statuses(
            &self,
            block: &BlockId<Block>,
        ) -> Option<Vec<TransactionStatus>>;
        /// Return the base fee at the given height.
        fn elasticity(&self, block: &BlockId<Block>) -> Option<Permill>;
        /// Return `true` if the request BlockId is post-eip1559.
        fn is_eip1559(&self, block: &BlockId<Block>) -> bool;
    }
    fn storage_prefix_build(module: &[u8], storage: &[u8]) -> Vec<u8> {
        [twox_128(module), twox_128(storage)].concat().to_vec()
    }
    fn blake2_128_extend(bytes: &[u8]) -> Vec<u8> {
        let mut ext: Vec<u8> = blake2_128(bytes).to_vec();
        ext.extend_from_slice(bytes);
        ext
    }
    /// A wrapper type for the Runtime API. This type implements `StorageOverride`, so it can be used
    /// when calling the runtime API is desired but a `dyn StorageOverride` is required.
    pub struct RuntimeApiStorageOverride<B: BlockT, C> {
        client: Arc<C>,
        _marker: PhantomData<B>,
    }
    impl<B: BlockT, C> RuntimeApiStorageOverride<B, C> {
        pub fn new(client: Arc<C>) -> Self {
            Self {
                client,
                _marker: PhantomData,
            }
        }
    }
    impl<Block, C> StorageOverride<Block> for RuntimeApiStorageOverride<Block, C>
    where
        Block: BlockT<Hash = H256> + Send + Sync + 'static,
        C: ProvideRuntimeApi<Block> + Send + Sync + 'static,
        C::Api: EthereumRuntimeRPCApi<Block>,
    {
        /// For a given account address, returns pallet_evm::AccountCodes.
        fn account_code_at(&self, block: &BlockId<Block>, address: H160) -> Option<Vec<u8>> {
            self.client
                .runtime_api()
                .account_code_at(block, address)
                .ok()
        }
        /// For a given account address and index, returns pallet_evm::AccountStorages.
        fn storage_at(&self, block: &BlockId<Block>, address: H160, index: U256) -> Option<H256> {
            self.client
                .runtime_api()
                .storage_at(block, address, index)
                .ok()
        }
        /// Return the current block.
        fn current_block(&self, block: &BlockId<Block>) -> Option<ethereum::BlockV2> {
            let api = self.client.runtime_api();
            let api_version = if let Ok(Some(api_version)) =
                api.api_version::<dyn EthereumRuntimeRPCApi<Block>>(block)
            {
                api_version
            } else {
                return None;
            };
            if api_version == 1 {
                #[allow(deprecated)]
                let old_block = api.current_block_before_version_2(block).ok()?;
                old_block.map(|block| block.into())
            } else {
                api.current_block(block).ok()?
            }
        }
        /// Return the current receipt.
        fn current_receipts(&self, block: &BlockId<Block>) -> Option<Vec<ethereum::ReceiptV3>> {
            let api = self.client.runtime_api();
            let api_version = if let Ok(Some(api_version)) =
                api.api_version::<dyn EthereumRuntimeRPCApi<Block>>(block)
            {
                api_version
            } else {
                return None;
            };
            if api_version < 4 {
                #[allow(deprecated)]
                let old_receipts = api.current_receipts_before_version_4(block).ok()?;
                old_receipts.map(|receipts| {
                    receipts
                        .into_iter()
                        .map(|r| {
                            ethereum::ReceiptV3::Legacy(ethereum::EIP658ReceiptData {
                                status_code: r.state_root.to_low_u64_be() as u8,
                                used_gas: r.used_gas,
                                logs_bloom: r.logs_bloom,
                                logs: r.logs,
                            })
                        })
                        .collect()
                })
            } else {
                self.client.runtime_api().current_receipts(block).ok()?
            }
        }
        /// Return the current transaction status.
        fn current_transaction_statuses(
            &self,
            block: &BlockId<Block>,
        ) -> Option<Vec<TransactionStatus>> {
            self.client
                .runtime_api()
                .current_transaction_statuses(block)
                .ok()?
        }
        /// Return the elasticity multiplier at the give post-eip1559 height.
        fn elasticity(&self, block: &BlockId<Block>) -> Option<Permill> {
            if self.is_eip1559(block) {
                self.client.runtime_api().elasticity(block).ok()?
            } else {
                None
            }
        }
        fn is_eip1559(&self, block: &BlockId<Block>) -> bool {
            if let Ok(Some(api_version)) = self
                .client
                .runtime_api()
                .api_version::<dyn EthereumRuntimeRPCApi<Block>>(block)
            {
                return api_version >= 2;
            }
            false
        }
    }
}
mod signer {
    use ethereum::TransactionV2 as EthereumTransaction;
    use ethereum_types::{H160, H256};
    use jsonrpsee::core::Error;
    use sp_core::hashing::keccak_256;
    use fc_rpc_core::types::TransactionMessage;
    use crate::internal_err;
    /// A generic Ethereum signer.
    pub trait EthSigner: Send + Sync {
        /// Available accounts from this signer.
        fn accounts(&self) -> Vec<H160>;
        /// Sign a transaction message using the given account in message.
        fn sign(
            &self,
            message: TransactionMessage,
            address: &H160,
        ) -> Result<EthereumTransaction, Error>;
    }
    pub struct EthDevSigner {
        keys: Vec<libsecp256k1::SecretKey>,
    }
    impl EthDevSigner {
        pub fn new() -> Self {
            Self {
                keys: <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([libsecp256k1::SecretKey::parse(&[
                        0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
                        0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
                        0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
                    ])
                    .expect("Test key is valid; qed")]),
                ),
            }
        }
    }
    fn secret_key_address(secret: &libsecp256k1::SecretKey) -> H160 {
        let public = libsecp256k1::PublicKey::from_secret_key(secret);
        public_key_address(&public)
    }
    fn public_key_address(public: &libsecp256k1::PublicKey) -> H160 {
        let mut res = [0u8; 64];
        res.copy_from_slice(&public.serialize()[1..65]);
        H160::from(H256::from(keccak_256(&res)))
    }
    impl EthSigner for EthDevSigner {
        fn accounts(&self) -> Vec<H160> {
            self.keys.iter().map(secret_key_address).collect()
        }
        fn sign(
            &self,
            message: TransactionMessage,
            address: &H160,
        ) -> Result<EthereumTransaction, Error> {
            let mut transaction = None;
            for secret in &self.keys {
                let key_address = secret_key_address(secret);
                if &key_address == address {
                    match message {
                        TransactionMessage::Legacy(m) => {
                            let signing_message = libsecp256k1::Message::parse_slice(&m.hash()[..])
                                .map_err(|_| internal_err("invalid signing message"))?;
                            let (signature, recid) = libsecp256k1::sign(&signing_message, secret);
                            let v = match m.chain_id {
                                None => 27 + recid.serialize() as u64,
                                Some(chain_id) => 2 * chain_id + 35 + recid.serialize() as u64,
                            };
                            let rs = signature.serialize();
                            let r = H256::from_slice(&rs[0..32]);
                            let s = H256::from_slice(&rs[32..64]);
                            transaction =
                                Some(EthereumTransaction::Legacy(ethereum::LegacyTransaction {
                                    nonce: m.nonce,
                                    gas_price: m.gas_price,
                                    gas_limit: m.gas_limit,
                                    action: m.action,
                                    value: m.value,
                                    input: m.input,
                                    signature: ethereum::TransactionSignature::new(v, r, s)
                                        .ok_or_else(|| {
                                            internal_err("signer generated invalid signature")
                                        })?,
                                }));
                        }
                        TransactionMessage::EIP2930(m) => {
                            let signing_message = libsecp256k1::Message::parse_slice(&m.hash()[..])
                                .map_err(|_| internal_err("invalid signing message"))?;
                            let (signature, recid) = libsecp256k1::sign(&signing_message, secret);
                            let rs = signature.serialize();
                            let r = H256::from_slice(&rs[0..32]);
                            let s = H256::from_slice(&rs[32..64]);
                            transaction =
                                Some(EthereumTransaction::EIP2930(ethereum::EIP2930Transaction {
                                    chain_id: m.chain_id,
                                    nonce: m.nonce,
                                    gas_price: m.gas_price,
                                    gas_limit: m.gas_limit,
                                    action: m.action,
                                    value: m.value,
                                    input: m.input.clone(),
                                    access_list: m.access_list,
                                    odd_y_parity: recid.serialize() != 0,
                                    r,
                                    s,
                                }));
                        }
                        TransactionMessage::EIP1559(m) => {
                            let signing_message = libsecp256k1::Message::parse_slice(&m.hash()[..])
                                .map_err(|_| internal_err("invalid signing message"))?;
                            let (signature, recid) = libsecp256k1::sign(&signing_message, secret);
                            let rs = signature.serialize();
                            let r = H256::from_slice(&rs[0..32]);
                            let s = H256::from_slice(&rs[32..64]);
                            transaction =
                                Some(EthereumTransaction::EIP1559(ethereum::EIP1559Transaction {
                                    chain_id: m.chain_id,
                                    nonce: m.nonce,
                                    max_priority_fee_per_gas: m.max_priority_fee_per_gas,
                                    max_fee_per_gas: m.max_fee_per_gas,
                                    gas_limit: m.gas_limit,
                                    action: m.action,
                                    value: m.value,
                                    input: m.input.clone(),
                                    access_list: m.access_list,
                                    odd_y_parity: recid.serialize() != 0,
                                    r,
                                    s,
                                }));
                        }
                    }
                    break;
                }
            }
            transaction.ok_or_else(|| internal_err("signer not available"))
        }
    }
}
mod web3 {
    use std::{marker::PhantomData, sync::Arc};
    use ethereum_types::H256;
    use jsonrpsee::core::RpcResult as Result;
    use sp_api::{Core, ProvideRuntimeApi};
    use sp_blockchain::HeaderBackend;
    use sp_core::keccak_256;
    use sp_runtime::{generic::BlockId, traits::Block as BlockT};
    use fc_rpc_core::{types::Bytes, Web3ApiServer};
    use fp_rpc::EthereumRuntimeRPCApi;
    use crate::internal_err;
    /// Web3 API implementation.
    pub struct Web3<B, C> {
        client: Arc<C>,
        _marker: PhantomData<B>,
    }
    impl<B, C> Web3<B, C> {
        pub fn new(client: Arc<C>) -> Self {
            Self {
                client,
                _marker: PhantomData,
            }
        }
    }
    impl<B, C> Web3ApiServer for Web3<B, C>
    where
        B: BlockT<Hash = H256> + Send + Sync + 'static,
        C: HeaderBackend<B> + ProvideRuntimeApi<B> + Send + Sync + 'static,
        C::Api: EthereumRuntimeRPCApi<B>,
    {
        fn client_version(&self) -> Result<String> {
            let hash = self.client.info().best_hash;
            let version = self
                .client
                .runtime_api()
                .version(&BlockId::Hash(hash))
                .map_err(|err| {
                    internal_err({
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &["fetch runtime version failed: "],
                            &[::core::fmt::ArgumentV1::new_debug(&err)],
                        ));
                        res
                    })
                })?;
            Ok({
                let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                    &["", "/v", ".", "/", "-"],
                    &match (
                        &version.spec_name,
                        &version.spec_version,
                        &version.impl_version,
                        &"fc-rpc",
                        &"2.0.0-dev",
                    ) {
                        args => [
                            ::core::fmt::ArgumentV1::new_display(args.0),
                            ::core::fmt::ArgumentV1::new_display(args.1),
                            ::core::fmt::ArgumentV1::new_display(args.2),
                            ::core::fmt::ArgumentV1::new_display(args.3),
                            ::core::fmt::ArgumentV1::new_display(args.4),
                        ],
                    },
                ));
                res
            })
        }
        fn sha3(&self, input: Bytes) -> Result<H256> {
            Ok(H256::from(keccak_256(&input.into_vec())))
        }
    }
}
pub use self::{
    eth::{format, EstimateGasAdapter, Eth, EthBlockDataCacheTask, EthFilter, EthTask},
    eth_pubsub::{EthPubSub, EthereumSubIdProvider},
    net::Net,
    overrides::{
        OverrideHandle, RuntimeApiStorageOverride, SchemaV1Override, SchemaV2Override,
        SchemaV3Override, StorageOverride,
    },
    signer::{EthDevSigner, EthSigner},
    web3::Web3,
};
pub use ethereum::TransactionV2 as EthereumTransaction;
pub use fc_rpc_core::{
    EthApiServer, EthFilterApiServer, EthPubSubApiServer, NetApiServer, Web3ApiServer,
};
pub mod frontier_backend_client {
    use super::internal_err;
    use ethereum_types::H256;
    use jsonrpsee::core::RpcResult;
    use scale_codec::Decode;
    use sc_client_api::backend::{Backend, StateBackend, StorageProvider};
    use sp_blockchain::HeaderBackend;
    use sp_runtime::{
        generic::BlockId,
        traits::{BlakeTwo256, Block as BlockT, UniqueSaturatedInto, Zero},
    };
    use sp_storage::StorageKey;
    use fc_rpc_core::types::BlockNumber;
    use fp_storage::{EthereumStorageSchema, PALLET_ETHEREUM_SCHEMA};
    pub fn native_block_id<B: BlockT, C>(
        client: &C,
        backend: &fc_db::Backend<B>,
        number: Option<BlockNumber>,
    ) -> RpcResult<Option<BlockId<B>>>
    where
        B: BlockT<Hash = H256> + Send + Sync + 'static,
        C: HeaderBackend<B> + Send + Sync + 'static,
    {
        Ok(match number.unwrap_or(BlockNumber::Latest) {
            BlockNumber::Hash { hash, .. } => {
                load_hash::<B, C>(client, backend, hash).unwrap_or(None)
            }
            BlockNumber::Num(number) => Some(BlockId::Number(number.unique_saturated_into())),
            BlockNumber::Latest => Some(BlockId::Hash(client.info().best_hash)),
            BlockNumber::Earliest => Some(BlockId::Number(Zero::zero())),
            BlockNumber::Pending => None,
            BlockNumber::Safe => Some(BlockId::Hash(client.info().finalized_hash)),
            BlockNumber::Finalized => Some(BlockId::Hash(client.info().finalized_hash)),
        })
    }
    pub fn load_hash<B: BlockT, C>(
        client: &C,
        backend: &fc_db::Backend<B>,
        hash: H256,
    ) -> RpcResult<Option<BlockId<B>>>
    where
        B: BlockT<Hash = H256> + Send + Sync + 'static,
        C: HeaderBackend<B> + Send + Sync + 'static,
    {
        let substrate_hashes = backend.mapping().block_hash(&hash).map_err(|err| {
            internal_err({
                let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                    &["fetch aux store failed: "],
                    &[::core::fmt::ArgumentV1::new_debug(&err)],
                ));
                res
            })
        })?;
        if let Some(substrate_hashes) = substrate_hashes {
            for substrate_hash in substrate_hashes {
                if is_canon::<B, C>(client, substrate_hash) {
                    return Ok(Some(BlockId::Hash(substrate_hash)));
                }
            }
        }
        Ok(None)
    }
    pub fn load_cached_schema<B: BlockT, C>(
        backend: &fc_db::Backend<B>,
    ) -> RpcResult<Option<Vec<(EthereumStorageSchema, H256)>>>
    where
        B: BlockT<Hash = H256> + Send + Sync + 'static,
        C: HeaderBackend<B> + Send + Sync + 'static,
    {
        let cache = backend.meta().ethereum_schema().map_err(|err| {
            internal_err({
                let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                    &["fetch backend failed: "],
                    &[::core::fmt::ArgumentV1::new_debug(&err)],
                ));
                res
            })
        })?;
        Ok(cache)
    }
    pub fn write_cached_schema<B: BlockT, C>(
        backend: &fc_db::Backend<B>,
        new_cache: Vec<(EthereumStorageSchema, H256)>,
    ) -> RpcResult<()>
    where
        B: BlockT<Hash = H256> + Send + Sync + 'static,
        C: HeaderBackend<B> + Send + Sync + 'static,
    {
        backend
            .meta()
            .write_ethereum_schema(new_cache)
            .map_err(|err| {
                internal_err({
                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                        &["write backend failed: "],
                        &[::core::fmt::ArgumentV1::new_debug(&err)],
                    ));
                    res
                })
            })?;
        Ok(())
    }
    pub fn onchain_storage_schema<B: BlockT, C, BE>(
        client: &C,
        at: BlockId<B>,
    ) -> EthereumStorageSchema
    where
        B: BlockT<Hash = H256> + Send + Sync + 'static,
        C: StorageProvider<B, BE> + HeaderBackend<B> + Send + Sync + 'static,
        BE: Backend<B> + 'static,
        BE::State: StateBackend<BlakeTwo256>,
    {
        if let Ok(Some(hash)) = client.block_hash_from_id(&at) {
            match client.storage(hash, &StorageKey(PALLET_ETHEREUM_SCHEMA.to_vec())) {
                Ok(Some(bytes)) => Decode::decode(&mut &bytes.0[..])
                    .ok()
                    .unwrap_or(EthereumStorageSchema::Undefined),
                _ => EthereumStorageSchema::Undefined,
            }
        } else {
            EthereumStorageSchema::Undefined
        }
    }
    pub fn is_canon<B: BlockT, C>(client: &C, target_hash: H256) -> bool
    where
        B: BlockT<Hash = H256> + Send + Sync + 'static,
        C: HeaderBackend<B> + Send + Sync + 'static,
    {
        if let Ok(Some(number)) = client.number(target_hash) {
            if let Ok(Some(hash)) = client.hash(number) {
                return hash == target_hash;
            }
        }
        false
    }
    pub fn load_transactions<B: BlockT, C>(
        client: &C,
        backend: &fc_db::Backend<B>,
        transaction_hash: H256,
        only_canonical: bool,
    ) -> RpcResult<Option<(H256, u32)>>
    where
        B: BlockT<Hash = H256> + Send + Sync + 'static,
        C: HeaderBackend<B> + Send + Sync + 'static,
    {
        let transaction_metadata = backend
            .mapping()
            .transaction_metadata(&transaction_hash)
            .map_err(|err| {
                internal_err({
                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                        &["fetch aux store failed: "],
                        &[::core::fmt::ArgumentV1::new_debug(&err)],
                    ));
                    res
                })
            })?;
        transaction_metadata
            .iter()
            .find(|meta| is_canon::<B, C>(client, meta.block_hash))
            .map_or_else(
                || {
                    if !only_canonical && transaction_metadata.len() > 0 {
                        Ok(Some((
                            transaction_metadata[0].ethereum_block_hash,
                            transaction_metadata[0].ethereum_index,
                        )))
                    } else {
                        Ok(None)
                    }
                },
                |meta| Ok(Some((meta.ethereum_block_hash, meta.ethereum_index))),
            )
    }
}
pub fn err<T: ToString>(code: i32, message: T, data: Option<&[u8]>) -> jsonrpsee::core::Error {
    jsonrpsee::core::Error::Call(jsonrpsee::types::error::CallError::Custom(
        jsonrpsee::types::error::ErrorObject::owned(
            code,
            message.to_string(),
            data.map(|bytes| {
                jsonrpsee::core::to_json_raw_value(&{
                    let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                        &["0x"],
                        &[::core::fmt::ArgumentV1::new_display(&hex::encode(bytes))],
                    ));
                    res
                })
                .expect("fail to serialize data")
            }),
        ),
    ))
}
pub fn internal_err<T: ToString>(message: T) -> jsonrpsee::core::Error {
    err(jsonrpsee::types::error::INTERNAL_ERROR_CODE, message, None)
}
pub fn internal_err_with_data<T: ToString>(message: T, data: &[u8]) -> jsonrpsee::core::Error {
    err(
        jsonrpsee::types::error::INTERNAL_ERROR_CODE,
        message,
        Some(data),
    )
}
pub fn public_key(transaction: &EthereumTransaction) -> Result<[u8; 64], sp_io::EcdsaVerifyError> {
    let mut sig = [0u8; 65];
    let mut msg = [0u8; 32];
    match transaction {
        EthereumTransaction::Legacy(t) => {
            sig[0..32].copy_from_slice(&t.signature.r()[..]);
            sig[32..64].copy_from_slice(&t.signature.s()[..]);
            sig[64] = t.signature.standard_v();
            msg.copy_from_slice(&ethereum::LegacyTransactionMessage::from(t.clone()).hash()[..]);
        }
        EthereumTransaction::EIP2930(t) => {
            sig[0..32].copy_from_slice(&t.r[..]);
            sig[32..64].copy_from_slice(&t.s[..]);
            sig[64] = t.odd_y_parity as u8;
            msg.copy_from_slice(&ethereum::EIP2930TransactionMessage::from(t.clone()).hash()[..]);
        }
        EthereumTransaction::EIP1559(t) => {
            sig[0..32].copy_from_slice(&t.r[..]);
            sig[32..64].copy_from_slice(&t.s[..]);
            sig[64] = t.odd_y_parity as u8;
            msg.copy_from_slice(&ethereum::EIP1559TransactionMessage::from(t.clone()).hash()[..]);
        }
    }
    sp_io::crypto::secp256k1_ecdsa_recover(&sig, &msg)
}
