// SPDX-License-Identifier: GPL-3.0-or-later WITH Classpath-exception-2.0
// This file is part of Frontier.
//
// Copyright (c) 2021-2022 Parity Technologies (UK) Ltd.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

use scale_codec::{Decode, Encode};
use sp_core::{H160, H256};
use sp_runtime::traits::Block as BlockT;
use std::sync::Arc;

pub mod kv;
// pub mod sql;

/// Defines the metadata structure for an ethereum transaction.
#[derive(Clone, Debug, Eq, PartialEq, Encode, Decode)]
pub struct TransactionMetadata<Block: BlockT> {
	pub block_hash: Block::Hash,
	pub ethereum_block_hash: H256,
	pub ethereum_index: u32,
}

/// Defines the structure for a filtered ethereum log.
#[derive(Debug, Eq, PartialEq)]
pub struct FilteredLog {
	pub substrate_block_hash: H256,
	pub ethereum_block_hash: H256,
	pub block_number: u32,
	pub ethereum_storage_schema: fp_storage::EthereumStorageSchema,
	pub transaction_index: u32,
	pub log_index: u32,
}

/// Defines a trait that provides access to the stored backend data.
#[async_trait::async_trait]
pub trait BackendReader<Block: BlockT>: Send + Sync {
	/// Returns the substrate block hash for a given ethereum block hash.
	fn block_hash(&self, ethereum_block_hash: &H256) -> Result<Option<Vec<Block::Hash>>, String>;

	/// Returns the ethereum transaction metadata for a provided ethereum transaction hash.
	fn transaction_metadata(
		&self,
		ethereum_transaction_hash: &H256,
	) -> Result<Vec<TransactionMetadata<Block>>, String>;

	/// Filters the logs based on the provided criteria.
	async fn filter_logs(
		&self,
		from_block: u64,
		to_block: u64,
		addresses: Vec<H160>,
		topics: Vec<Vec<Option<H256>>>,
	) -> Result<Vec<FilteredLog>, String>;

	/// Returns true if the backend is indexed.
	fn is_indexed(&self) -> bool;
}

/// Defines a backend type for frontier. The options are `KeyValue` which is backed
/// by RocksDB, or `Sql` which is backed by Sqlite.
pub enum Backend<Block: BlockT> {
	KeyValue(Arc<kv::Backend<Block>>),
	// Sql(sql::Backend<Block>),
}

#[async_trait::async_trait]
impl<Block: BlockT> crate::BackendReader<Block> for Backend<Block> {
	fn block_hash(&self, ethereum_block_hash: &H256) -> Result<Option<Vec<Block::Hash>>, String> {
		match self {
			Backend::KeyValue(b) => b.block_hash(ethereum_block_hash),
		}
	}

	fn transaction_metadata(
		&self,
		ethereum_transaction_hash: &H256,
	) -> Result<Vec<TransactionMetadata<Block>>, String> {
		match self {
			Backend::KeyValue(b) => b.transaction_metadata(ethereum_transaction_hash),
		}
	}

	async fn filter_logs(
		&self,
		from_block: u64,
		to_block: u64,
		addresses: Vec<sp_core::H160>,
		topics: Vec<Vec<Option<H256>>>,
	) -> Result<Vec<crate::FilteredLog>, String> {
		match self {
			Backend::KeyValue(b) => b.filter_logs(from_block, to_block, addresses, topics).await,
		}
	}

	fn is_indexed(&self) -> bool {
		match self {
			Backend::KeyValue(b) => b.is_indexed(),
		}
	}
}
