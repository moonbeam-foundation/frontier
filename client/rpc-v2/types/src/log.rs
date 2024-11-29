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

use ethereum_types::{Address, H256, U256};
use serde::{Deserialize, Serialize};

use crate::bytes::Bytes;

/// Log represents a contract log event, which is emitted by a transaction.
/// These events are generated by the LOG opcode and stored/indexed by the node.
#[derive(Clone, Debug, Eq, PartialEq, Default, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Log {
	// Consensus fields:
	/// Address of the contract that generated the event.
	pub address: Address,
	/// List of topics provided by the contract.
	pub topics: Vec<H256>,
	/// Additional data fields of the log.
	pub data: Bytes,

	// Derived fields:
	/// Hash of block in which the transaction was included.
	pub block_hash: Option<H256>,
	/// Number of the block in which the transaction was included.
	pub block_number: Option<U256>,
	/// Transaction hash.
	pub transaction_hash: Option<H256>,
	/// Index of the transaction in the block.
	pub transaction_index: Option<U256>,
	/// Index of the log in the block.
	pub log_index: Option<U256>,

	/// Whether this log was removed due to a chain reorganisation.
	#[serde(default)]
	pub removed: bool,
}
