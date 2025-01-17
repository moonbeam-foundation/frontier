// This file is part of Frontier.

// Copyright (c) Moonsong Labs.
// Copyright (C) Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: Apache-2.0

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Cost calculations.
//! TODO: PR EVM to make those cost calculations public.

use crate::EvmResult;
use fp_evm::{ExitError, PrecompileFailure};
use sp_core::U256;

pub fn log_costs(topics: usize, data_len: usize) -> EvmResult<u64> {
	// Cost calculation is copied from EVM code that is not publicly exposed by the crates.
	// https://github.com/rust-ethereum/evm/blob/master/src/standard/gasometer/costs.rs#L148

	const G_LOG: u64 = 375;
	const G_LOGDATA: u64 = 8;
	const G_LOGTOPIC: u64 = 375;

	let topic_cost = G_LOGTOPIC
		.checked_mul(topics as u64)
		.ok_or(PrecompileFailure::Error {
			exit_status: ExitError::OutOfGas,
		})?;

	let data_cost = G_LOGDATA
		.checked_mul(data_len as u64)
		.ok_or(PrecompileFailure::Error {
			exit_status: ExitError::OutOfGas,
		})?;

	G_LOG
		.checked_add(topic_cost)
		.ok_or(PrecompileFailure::Error {
			exit_status: ExitError::OutOfGas,
		})?
		.checked_add(data_cost)
		.ok_or(PrecompileFailure::Error {
			exit_status: ExitError::OutOfGas,
		})
}

// Compute the cost of doing a subcall.
// Some parameters cannot be known in advance, so we estimate the worst possible cost.
pub fn call_cost(value: U256, config: &evm::Config) -> u64 {
	// Copied from EVM code since not public.
	pub const G_CALLVALUE: u64 = 9000;
	pub const G_NEWACCOUNT: u64 = 25000;

	fn address_access_cost(is_cold: bool, regular_value: u64, config: &evm::Config) -> u64 {
		if config.increase_state_access_gas {
			if is_cold {
				config.gas_account_access_cold
			} else {
				config.gas_storage_read_warm
			}
		} else {
			regular_value
		}
	}

	fn xfer_cost(is_call_or_callcode: bool, transfers_value: bool) -> u64 {
		if is_call_or_callcode && transfers_value {
			G_CALLVALUE
		} else {
			0
		}
	}

	fn new_cost(
		is_call_or_staticcall: bool,
		new_account: bool,
		transfers_value: bool,
		config: &evm::Config,
	) -> u64 {
		let eip161 = !config.empty_considered_exists;
		if is_call_or_staticcall {
			if eip161 {
				if transfers_value && new_account {
					G_NEWACCOUNT
				} else {
					0
				}
			} else if new_account {
				G_NEWACCOUNT
			} else {
				0
			}
		} else {
			0
		}
	}

	let transfers_value = value != U256::default();
	let is_cold = true;
	let is_call_or_callcode = true;
	let is_call_or_staticcall = true;
	let new_account = true;

	address_access_cost(is_cold, config.gas_call, config)
		+ xfer_cost(is_call_or_callcode, transfers_value)
		+ new_cost(is_call_or_staticcall, new_account, transfers_value, config)
}
