// SPDX-License-Identifier: Apache-2.0
// This file is part of Frontier.
//
// Copyright (c) 2020-2022 Parity Technologies (UK) Ltd.
//
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

//! Consensus extension module tests for BABE consensus.

use super::*;
use fp_xcm::{
	EthereumXcmFee, EthereumXcmTransaction, EthereumXcmTransactionV1, ManualEthereumXcmFee,
};
use frame_support::{
	assert_noop,
	weights::{Pays, PostDispatchInfo},
};
use sp_runtime::{DispatchError, DispatchErrorWithPostInfo};

// 	pragma solidity ^0.6.6;
// 	contract Test {
// 		function foo() external pure returns (bool) {
// 			return true;
// 		}
// 		function bar() external pure {
// 			require(false, "error_msg");
// 		}
// 	}
const CONTRACT: &str = "608060405234801561001057600080fd5b50610113806100206000396000f3fe6080604052348015600f57600080fd5b506004361060325760003560e01c8063c2985578146037578063febb0f7e146057575b600080fd5b603d605f565b604051808215151515815260200191505060405180910390f35b605d6068565b005b60006001905090565b600060db576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004018080602001828103825260098152602001807f6572726f725f6d7367000000000000000000000000000000000000000000000081525060200191505060405180910390fd5b56fea2646970667358221220fde68a3968e0e99b16fabf9b2997a78218b32214031f8e07e2c502daf603a69e64736f6c63430006060033";

fn xcm_evm_transfer_eip_2930_transaction(destination: H160, value: U256) -> EthereumXcmTransaction {
	let access_list = Some(vec![(H160::default(), vec![H256::default()])]);

	EthereumXcmTransaction::V1(EthereumXcmTransactionV1 {
		fee_payment: EthereumXcmFee::Manual(ManualEthereumXcmFee {
			gas_price: Some(U256::from(1)),
			max_fee_per_gas: None,
			max_priority_fee_per_gas: None,
		}),
		gas_limit: U256::from(0x100000),
		action: ethereum::TransactionAction::Call(destination),
		value,
		input: vec![],
		access_list,
	})
}

fn xcm_evm_call_eip_2930_transaction(destination: H160, input: Vec<u8>) -> EthereumXcmTransaction {
	let access_list = Some(vec![(H160::default(), vec![H256::default()])]);

	EthereumXcmTransaction::V1(EthereumXcmTransactionV1 {
		fee_payment: EthereumXcmFee::Manual(ManualEthereumXcmFee {
			gas_price: Some(U256::from(1)),
			max_fee_per_gas: None,
			max_priority_fee_per_gas: None,
		}),
		gas_limit: U256::from(0x100000),
		action: ethereum::TransactionAction::Call(destination),
		value: U256::zero(),
		input,
		access_list,
	})
}

fn xcm_erc20_creation_eip_2930_transaction() -> EthereumXcmTransaction {
	let access_list = Some(vec![(H160::default(), vec![H256::default()])]);

	EthereumXcmTransaction::V1(EthereumXcmTransactionV1 {
		fee_payment: EthereumXcmFee::Manual(ManualEthereumXcmFee {
			gas_price: Some(U256::from(1)),
			max_fee_per_gas: None,
			max_priority_fee_per_gas: None,
		}),
		gas_limit: U256::from(0x100000),
		action: ethereum::TransactionAction::Create,
		value: U256::zero(),
		input: hex::decode(ERC20_CONTRACT_BYTECODE.trim_end()).unwrap(),
		access_list,
	})
}

#[test]
fn test_transact_xcm_evm_transfer() {
	let (pairs, mut ext) = new_test_ext(2);
	let alice = &pairs[0];
	let bob = &pairs[1];

	ext.execute_with(|| {
		let balances_before = System::account(&bob.account_id);
		EthereumXcm::transact(
			RawOrigin::XcmEthereumTransaction(alice.address).into(),
			xcm_evm_transfer_eip_2930_transaction(bob.address, U256::from(100)),
		)
		.expect("Failed to execute transaction");

		assert_eq!(
			System::account(&bob.account_id).data.free,
			balances_before.data.free + 100
		);
	});
}

#[test]
fn test_transact_xcm_create() {
	let (pairs, mut ext) = new_test_ext(1);
	let alice = &pairs[0];

	ext.execute_with(|| {
		assert_noop!(
			EthereumXcm::transact(
				RawOrigin::XcmEthereumTransaction(alice.address).into(),
				xcm_erc20_creation_eip_2930_transaction()
			),
			DispatchErrorWithPostInfo {
				post_info: PostDispatchInfo {
					actual_weight: Some(0),
					pays_fee: Pays::Yes,
				},
				error: DispatchError::Other("Cannot convert xcm payload to known type"),
			}
		);
	});
}

#[test]
fn test_transact_xcm_evm_call_works() {
	let (pairs, mut ext) = new_test_ext(2);
	let alice = &pairs[0];
	let bob = &pairs[1];

	ext.execute_with(|| {
		let t = EIP2930UnsignedTransaction {
			nonce: U256::zero(),
			gas_price: U256::from(1),
			gas_limit: U256::from(0x100000),
			action: ethereum::TransactionAction::Create,
			value: U256::zero(),
			input: hex::decode(CONTRACT).unwrap(),
		}
		.sign(&alice.private_key, None);
		assert_ok!(Ethereum::execute(alice.address, &t, None,));

		let contract_address = hex::decode("32dcab0ef3fb2de2fce1d2e0799d36239671f04a").unwrap();
		let foo = hex::decode("c2985578").unwrap();
		let bar = hex::decode("febb0f7e").unwrap();

		let _ = EthereumXcm::transact(
			RawOrigin::XcmEthereumTransaction(bob.address).into(),
			xcm_evm_call_eip_2930_transaction(H160::from_slice(&contract_address), foo),
		)
		.expect("Failed to call `foo`");

		// Evm call failing still succesfully dispatched
		let _ = EthereumXcm::transact(
			RawOrigin::XcmEthereumTransaction(bob.address).into(),
			xcm_evm_call_eip_2930_transaction(H160::from_slice(&contract_address), bar),
		)
		.expect("Failed to call `bar`");

		let pending = Ethereum::pending();
		assert!(pending.len() == 2);

		// Transaction is in Pending storage, with nonce 0 and status 1 (evm succeed).
		let (transaction_0, _, receipt_0) = &pending[0];
		match (transaction_0, receipt_0) {
			(&crate::Transaction::EIP2930(ref t), &crate::Receipt::EIP2930(ref r)) => {
				assert!(t.nonce == U256::from(0u8));
				assert!(r.status_code == 1u8);
			}
			_ => unreachable!(),
		}

		// Transaction is in Pending storage, with nonce 1 and status 0 (evm failed).
		let (transaction_1, _, receipt_1) = &pending[1];
		match (transaction_1, receipt_1) {
			(&crate::Transaction::EIP2930(ref t), &crate::Receipt::EIP2930(ref r)) => {
				assert!(t.nonce == U256::from(1u8));
				assert!(r.status_code == 0u8);
			}
			_ => unreachable!(),
		}
	});
}

#[test]
fn test_transact_xcm_validation_works() {
	let (pairs, mut ext) = new_test_ext(2);
	let alice = &pairs[0];
	let bob = &pairs[1];

	ext.execute_with(|| {
		// Not enough balance fails to validate.
		assert_noop!(
			EthereumXcm::transact(
				RawOrigin::XcmEthereumTransaction(alice.address).into(),
				xcm_evm_transfer_eip_2930_transaction(bob.address, U256::MAX),
			),
			DispatchErrorWithPostInfo {
				post_info: PostDispatchInfo {
					actual_weight: Some(0),
					pays_fee: Pays::Yes,
				},
				error: DispatchError::Other("Failed to validate ethereum transaction"),
			}
		);
		// Not enough base fee fails to validate.
		assert_noop!(
			EthereumXcm::transact(
				RawOrigin::XcmEthereumTransaction(alice.address).into(),
				EthereumXcmTransaction::V1(EthereumXcmTransactionV1 {
					fee_payment: EthereumXcmFee::Manual(fp_xcm::ManualEthereumXcmFee {
						gas_price: Some(U256::from(0)),
						max_fee_per_gas: None,
						max_priority_fee_per_gas: None,
					}),
					gas_limit: U256::from(0x100000),
					action: ethereum::TransactionAction::Call(bob.address),
					value: U256::from(1),
					input: vec![],
					access_list: Some(vec![(H160::default(), vec![H256::default()])]),
				}),
			),
			DispatchErrorWithPostInfo {
				post_info: PostDispatchInfo {
					actual_weight: Some(0),
					pays_fee: Pays::Yes,
				},
				error: DispatchError::Other("Failed to validate ethereum transaction"),
			}
		);
	});
}
