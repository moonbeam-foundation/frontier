// This file is part of Frontier.

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

use super::*;
use crate::mock::*;

use evm::ExitReason;
use frame_support::{
	assert_ok,
	traits::{LockIdentifier, LockableCurrency, WithdrawReasons},
};
use sp_runtime::BuildStorage;
use std::{collections::BTreeMap, str::FromStr};

mod proof_size_test {
	use super::*;
	use fp_evm::{
		CreateInfo, ACCOUNT_BASIC_PROOF_SIZE, ACCOUNT_CODES_METADATA_PROOF_SIZE,
		ACCOUNT_STORAGE_PROOF_SIZE, IS_EMPTY_CHECK_PROOF_SIZE, WRITE_PROOF_SIZE,
	};
	use frame_support::traits::StorageInfoTrait;
	// pragma solidity ^0.8.2;
	// contract Callee {
	//     // ac4c25b2
	//     function void() public {
	//         uint256 foo = 1;
	//     }
	// }
	pub const PROOF_SIZE_TEST_CALLEE_CONTRACT_BYTECODE: &str =
		include_str!("./res/proof_size_test_callee_contract_bytecode.txt");
	// pragma solidity ^0.8.2;
	// contract ProofSizeTest {
	//     uint256 foo;
	//     constructor() {
	//         foo = 6;
	//     }
	//     // 35f56c3b
	//     function test_balance(address who) public {
	//         // cold
	//         uint256 a = address(who).balance;
	//         // warm
	//         uint256 b = address(who).balance;
	//     }
	//     // e27a0ecd
	//     function test_sload() public returns (uint256) {
	//         // cold
	//         uint256 a = foo;
	//         // warm
	//         uint256 b = foo;
	//         return b;
	//     }
	//     // 4f3080a9
	//     function test_sstore() public {
	//         // cold
	//         foo = 4;
	//         // warm
	//         foo = 5;
	//     }
	//     // c6d6f606
	//     function test_call(Callee _callee) public {
	//         _callee.void();
	//     }
	//     // 944ddc62
	//     function test_oog() public {
	//         uint256 i = 1;
	//         while(true) {
	//             address who = address(uint160(uint256(keccak256(abi.encodePacked(bytes32(i))))));
	//             uint256 a = address(who).balance;
	//             i = i + 1;
	//         }
	//     }
	// }
	pub const PROOF_SIZE_TEST_CONTRACT_BYTECODE: &str =
		include_str!("./res/proof_size_test_contract_bytecode.txt");

	fn create_proof_size_test_callee_contract(
		gas_limit: u64,
		weight_limit: Option<Weight>,
	) -> Result<CreateInfo, crate::RunnerError<crate::Error<Test>>> {
		<Test as Config>::Runner::create(
			H160::default(),
			hex::decode(PROOF_SIZE_TEST_CALLEE_CONTRACT_BYTECODE.trim_end()).unwrap(),
			U256::zero(),
			gas_limit,
			Some(FixedGasPrice::min_gas_price().0),
			None,
			None,
			Vec::new(),
			Vec::new(),
			true, // transactional
			true, // must be validated
			weight_limit,
			Some(0),
			&<Test as Config>::config().clone(),
		)
	}

	fn create_proof_size_test_contract(
		gas_limit: u64,
		weight_limit: Option<Weight>,
	) -> Result<CreateInfo, crate::RunnerError<crate::Error<Test>>> {
		<Test as Config>::Runner::create(
			H160::default(),
			hex::decode(PROOF_SIZE_TEST_CONTRACT_BYTECODE.trim_end()).unwrap(),
			U256::zero(),
			gas_limit,
			Some(FixedGasPrice::min_gas_price().0),
			None,
			None,
			Vec::new(),
			Vec::new(),
			true, // non-transactional
			true, // must be validated
			weight_limit,
			Some(0),
			&<Test as Config>::config().clone(),
		)
	}

	#[test]
	fn account_basic_proof_size_constant_matches() {
		assert_eq!(
			ACCOUNT_BASIC_PROOF_SIZE,
			frame_system::Account::<Test>::storage_info()
				.first()
				.expect("item")
				.max_size
				.expect("size") as u64
		);
	}

	#[test]
	fn account_storage_proof_size_constant_matches() {
		assert_eq!(
			ACCOUNT_STORAGE_PROOF_SIZE,
			AccountStorages::<Test>::storage_info()
				.first()
				.expect("item")
				.max_size
				.expect("size") as u64
		);
	}

	#[test]
	fn account_codes_metadata_proof_size_constant_matches() {
		assert_eq!(
			ACCOUNT_CODES_METADATA_PROOF_SIZE,
			AccountCodesMetadata::<Test>::storage_info()
				.first()
				.expect("item")
				.max_size
				.expect("size") as u64
		);
	}

	#[test]
	fn proof_size_create_accounting_works() {
		new_test_ext().execute_with(|| {
			let gas_limit: u64 = 1_000_000;
			let weight_limit = FixedGasWeightMapping::<Test>::gas_to_weight(gas_limit, true);

			let result = create_proof_size_test_callee_contract(gas_limit, Some(weight_limit))
				.expect("create succeeds");

			// Creating a new contract does not involve reading the code from storage.
			// We account for a fixed hash proof size write, an empty check and .
			let write_cost = WRITE_PROOF_SIZE;
			let is_empty_check = IS_EMPTY_CHECK_PROOF_SIZE;
			let nonce_increases = ACCOUNT_BASIC_PROOF_SIZE * 2;
			let expected_proof_size = write_cost + is_empty_check + nonce_increases;

			let actual_proof_size = result
				.weight_info
				.expect("weight info")
				.proof_size_usage
				.expect("proof size usage");

			assert_eq!(expected_proof_size, actual_proof_size);
		});
	}

	#[test]
	fn proof_size_subcall_accounting_works() {
		new_test_ext().execute_with(|| {
			// Create callee contract A
			let gas_limit: u64 = 1_000_000;
			let weight_limit = FixedGasWeightMapping::<Test>::gas_to_weight(gas_limit, true);
			let result =
				create_proof_size_test_callee_contract(gas_limit, None).expect("create succeeds");

			let subcall_contract_address = result.value;

			// Create proof size test contract B
			let result = create_proof_size_test_contract(gas_limit, None).expect("create succeeds");

			let call_contract_address = result.value;

			// Call B, that calls A, with weight limit
			// selector for ProofSizeTest::test_call function..
			let mut call_data: String = "c6d6f606000000000000000000000000".to_owned();
			// ..encode the callee address argument
			call_data.push_str(&format!("{subcall_contract_address:x}"));

			let result = <Test as Config>::Runner::call(
				H160::default(),
				call_contract_address,
				hex::decode(&call_data).unwrap(),
				U256::zero(),
				gas_limit,
				Some(FixedGasPrice::min_gas_price().0),
				None,
				None,
				Vec::new(),
				Vec::new(),
				true, // transactional
				true, // must be validated
				Some(weight_limit),
				Some(0),
				&<Test as Config>::config().clone(),
			)
			.expect("call succeeds");

			// Expected proof size
			let reading_main_contract_len = AccountCodes::<Test>::get(call_contract_address).len();
			let reading_contract_len = AccountCodes::<Test>::get(subcall_contract_address).len();
			let read_account_metadata = ACCOUNT_CODES_METADATA_PROOF_SIZE as usize;
			let is_empty_check = (IS_EMPTY_CHECK_PROOF_SIZE * 2) as usize;
			let increase_nonce = (ACCOUNT_BASIC_PROOF_SIZE * 3) as usize;
			let expected_proof_size = ((read_account_metadata * 2)
				+ reading_contract_len
				+ reading_main_contract_len
				+ is_empty_check
				+ increase_nonce) as u64;

			let actual_proof_size = result
				.weight_info
				.expect("weight info")
				.proof_size_usage
				.expect("proof size usage");

			assert_eq!(expected_proof_size, actual_proof_size);
		});
	}

	#[test]
	fn proof_size_balance_accounting_works() {
		new_test_ext().execute_with(|| {
			let gas_limit: u64 = 1_000_000;
			let weight_limit = FixedGasWeightMapping::<Test>::gas_to_weight(gas_limit, true);

			// Create proof size test contract
			let result = create_proof_size_test_contract(gas_limit, None).expect("create succeeds");

			let call_contract_address = result.value;

			// selector for ProofSizeTest::balance function..
			let mut call_data: String = "35f56c3b000000000000000000000000".to_owned();
			// ..encode bobs address
			call_data.push_str(&format!("{:x}", H160::random()));

			let result = <Test as Config>::Runner::call(
				H160::default(),
				call_contract_address,
				hex::decode(&call_data).unwrap(),
				U256::zero(),
				gas_limit,
				Some(FixedGasPrice::min_gas_price().0),
				None,
				None,
				Vec::new(),
				Vec::new(),
				true, // transactional
				true, // must be validated
				Some(weight_limit),
				Some(0),
				&<Test as Config>::config().clone(),
			)
			.expect("call succeeds");

			// - Three account reads.
			// - Main contract code read.
			// - One metadata read.
			let basic_account_size = (ACCOUNT_BASIC_PROOF_SIZE * 3) as usize;
			let read_account_metadata = ACCOUNT_CODES_METADATA_PROOF_SIZE as usize;
			let is_empty_check = IS_EMPTY_CHECK_PROOF_SIZE as usize;
			let increase_nonce = ACCOUNT_BASIC_PROOF_SIZE as usize;
			let reading_main_contract_len = AccountCodes::<Test>::get(call_contract_address).len();
			let expected_proof_size = (basic_account_size
				+ read_account_metadata
				+ reading_main_contract_len
				+ is_empty_check
				+ increase_nonce) as u64;

			let actual_proof_size = result
				.weight_info
				.expect("weight info")
				.proof_size_usage
				.expect("proof size usage");

			assert_eq!(expected_proof_size, actual_proof_size);
		});
	}

	#[test]
	fn proof_size_sload_accounting_works() {
		new_test_ext().execute_with(|| {
			let gas_limit: u64 = 1_000_000;
			let weight_limit = FixedGasWeightMapping::<Test>::gas_to_weight(gas_limit, true);

			// Create proof size test contract
			let result = create_proof_size_test_contract(gas_limit, None).expect("create succeeds");

			let call_contract_address = result.value;

			// selector for ProofSizeTest::test_sload function..
			let call_data: String = "e27a0ecd".to_owned();
			let result = <Test as Config>::Runner::call(
				H160::default(),
				call_contract_address,
				hex::decode(call_data).unwrap(),
				U256::zero(),
				gas_limit,
				Some(FixedGasPrice::min_gas_price().0),
				None,
				None,
				Vec::new(),
				Vec::new(),
				true, // transactional
				true, // must be validated
				Some(weight_limit),
				Some(0),
				&<Test as Config>::config().clone(),
			)
			.expect("call succeeds");

			let reading_main_contract_len =
				AccountCodes::<Test>::get(call_contract_address).len() as u64;
			let expected_proof_size = reading_main_contract_len
				+ ACCOUNT_STORAGE_PROOF_SIZE
				+ ACCOUNT_CODES_METADATA_PROOF_SIZE
				+ IS_EMPTY_CHECK_PROOF_SIZE
				+ (ACCOUNT_BASIC_PROOF_SIZE * 2);

			let actual_proof_size = result
				.weight_info
				.expect("weight info")
				.proof_size_usage
				.expect("proof size usage");

			assert_eq!(expected_proof_size, actual_proof_size);
		});
	}

	#[test]
	fn proof_size_sstore_accounting_works() {
		new_test_ext().execute_with(|| {
			let gas_limit: u64 = 1_000_000;
			let weight_limit = FixedGasWeightMapping::<Test>::gas_to_weight(gas_limit, true);

			// Create proof size test contract
			let result = create_proof_size_test_contract(gas_limit, None).expect("create succeeds");

			let call_contract_address = result.value;

			// selector for ProofSizeTest::test_sstore function..
			let call_data: String = "4f3080a9".to_owned();
			let result = <Test as Config>::Runner::call(
				H160::default(),
				call_contract_address,
				hex::decode(call_data).unwrap(),
				U256::zero(),
				gas_limit,
				Some(FixedGasPrice::min_gas_price().0),
				None,
				None,
				Vec::new(),
				Vec::new(),
				true, // transactional
				true, // must be validated
				Some(weight_limit),
				Some(0),
				&<Test as Config>::config().clone(),
			)
			.expect("call succeeds");

			let reading_main_contract_len =
				AccountCodes::<Test>::get(call_contract_address).len() as u64;
			let expected_proof_size = reading_main_contract_len
				+ WRITE_PROOF_SIZE
				+ ACCOUNT_CODES_METADATA_PROOF_SIZE
				+ ACCOUNT_STORAGE_PROOF_SIZE
				+ IS_EMPTY_CHECK_PROOF_SIZE
				+ (ACCOUNT_BASIC_PROOF_SIZE * 2);

			let actual_proof_size = result
				.weight_info
				.expect("weight info")
				.proof_size_usage
				.expect("proof size usage");

			assert_eq!(expected_proof_size, actual_proof_size);
		});
	}

	#[test]
	fn proof_size_oog_works() {
		new_test_ext().execute_with(|| {
			let gas_limit: u64 = 1_000_000;
			let mut weight_limit = FixedGasWeightMapping::<Test>::gas_to_weight(gas_limit, true);

			// Artifically set a lower proof size limit so we OOG this instead gas.
			let proof_size_limit = weight_limit.proof_size() / 2;
			*weight_limit.proof_size_mut() = proof_size_limit;

			// Create proof size test contract
			let result = create_proof_size_test_contract(gas_limit, None).expect("create succeeds");

			let call_contract_address = result.value;

			// selector for ProofSizeTest::test_oog function..
			let call_data: String = "944ddc62".to_owned();
			let result = <Test as Config>::Runner::call(
				H160::default(),
				call_contract_address,
				hex::decode(call_data).unwrap(),
				U256::zero(),
				gas_limit,
				Some(FixedGasPrice::min_gas_price().0),
				None,
				None,
				Vec::new(),
				Vec::new(),
				true, // transactional
				true, // must be validated
				Some(weight_limit),
				Some(0),
				&<Test as Config>::config().clone(),
			)
			.expect("call succeeds");

			let actual_proof_size = result
				.weight_info
				.expect("weight info")
				.proof_size_usage
				.expect("proof size usage");

			assert_eq!(proof_size_limit, actual_proof_size);
		});
	}

	#[test]
	fn uncached_account_code_proof_size_accounting_works() {
		new_test_ext().execute_with(|| {
			// Create callee contract A
			let gas_limit: u64 = 1_000_000;
			let weight_limit = FixedGasWeightMapping::<Test>::gas_to_weight(gas_limit, true);
			let result =
				create_proof_size_test_callee_contract(gas_limit, None).expect("create succeeds");

			let subcall_contract_address = result.value;

			// Expect callee contract code hash and size to be cached
			let _ = <AccountCodesMetadata<Test>>::get(subcall_contract_address)
				.expect("contract code hash and size are cached");

			// Remove callee cache
			<AccountCodesMetadata<Test>>::remove(subcall_contract_address);

			// Create proof size test contract B
			let result = create_proof_size_test_contract(gas_limit, None).expect("create succeeds");

			let call_contract_address = result.value;

			// Call B, that calls A, with weight limit
			// selector for ProofSizeTest::test_call function..
			let mut call_data: String = "c6d6f606000000000000000000000000".to_owned();
			// ..encode the callee address argument
			call_data.push_str(&format!("{subcall_contract_address:x}"));
			let result = <Test as Config>::Runner::call(
				H160::default(),
				call_contract_address,
				hex::decode(&call_data).unwrap(),
				U256::zero(),
				gas_limit,
				Some(FixedGasPrice::min_gas_price().0),
				None,
				None,
				Vec::new(),
				Vec::new(),
				true, // transactional
				true, // must be validated
				Some(weight_limit),
				Some(0),
				&<Test as Config>::config().clone(),
			)
			.expect("call succeeds");

			// Expected proof size
			let read_account_metadata = ACCOUNT_CODES_METADATA_PROOF_SIZE as usize;
			let is_empty_check = (IS_EMPTY_CHECK_PROOF_SIZE * 2) as usize;
			let increase_nonce = (ACCOUNT_BASIC_PROOF_SIZE * 3) as usize;
			let reading_main_contract_len = AccountCodes::<Test>::get(call_contract_address).len();
			let reading_callee_contract_len =
				AccountCodes::<Test>::get(subcall_contract_address).len();
			// In order to do the subcall, we need to check metadata 3 times -
			// one for each contract + one for the call opcode -, load two bytecodes - caller and callee.
			let expected_proof_size = ((read_account_metadata * 2)
				+ reading_callee_contract_len
				+ reading_main_contract_len
				+ is_empty_check
				+ increase_nonce) as u64;

			let actual_proof_size = result
				.weight_info
				.expect("weight info")
				.proof_size_usage
				.expect("proof size usage");

			assert_eq!(expected_proof_size, actual_proof_size);
		});
	}

	#[test]
	fn proof_size_breaks_standard_transfer() {
		new_test_ext().execute_with(|| {
			// In this test we do a simple transfer to an address with an stored code which is
			// greater in size (and thus load cost) than the transfer flat fee of 21_000.

			// We assert that providing 21_000 gas limit will not work, because the pov size limit
			// will OOG.
			let fake_contract_address = H160::random();
			let config = <Test as Config>::config().clone();
			let fake_contract_code = vec![0; config.create_contract_limit.expect("a value")];
			AccountCodes::<Test>::insert(fake_contract_address, fake_contract_code);

			let gas_limit: u64 = 21_000;
			let weight_limit = FixedGasWeightMapping::<Test>::gas_to_weight(gas_limit, true);

			let result = <Test as Config>::Runner::call(
				H160::default(),
				fake_contract_address,
				Vec::new(),
				U256::from(777),
				gas_limit,
				Some(FixedGasPrice::min_gas_price().0),
				None,
				None,
				Vec::new(),
				Vec::new(),
				true, // transactional
				true, // must be validated
				Some(weight_limit),
				Some(0),
				&config,
			)
			.expect("call succeeds");

			assert_eq!(
				result.exit_reason,
				crate::ExitReason::Error(crate::ExitError::OutOfGas)
			);
		});
	}

	#[test]
	fn proof_size_based_refunding_works() {
		new_test_ext().execute_with(|| {
			// In this test we do a simple transfer to an address with an stored code which is
			// greater in size (and thus load cost) than the transfer flat fee of 21_000.

			// Assert that if we provide enough gas limit, the refund will be based on the pov
			// size consumption, not the 21_000 gas.
			let fake_contract_address = H160::random();
			let config = <Test as Config>::config().clone();
			let fake_contract_code = vec![0; config.create_contract_limit.expect("a value")];
			AccountCodes::<Test>::insert(fake_contract_address, fake_contract_code);

			let gas_limit: u64 = 700_000;
			let weight_limit = FixedGasWeightMapping::<Test>::gas_to_weight(gas_limit, true);

			let result = <Test as Config>::Runner::call(
				H160::default(),
				fake_contract_address,
				Vec::new(),
				U256::from(777),
				gas_limit,
				Some(FixedGasPrice::min_gas_price().0),
				None,
				None,
				Vec::new(),
				Vec::new(),
				true, // transactional
				true, // must be validated
				Some(weight_limit),
				Some(0),
				&config,
			)
			.expect("call succeeds");

			let ratio = <<Test as Config>::GasLimitPovSizeRatio as Get<u64>>::get();
			let used_gas = result.used_gas;
			let actual_proof_size = result
				.weight_info
				.expect("weight info")
				.proof_size_usage
				.expect("proof size usage");

			assert_eq!(used_gas.standard, U256::from(21_000));
			assert_eq!(used_gas.effective, U256::from(actual_proof_size * ratio));
		});
	}
}

mod storage_growth_test {
	use super::*;
	use crate::tests::proof_size_test::PROOF_SIZE_TEST_CALLEE_CONTRACT_BYTECODE;
	use fp_evm::{
		ACCOUNT_CODES_KEY_SIZE, ACCOUNT_CODES_METADATA_PROOF_SIZE, ACCOUNT_STORAGE_PROOF_SIZE,
	};

	const PROOF_SIZE_CALLEE_CONTRACT_BYTECODE_LEN: u64 = 116;
	// The contract bytecode stored on chain.
	const STORAGE_GROWTH_TEST_CONTRACT: &str =
		include_str!("./res/storage_growth_test_contract_bytecode.txt");
	const STORAGE_GROWTH_TEST_CONTRACT_BYTECODE_LEN: u64 = 455;

	fn create_test_contract(
		contract: &str,
		gas_limit: u64,
	) -> Result<CreateInfo, crate::RunnerError<crate::Error<Test>>> {
		<Test as Config>::Runner::create(
			H160::default(),
			hex::decode(contract.trim_end()).expect("Failed to decode contract"),
			U256::zero(),
			gas_limit,
			Some(FixedGasPrice::min_gas_price().0),
			None,
			None,
			Vec::new(),
			Vec::new(),
			true, // transactional
			true, // must be validated
			Some(FixedGasWeightMapping::<Test>::gas_to_weight(
				gas_limit, true,
			)),
			Some(0),
			<Test as Config>::config(),
		)
	}

	// Calls the given contract
	fn call_test_contract(
		contract_addr: H160,
		call_data: &[u8],
		value: U256,
		gas_limit: u64,
	) -> Result<CallInfo, crate::RunnerError<crate::Error<Test>>> {
		<Test as Config>::Runner::call(
			H160::default(),
			contract_addr,
			call_data.to_vec(),
			value,
			gas_limit,
			Some(FixedGasPrice::min_gas_price().0),
			None,
			None,
			Vec::new(),
			Vec::new(),
			true, // transactional
			true, // must be validated
			None,
			Some(0),
			<Test as Config>::config(),
		)
	}

	// Computes the expected gas for contract creation (related to storage growth).
	// `byte_code_len` represents the length of the contract bytecode stored on-chain.
	fn expected_contract_create_storage_growth_gas(bytecode_len: u64) -> u64 {
		let ratio = <<Test as Config>::GasLimitStorageGrowthRatio as Get<u64>>::get();
		(ACCOUNT_CODES_KEY_SIZE + ACCOUNT_CODES_METADATA_PROOF_SIZE + bytecode_len) * ratio
	}

	/// Test that contract deployment succeeds when the necessary storage growth gas is provided.
	#[test]
	fn contract_deployment_should_succeed() {
		new_test_ext().execute_with(|| {
			let gas_limit: u64 = 85_000;

			let result = create_test_contract(PROOF_SIZE_TEST_CALLEE_CONTRACT_BYTECODE, gas_limit)
				.expect("create succeeds");

			assert_eq!(
				result.used_gas.effective.as_u64(),
				expected_contract_create_storage_growth_gas(
					PROOF_SIZE_CALLEE_CONTRACT_BYTECODE_LEN
				)
			);
			assert_eq!(
				result.exit_reason,
				crate::ExitReason::Succeed(ExitSucceed::Returned)
			);
			// Assert that the contract entry exists in the storage.
			assert!(AccountCodes::<Test>::contains_key(result.value));
		});
	}

	// Test that contract creation with code initialization that results in new storage entries
	// succeeds when the necessary storage growth gas is provided.
	#[test]
	fn contract_creation_with_code_initialization_should_succeed() {
		new_test_ext().execute_with(|| {
			let gas_limit: u64 = 863_394;
			let ratio = <<Test as Config>::GasLimitStorageGrowthRatio as Get<u64>>::get();
			// The constructor of the contract creates 3 new storage entries (uint256). So,
			// the expected gas is the gas for contract creation + 3 * ACCOUNT_STORAGE_PROOF_SIZE.
			let expected_storage_growth_gas = expected_contract_create_storage_growth_gas(
				STORAGE_GROWTH_TEST_CONTRACT_BYTECODE_LEN,
			) + (3 * ACCOUNT_STORAGE_PROOF_SIZE * ratio);

			// Deploy the contract.
			let result = create_test_contract(STORAGE_GROWTH_TEST_CONTRACT, gas_limit)
				.expect("create succeeds");

			assert_eq!(
				result.used_gas.effective.as_u64(),
				expected_storage_growth_gas
			);
			assert_eq!(
				result.exit_reason,
				crate::ExitReason::Succeed(ExitSucceed::Returned)
			);
		});
	}

	// Verify that saving new entries fails when insufficient storage growth gas is supplied.
	#[test]
	fn store_new_entries_should_fail_oog() {
		new_test_ext().execute_with(|| {
			let gas_limit: u64 = 863_394;
			// Deploy the contract.
			let res = create_test_contract(STORAGE_GROWTH_TEST_CONTRACT, gas_limit)
				.expect("create succeeds");
			let contract_addr = res.value;

			let gas_limit = 120_000;
			// Call the contract method store to store new entries.
			let result = call_test_contract(
				contract_addr,
				&hex::decode("975057e7").unwrap(),
				U256::zero(),
				gas_limit,
			)
			.expect("call should succeed");

			assert_eq!(
				result.exit_reason,
				crate::ExitReason::Error(crate::ExitError::OutOfGas)
			);
		});
	}

	// Verify that saving new entries succeeds when sufficient storage growth gas is supplied.
	#[test]
	fn store_new_entries_should_succeeds() {
		new_test_ext().execute_with(|| {
			let gas_limit: u64 = 863_394;
			// Deploy the contract.
			let res = create_test_contract(STORAGE_GROWTH_TEST_CONTRACT, gas_limit)
				.expect("create succeeds");
			let contract_addr = res.value;

			let gas_limit = 128_000;
			// Call the contract method store to store new entries.
			let result = call_test_contract(
				contract_addr,
				&hex::decode("975057e7").unwrap(),
				U256::zero(),
				gas_limit,
			)
			.expect("call should succeed");

			let expected_storage_growth_gas = 3
				* ACCOUNT_STORAGE_PROOF_SIZE
				* <<Test as Config>::GasLimitStorageGrowthRatio as Get<u64>>::get();
			assert_eq!(
				result.exit_reason,
				crate::ExitReason::Succeed(ExitSucceed::Stopped)
			);
			assert_eq!(
				result.used_gas.effective.as_u64(),
				expected_storage_growth_gas
			);
		});
	}

	// Verify that updating existing storage entries does not incur any storage growth charges.
	#[test]
	fn update_exisiting_entries_succeeds() {
		new_test_ext().execute_with(|| {
			let gas_limit: u64 = 863_394;
			// Deploy the contract.
			let res = create_test_contract(STORAGE_GROWTH_TEST_CONTRACT, gas_limit)
				.expect("create succeeds");
			let contract_addr = res.value;

			// Providing gas limit of 37_000 is enough to update existing entries, but not enough
			// to store new entries.
			let gas_limit = 37_000;
			// Call the contract method update to update existing entries.
			let result = call_test_contract(
				contract_addr,
				&hex::decode("a2e62045").unwrap(),
				U256::zero(),
				gas_limit,
			)
			.expect("call should succeed");

			assert_eq!(
				result.exit_reason,
				crate::ExitReason::Succeed(ExitSucceed::Stopped)
			);
		});
	}
}

mod eip7702_delegation_storage_meter_tests {
	use super::*;
	use crate::{Config, ExitReason, FixedGasWeightMapping};
	use ethereum::AuthorizationListItem;
	use evm::delegation::{EIP_7702_DELEGATION_PREFIX, EIP_7702_DELEGATION_SIZE};
	use fp_evm::{ACCOUNT_CODES_KEY_SIZE, ACCOUNT_CODES_METADATA_PROOF_SIZE, WRITE_PROOF_SIZE};
	use frame_support::traits::Get;
	use libsecp256k1::{Message, SecretKey};
	use rlp::RlpStream;
	use sp_runtime::traits::UniqueSaturatedInto;

	fn sign_authorization(
		chain_id: u64,
		delegate: H160,
		nonce: u64,
		sk: &SecretKey,
	) -> AuthorizationListItem {
		let magic: u8 = 0x05;
		let mut stream = RlpStream::new_list(3);
		stream.append(&chain_id);
		stream.append(&delegate);
		stream.append(&nonce);
		let mut msg_data = vec![magic];
		msg_data.extend_from_slice(&stream.out());
		let msg_hash = sp_io::hashing::keccak_256(&msg_data);
		let msg = Message::parse_slice(&msg_hash).expect("digest length");
		let (sig, rec_id) = libsecp256k1::sign(&msg, sk);
		let rs = sig.serialize();
		let r = H256::from_slice(&rs[0..32]);
		let s = H256::from_slice(&rs[32..64]);

		AuthorizationListItem {
			chain_id,
			address: delegate,
			nonce: U256::from(nonce),
			signature: ethereum::eip2930::MalleableTransactionSignature {
				odd_y_parity: rec_id.serialize() != 0,
				r,
				s,
			},
		}
	}

	fn gas_too_low_for_single_delegation_storage_budget() -> u64 {
		let ratio = <<Test as Config>::GasLimitStorageGrowthRatio as Get<u64>>::get();
		let designator_len = EIP_7702_DELEGATION_SIZE as u64;
		let bytes_per_delegation = ACCOUNT_CODES_KEY_SIZE
			.saturating_add(ACCOUNT_CODES_METADATA_PROOF_SIZE)
			.saturating_add(designator_len);
		ratio.saturating_mul(bytes_per_delegation).saturating_sub(1)
	}

	/// Largest `gas_limit` such that `gas_limit / GasLimitStorageGrowthRatio` is **strictly**
	/// below two EIP-7702 delegation writes (each ~135 bytes of metered storage growth).
	///
	/// With ratio 366: storage byte budget = floor(gas / 366) = 269 < 2 × 135, but still ≥ 135
	/// so the first delegation can be recorded before the second trips the meter.
	fn gas_limit_two_delegations_exceeds_storage_budget() -> u64 {
		let ratio = <<Test as Config>::GasLimitStorageGrowthRatio as Get<u64>>::get();
		let designator_len = EIP_7702_DELEGATION_SIZE as u64;
		let bytes_per_delegation = ACCOUNT_CODES_KEY_SIZE
			.saturating_add(ACCOUNT_CODES_METADATA_PROOF_SIZE)
			.saturating_add(designator_len);
		let two = 2u64.saturating_mul(bytes_per_delegation);
		// floor(gas_limit / ratio) == two - 1  →  second delegation pushes usage past limit
		ratio.saturating_mul(two).saturating_sub(1)
	}

	#[test]
	fn delegation_triggers_storage_meter_oog_when_gas_limit_storage_budget_too_small() {
		new_test_ext().execute_with(|| {
			let sk = SecretKey::parse_slice(&hex_literal::hex!(
				"0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"
			))
			.expect("valid test secret key");

			let chain_id = <Test as Config>::ChainId::get();
			let delegate = H160(hex_literal::hex!(
				"4242424242424242424242424242424242424242"
			));
			let auth = sign_authorization(chain_id, delegate, 0, &sk);

			let gas_limit = gas_too_low_for_single_delegation_storage_budget();

			let info = <Test as Config>::Runner::call(
				H160::default(),
				H160::repeat_byte(0x77),
				vec![],
				U256::zero(),
				gas_limit,
				Some(FixedGasPrice::min_gas_price().0),
				None,
				None,
				vec![],
				vec![auth],
				true,
				false,
				None,
				Some(0),
				<Test as Config>::config(),
			)
			.expect("runner returns CallInfo");

			assert_eq!(
				info.exit_reason,
				ExitReason::Error(crate::ExitError::OutOfGas),
				"gas_limit {gas_limit} leaves storage budget below one EIP-7702 delegation write",
			);
		});
	}

	/// Regression for EIP-7702: two distinct authorities delegating in one tx must respect the
	/// per-transaction storage byte budget derived from `gas_limit` (not bypass `StorageMeter`).
	///
	/// Before the fix, `set_delegation` did not call `record_external_operation(Write)`, so two
	/// delegations could succeed even when `gas_limit / ratio` only allowed one write’s worth of
	/// growth. After the fix, the second delegation must fail with `OutOfGas`.
	#[test]
	fn eip7702_delegation_bypasses_storage_meter_safety_check() {
		new_test_ext().execute_with(|| {
			let sk_a = SecretKey::parse_slice(&hex_literal::hex!(
				"0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"
			))
			.expect("valid test secret key");
			let sk_b = SecretKey::parse_slice(&hex_literal::hex!(
				"11223344556677889900aabbccddeeff00112233445566778899aabbccddeeff"
			))
			.expect("valid test secret key");

			let chain_id = <Test as Config>::ChainId::get();
			let delegate =
				H160(hex_literal::hex!("1000000000000000000000000000000000000001"));

			let auth_a = sign_authorization(chain_id, delegate, 0, &sk_a);
			let auth_b = sign_authorization(chain_id, delegate, 0, &sk_b);

			let gas_limit = gas_limit_two_delegations_exceeds_storage_budget();
			let ratio = <<Test as Config>::GasLimitStorageGrowthRatio as Get<u64>>::get();
			let storage_budget = gas_limit.saturating_div(ratio);
			let bytes_per = ACCOUNT_CODES_KEY_SIZE
				.saturating_add(ACCOUNT_CODES_METADATA_PROOF_SIZE)
				.saturating_add(EIP_7702_DELEGATION_SIZE as u64);
			assert!(
				storage_budget >= bytes_per,
				"test setup: first delegation must fit (budget={storage_budget}, need {bytes_per})"
			);
			assert!(
				storage_budget < 2.saturating_mul(bytes_per),
				"test setup: two delegations must exceed budget (budget={storage_budget}, two={})",
				2.saturating_mul(bytes_per)
			);

			let info = <Test as Config>::Runner::call(
				H160::default(),
				H160::repeat_byte(0x88),
				vec![],
				U256::zero(),
				gas_limit,
				Some(FixedGasPrice::min_gas_price().0),
				None,
				None,
				vec![],
				vec![auth_a, auth_b],
				true,
				false,
				None,
				Some(0),
				<Test as Config>::config(),
			)
			.expect("runner returns CallInfo");

			assert_eq!(
				info.exit_reason,
				ExitReason::Error(crate::ExitError::OutOfGas),
				"second EIP-7702 delegation must trip StorageMeter (gas_limit={gas_limit}, storage_budget_bytes={storage_budget})",
			);
		});
	}

	#[test]
	fn delegation_records_storage_and_pov_when_gas_budget_sufficient() {
		new_test_ext().execute_with(|| {
			let sk = SecretKey::parse_slice(&hex_literal::hex!(
				"0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"
			))
			.expect("valid test secret key");

			let chain_id = <Test as Config>::ChainId::get();
			let delegate =
				H160(hex_literal::hex!("4242424242424242424242424242424242424242"));
			let auth = sign_authorization(chain_id, delegate, 0, &sk);

			let authority_h160 =
				H160(auth.authorizing_address().expect("recover signer").0);

			let gas_limit = 2_000_000u64;
			let weight_limit = FixedGasWeightMapping::<Test>::gas_to_weight(gas_limit, true);

			let info = <Test as Config>::Runner::call(
				H160::default(),
				H160::repeat_byte(0x77),
				vec![],
				U256::zero(),
				gas_limit,
				Some(FixedGasPrice::min_gas_price().0),
				None,
				None,
				vec![],
				vec![auth],
				true,
				false,
				Some(weight_limit),
				Some(0),
				<Test as Config>::config(),
			)
			.expect("runner returns CallInfo");

			assert!(
				info.exit_reason.is_succeed(),
				"unexpected exit: {:?}",
				info.exit_reason
			);

			let code = AccountCodes::<Test>::get(authority_h160);
			assert!(
				code.starts_with(EIP_7702_DELEGATION_PREFIX),
				"delegation designator should be stored"
			);
			assert_eq!(code.len(), EIP_7702_DELEGATION_SIZE);

			let ratio = <<Test as Config>::GasLimitStorageGrowthRatio as Get<u64>>::get();
			let delegation_storage_gas = (ACCOUNT_CODES_KEY_SIZE
				+ ACCOUNT_CODES_METADATA_PROOF_SIZE
				+ EIP_7702_DELEGATION_SIZE as u64)
				.saturating_mul(ratio);
			let effective: u64 =
				UniqueSaturatedInto::<u64>::unique_saturated_into(info.used_gas.effective);
			assert!(
				effective >= delegation_storage_gas,
				"effective gas is max(execution, pov, storage); it must be at least the delegation storage component (effective={effective}, delegation_storage_gas={delegation_storage_gas})",
			);

			let proof = info
				.weight_info
				.expect("weight limit was set")
				.proof_size_usage
				.expect("proof accounting");
			assert!(
				proof >= WRITE_PROOF_SIZE,
				"delegation write should contribute PoV ({proof} < {WRITE_PROOF_SIZE})",
			);
		});
	}
}

type Balances = pallet_balances::Pallet<Test>;
#[allow(clippy::upper_case_acronyms)]
type EVM = Pallet<Test>;

pub fn new_test_ext() -> sp_io::TestExternalities {
	let mut t = frame_system::GenesisConfig::<Test>::default()
		.build_storage()
		.unwrap();

	let mut accounts = BTreeMap::new();
	accounts.insert(
		H160::from_str("1000000000000000000000000000000000000001").unwrap(),
		GenesisAccount {
			nonce: U256::from(1),
			balance: U256::from(1000000),
			storage: Default::default(),
			code: vec![
				0x00, // STOP
			],
		},
	);
	accounts.insert(
		H160::from_str("1000000000000000000000000000000000000002").unwrap(),
		GenesisAccount {
			nonce: U256::from(1),
			balance: U256::from(1000000),
			storage: Default::default(),
			code: vec![
				0xff, // INVALID
			],
		},
	);
	accounts.insert(
		H160::default(), // root
		GenesisAccount {
			nonce: U256::from(1),
			balance: U256::max_value(),
			storage: Default::default(),
			code: vec![],
		},
	);
	accounts.insert(
		H160::from([4u8; 20]), // alith
		GenesisAccount {
			nonce: U256::from(1),
			balance: U256::max_value(),
			storage: Default::default(),
			code: vec![],
		},
	);
	accounts.insert(
		H160::from([5u8; 20]), // bob
		GenesisAccount {
			nonce: U256::from(1),
			balance: U256::max_value(),
			storage: Default::default(),
			code: vec![],
		},
	);
	accounts.insert(
		H160::from([6u8; 20]), // charleth
		GenesisAccount {
			nonce: U256::from(1),
			balance: U256::max_value(),
			storage: Default::default(),
			code: vec![],
		},
	);

	// Create the block author account with some balance.
	let author = H160::from_str("0x1234500000000000000000000000000000000000").unwrap();
	pallet_balances::GenesisConfig::<Test> {
		balances: vec![(
			<Test as Config>::AddressMapping::into_account_id(author),
			12345,
		)],
		dev_accounts: None,
	}
	.assimilate_storage(&mut t)
	.expect("Pallet balances storage can be assimilated");

	crate::GenesisConfig::<Test> {
		accounts,
		..Default::default()
	}
	.assimilate_storage(&mut t)
	.unwrap();

	t.into()
}

// pragma solidity ^0.8.2;

// contract Foo {

//  function newBar() // 2fc11060
//    public
//    returns(Bar newContract)
//  {
//    Bar b = new Bar();
//    return b;
//  }
//}

// contract Bar {
//  function getNumber()
//    public
//    pure
//    returns (uint32 number)
//  {
//    return 10;
//  }
//}
pub const FOO_BAR_CONTRACT_CREATOR_BYTECODE: &str =
	include_str!("./res/foo_bar_contract_creator.txt");

fn create_foo_bar_contract_creator(
	gas_limit: u64,
	weight_limit: Option<Weight>,
) -> Result<CreateInfo, crate::RunnerError<crate::Error<Test>>> {
	<Test as Config>::Runner::create(
		H160::default(),
		hex::decode(FOO_BAR_CONTRACT_CREATOR_BYTECODE.trim_end()).unwrap(),
		U256::zero(),
		gas_limit,
		Some(FixedGasPrice::min_gas_price().0),
		None,
		None,
		Vec::new(),
		Vec::new(),
		true, // transactional
		true, // must be validated
		weight_limit,
		Some(0),
		&<Test as Config>::config().clone(),
	)
}

#[test]
fn test_contract_deploy_succeeds_if_address_is_allowed() {
	new_test_ext().execute_with(|| {
		let gas_limit: u64 = 1_000_000;
		let weight_limit = FixedGasWeightMapping::<Test>::gas_to_weight(gas_limit, true);

		assert!(<Test as Config>::Runner::create(
			// Alith is allowed to deploy contracts
			H160::from([4u8; 20]),
			hex::decode(FOO_BAR_CONTRACT_CREATOR_BYTECODE.trim_end()).unwrap(),
			U256::zero(),
			gas_limit,
			Some(FixedGasPrice::min_gas_price().0),
			None,
			None,
			Vec::new(),
			Vec::new(),
			true, // transactional
			true, // must be validated
			Some(weight_limit),
			Some(0),
			&<Test as Config>::config().clone(),
		)
		.is_ok());
	});
}

#[test]
fn test_contract_deploy_fails_if_address_not_allowed() {
	new_test_ext().execute_with(|| {
		let gas_limit: u64 = 1_000_000;
		let weight_limit = FixedGasWeightMapping::<Test>::gas_to_weight(gas_limit, true);

		match <Test as Config>::Runner::create(
			// Bob is not allowed to deploy contracts
			H160::from([5u8; 20]),
			hex::decode(FOO_BAR_CONTRACT_CREATOR_BYTECODE.trim_end()).unwrap(),
			U256::zero(),
			gas_limit,
			Some(FixedGasPrice::min_gas_price().0),
			None,
			None,
			Vec::new(),
			Vec::new(),
			true, // transactional
			true, // must be validated
			Some(weight_limit),
			Some(0),
			&<Test as Config>::config().clone(),
		) {
			Err(RunnerError {
				error: Error::CreateOriginNotAllowed,
				..
			}) => (),
			_ => panic!("Should have failed with CreateOriginNotAllowed"),
		}
	});
}

#[test]
fn test_inner_contract_deploy_succeeds_if_address_is_allowed() {
	new_test_ext().execute_with(|| {
		let gas_limit: u64 = 1_000_000;
		let weight_limit = FixedGasWeightMapping::<Test>::gas_to_weight(gas_limit, true);

		let result1 = create_foo_bar_contract_creator(gas_limit, Some(weight_limit))
			.expect("create succeeds");

		let call_data: String = "2fc11060".to_owned();
		let call_contract_address = result1.value;

		let result = <Test as Config>::Runner::call(
			// Alith is allowed to deploy inner contracts
			H160::from([4u8; 20]),
			call_contract_address,
			hex::decode(&call_data).unwrap(),
			U256::zero(),
			gas_limit,
			Some(FixedGasPrice::min_gas_price().0),
			None,
			None,
			Vec::new(),
			Vec::new(),
			true, // transactional
			true, // must be validated
			Some(weight_limit),
			Some(0),
			&<Test as Config>::config().clone(),
		)
		.expect("call succeeds");

		assert_eq!(
			result.exit_reason,
			ExitReason::Succeed(ExitSucceed::Returned)
		);
	});
}

#[test]
fn test_inner_contract_deploy_reverts_if_address_not_allowed() {
	new_test_ext().execute_with(|| {
		let gas_limit: u64 = 1_000_000;
		let weight_limit = FixedGasWeightMapping::<Test>::gas_to_weight(gas_limit, true);

		let result1 = create_foo_bar_contract_creator(gas_limit, Some(weight_limit))
			.expect("create succeeds");

		let call_data: String = "2fc11060".to_owned();
		let call_contract_address = result1.value;

		let result = <Test as Config>::Runner::call(
			// Charleth is not allowed to deploy inner contracts
			H160::from([6u8; 20]),
			call_contract_address,
			hex::decode(&call_data).unwrap(),
			U256::zero(),
			gas_limit,
			Some(FixedGasPrice::min_gas_price().0),
			None,
			None,
			Vec::new(),
			Vec::new(),
			true, // transactional
			true, // must be validated
			Some(weight_limit),
			Some(0),
			&<Test as Config>::config().clone(),
		)
		.expect("call succeeds");

		assert_eq!(result.exit_reason, ExitReason::Revert(ExitRevert::Reverted));
	});
}

#[test]
fn fail_call_return_ok() {
	new_test_ext().execute_with(|| {
		assert_ok!(EVM::call(
			RuntimeOrigin::root(),
			H160::default(),
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::default(),
			1000000,
			U256::from(1_000_000_000),
			None,
			None,
			Vec::new(),
			Vec::new(),
		));

		assert_ok!(EVM::call(
			RuntimeOrigin::root(),
			H160::default(),
			H160::from_str("1000000000000000000000000000000000000002").unwrap(),
			Vec::new(),
			U256::default(),
			1000000,
			U256::from(1_000_000_000),
			None,
			None,
			Vec::new(),
			Vec::new(),
		));
	});
}

#[test]
fn fee_deduction() {
	new_test_ext().execute_with(|| {
		// Create an EVM address and the corresponding Substrate address that will be charged fees and refunded
		let evm_addr = H160::from_str("1000000000000000000000000000000000000003").unwrap();
		let substrate_addr = <Test as Config>::AddressMapping::into_account_id(evm_addr);

		// Seed account
		let _ = <Test as Config>::Currency::deposit_creating(&substrate_addr, 100);
		assert_eq!(Balances::free_balance(&substrate_addr), 100);

		// Deduct fees as 10 units
		let imbalance = <<Test as Config>::OnChargeTransaction as OnChargeEVMTransaction<Test>>::withdraw_fee(&evm_addr, U256::from(10)).unwrap();
		assert_eq!(Balances::free_balance(&substrate_addr), 90);

		// Refund fees as 5 units
		<<Test as Config>::OnChargeTransaction as OnChargeEVMTransaction<Test>>::correct_and_deposit_fee(&evm_addr, U256::from(5), U256::from(5), imbalance);
		assert_eq!(Balances::free_balance(&substrate_addr), 95);
	});
}

#[test]
fn ed_0_refund_patch_works() {
	new_test_ext().execute_with(|| {
		// Verifies that the OnChargeEVMTransaction patch is applied and fixes a known bug in Substrate for evm transactions.
		// https://github.com/paritytech/substrate/issues/10117
		let evm_addr = H160::from_str("1000000000000000000000000000000000000003").unwrap();
		let substrate_addr = <Test as Config>::AddressMapping::into_account_id(evm_addr);

		let _ = <Test as Config>::Currency::deposit_creating(&substrate_addr, 21_777_000_000_000);
		assert_eq!(Balances::free_balance(&substrate_addr), 21_777_000_000_000);

		let _ = EVM::call(
			RuntimeOrigin::root(),
			evm_addr,
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::from(1_000_000_000),
			21776,
			U256::from(1_000_000_000),
			None,
			Some(U256::from(0)),
			Vec::new(),
			Vec::new(),
		);
		// All that was due, was refunded.
		assert_eq!(Balances::free_balance(&substrate_addr), 776_000_000_000);
	});
}

#[test]
fn ed_0_refund_patch_is_required() {
	new_test_ext().execute_with(|| {
		// This test proves that the patch is required, verifying that the current Substrate behaviour is incorrect
		// for ED 0 configured chains.
		let evm_addr = H160::from_str("1000000000000000000000000000000000000003").unwrap();
		let substrate_addr = <Test as Config>::AddressMapping::into_account_id(evm_addr);

		let _ = <Test as Config>::Currency::deposit_creating(&substrate_addr, 100);
		assert_eq!(Balances::free_balance(&substrate_addr), 100);

		// Drain funds
		let _ =
			<<Test as Config>::OnChargeTransaction as OnChargeEVMTransaction<Test>>::withdraw_fee(
				&evm_addr,
				U256::from(100),
			)
			.unwrap();
		assert_eq!(Balances::free_balance(&substrate_addr), 0);

		// Try to refund. With ED 0, although the balance is now 0, the account still exists.
		// So its expected that calling `deposit_into_existing` results in the AccountData to increase the Balance.
		//
		// Is not the case, and this proves that the refund logic needs to be handled taking this into account.
		assert!(
			<Test as Config>::Currency::deposit_into_existing(&substrate_addr, 5u32.into())
				.is_err()
		);
		// Balance didn't change, and should be 5.
		assert_eq!(Balances::free_balance(substrate_addr), 0);
	});
}

#[test]
fn find_author() {
	new_test_ext().execute_with(|| {
		let author = EVM::find_author();
		assert_eq!(
			author,
			H160::from_str("1234500000000000000000000000000000000000").unwrap()
		);
	});
}

#[test]
fn reducible_balance() {
	new_test_ext().execute_with(|| {
		let evm_addr = H160::from_str("1000000000000000000000000000000000000001").unwrap();
		let account_id = <Test as Config>::AddressMapping::into_account_id(evm_addr);
		let existential = ExistentialDeposit::get();

		// Genesis Balance.
		let genesis_balance = EVM::account_basic(&evm_addr).0.balance;

		// Lock identifier.
		let lock_id: LockIdentifier = *b"te/stlok";
		// Reserve some funds.
		let to_lock = 1000;
		Balances::set_lock(lock_id, &account_id, to_lock, WithdrawReasons::RESERVE);
		// Reducible is, as currently configured in `account_basic`, (balance - lock - existential).
		let reducible_balance = EVM::account_basic(&evm_addr).0.balance;
		assert_eq!(reducible_balance, (genesis_balance - to_lock - existential));
	});
}

#[test]
fn author_should_get_tip() {
	new_test_ext().execute_with(|| {
		let author = EVM::find_author();
		let before_tip = EVM::account_basic(&author).0.balance;
		let result = EVM::call(
			RuntimeOrigin::root(),
			H160::default(),
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::from(1),
			1000000,
			U256::from(2_000_000_000),
			Some(U256::from(1)),
			None,
			Vec::new(),
			Vec::new(),
		);
		result.expect("EVM can be called");
		let after_tip = EVM::account_basic(&author).0.balance;
		assert_eq!(after_tip, (before_tip + 21000));
	});
}

#[test]
fn issuance_after_tip() {
	new_test_ext().execute_with(|| {
		let before_tip = <Test as Config>::Currency::total_issuance();
		let result = EVM::call(
			RuntimeOrigin::root(),
			H160::default(),
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::from(1),
			1000000,
			U256::from(2_000_000_000),
			Some(U256::from(1)),
			None,
			Vec::new(),
			Vec::new(),
		);
		result.expect("EVM can be called");
		let after_tip = <Test as Config>::Currency::total_issuance();
		// Only base fee is burned
		let base_fee: u64 = <Test as Config>::FeeCalculator::min_gas_price()
			.0
			.unique_saturated_into();
		assert_eq!(after_tip, (before_tip - (base_fee * 21_000)));
	});
}

#[test]
fn author_same_balance_without_tip() {
	new_test_ext().execute_with(|| {
		let author = EVM::find_author();
		let before_tip = EVM::account_basic(&author).0.balance;
		let _ = EVM::call(
			RuntimeOrigin::root(),
			H160::default(),
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::default(),
			1000000,
			U256::default(),
			None,
			None,
			Vec::new(),
			Vec::new(),
		);
		let after_tip = EVM::account_basic(&author).0.balance;
		assert_eq!(after_tip, before_tip);
	});
}

#[test]
fn refunds_should_work() {
	new_test_ext().execute_with(|| {
		let before_call = EVM::account_basic(&H160::default()).0.balance;
		// Gas price is not part of the actual fee calculations anymore, only the base fee.
		//
		// Because we first deduct max_fee_per_gas * gas_limit (2_000_000_000 * 1000000) we need
		// to ensure that the difference (max fee VS base fee) is refunded.
		let _ = EVM::call(
			RuntimeOrigin::root(),
			H160::default(),
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::from(1),
			1000000,
			U256::from(2_000_000_000),
			None,
			None,
			Vec::new(),
			Vec::new(),
		);
		let (base_fee, _) = <Test as Config>::FeeCalculator::min_gas_price();
		let total_cost = (U256::from(21_000) * base_fee) + U256::from(1);
		let after_call = EVM::account_basic(&H160::default()).0.balance;
		assert_eq!(after_call, before_call - total_cost);
	});
}

#[test]
fn refunds_and_priority_should_work() {
	new_test_ext().execute_with(|| {
		let author = EVM::find_author();
		let before_tip = EVM::account_basic(&author).0.balance;
		let before_call = EVM::account_basic(&H160::default()).0.balance;
		// We deliberately set a base fee + max tip > max fee.
		// The effective priority tip will be 1GWEI instead 1.5GWEI:
		// 		(max_fee_per_gas - base_fee).min(max_priority_fee)
		//		(2 - 1).min(1.5)
		let tip = U256::from(1_500_000_000);
		let max_fee_per_gas = U256::from(2_000_000_000);
		let used_gas = U256::from(21_000);
		let _ = EVM::call(
			RuntimeOrigin::root(),
			H160::default(),
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::from(1),
			1000000,
			max_fee_per_gas,
			Some(tip),
			None,
			Vec::new(),
			Vec::new(),
		);
		let (base_fee, _) = <Test as Config>::FeeCalculator::min_gas_price();
		let actual_tip = (max_fee_per_gas - base_fee).min(tip) * used_gas;
		let total_cost = (used_gas * base_fee) + actual_tip + U256::from(1);
		let after_call = EVM::account_basic(&H160::default()).0.balance;
		// The tip is deducted but never refunded to the caller.
		assert_eq!(after_call, before_call - total_cost);

		let after_tip = EVM::account_basic(&author).0.balance;
		assert_eq!(after_tip, (before_tip + actual_tip));
	});
}

#[test]
fn call_should_fail_with_priority_greater_than_max_fee() {
	new_test_ext().execute_with(|| {
		// Max priority greater than max fee should fail.
		let tip: u128 = 1_100_000_000;
		let result = EVM::call(
			RuntimeOrigin::root(),
			H160::default(),
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::from(1),
			1000000,
			U256::from(1_000_000_000),
			Some(U256::from(tip)),
			None,
			Vec::new(),
			Vec::new(),
		);
		assert!(result.is_err());
		// Some used weight is returned as part of the error.
		assert_eq!(
			result.unwrap_err().post_info.actual_weight,
			Some(Weight::from_parts(7, 0))
		);
	});
}

#[test]
fn call_should_succeed_with_priority_equal_to_max_fee() {
	new_test_ext().execute_with(|| {
		let tip: u128 = 1_000_000_000;
		// Mimics the input for pre-eip-1559 transaction types where `gas_price`
		// is used for both `max_fee_per_gas` and `max_priority_fee_per_gas`.
		let result = EVM::call(
			RuntimeOrigin::root(),
			H160::default(),
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::from(1),
			1000000,
			U256::from(1_000_000_000),
			Some(U256::from(tip)),
			None,
			Vec::new(),
			Vec::new(),
		);
		assert!(result.is_ok());
	});
}

#[test]
fn handle_sufficient_reference() {
	new_test_ext().execute_with(|| {
		let addr = H160::from_str("1230000000000000000000000000000000000001").unwrap();
		let addr_2 = H160::from_str("1234000000000000000000000000000000000001").unwrap();
		let substrate_addr: <Test as frame_system::Config>::AccountId =
			<Test as Config>::AddressMapping::into_account_id(addr);
		let substrate_addr_2: <Test as frame_system::Config>::AccountId =
			<Test as Config>::AddressMapping::into_account_id(addr_2);

		// Sufficients should increase when creating EVM accounts.
		<crate::AccountCodes<Test>>::insert(addr, vec![0]);
		let account = frame_system::Account::<Test>::get(substrate_addr);
		// Using storage is not correct as it leads to a sufficient reference mismatch.
		assert_eq!(account.sufficients, 0);

		// Using the create / remove account functions is the correct way to handle it.
		assert!(EVM::create_account(addr_2, vec![1, 2, 3], None).is_ok());
		let account_2 = frame_system::Account::<Test>::get(substrate_addr_2.clone());
		// We increased the sufficient reference by 1.
		assert_eq!(account_2.sufficients, 1);
		EVM::remove_account(&addr_2);
		let account_2 = frame_system::Account::<Test>::get(substrate_addr_2);
		assert_eq!(account_2.sufficients, 0);
	});
}

#[test]
fn runner_non_transactional_calls_with_non_balance_accounts_is_ok_without_gas_price() {
	// Expect to skip checks for gas price and account balance when both:
	//	- The call is non transactional (`is_transactional == false`).
	//	- The `max_fee_per_gas` is None.
	new_test_ext().execute_with(|| {
		let non_balance_account =
			H160::from_str("7700000000000000000000000000000000000001").unwrap();
		assert_eq!(
			EVM::account_basic(&non_balance_account).0.balance,
			U256::zero()
		);
		let _ = <Test as Config>::Runner::call(
			non_balance_account,
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::from(1u32),
			1000000,
			None,
			None,
			None,
			Vec::new(),
			Vec::new(),
			false, // non-transactional
			true,  // must be validated
			None,
			None,
			&<Test as Config>::config().clone(),
		)
		.expect("Non transactional call succeeds");
		assert_eq!(
			EVM::account_basic(&non_balance_account).0.balance,
			U256::zero()
		);
	});
}

#[test]
fn runner_non_transactional_calls_with_non_balance_accounts_is_err_with_gas_price() {
	// In non transactional calls where `Some(gas_price)` is defined, expect it to be
	// checked against the `BaseFee`, and expect the account to have enough balance
	// to pay for the call.
	new_test_ext().execute_with(|| {
		let non_balance_account =
			H160::from_str("7700000000000000000000000000000000000001").unwrap();
		assert_eq!(
			EVM::account_basic(&non_balance_account).0.balance,
			U256::zero()
		);
		let res = <Test as Config>::Runner::call(
			non_balance_account,
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::from(1u32),
			1000000,
			Some(U256::from(1_000_000_000)),
			None,
			None,
			Vec::new(),
			Vec::new(),
			false, // non-transactional
			true,  // must be validated
			None,
			None,
			&<Test as Config>::config().clone(),
		);
		assert!(res.is_err());
	});
}

#[test]
fn runner_transactional_call_with_zero_gas_price_fails() {
	// Transactional calls are rejected when `max_fee_per_gas == None`.
	new_test_ext().execute_with(|| {
		let res = <Test as Config>::Runner::call(
			H160::default(),
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::from(1u32),
			1000000,
			None,
			None,
			None,
			Vec::new(),
			Vec::new(),
			true, // transactional
			true, // must be validated
			None,
			None,
			&<Test as Config>::config().clone(),
		);
		assert!(res.is_err());
	});
}

#[test]
fn runner_max_fee_per_gas_gte_max_priority_fee_per_gas() {
	// Transactional and non transactional calls enforce `max_fee_per_gas >= max_priority_fee_per_gas`.
	new_test_ext().execute_with(|| {
		let res = <Test as Config>::Runner::call(
			H160::default(),
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::from(1u32),
			1000000,
			Some(U256::from(1_000_000_000)),
			Some(U256::from(2_000_000_000)),
			None,
			Vec::new(),
			Vec::new(),
			true, // transactional
			true, // must be validated
			None,
			None,
			&<Test as Config>::config().clone(),
		);
		assert!(res.is_err());
		let res = <Test as Config>::Runner::call(
			H160::default(),
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::from(1u32),
			1000000,
			Some(U256::from(1_000_000_000)),
			Some(U256::from(2_000_000_000)),
			None,
			Vec::new(),
			Vec::new(),
			false, // non-transactional
			true,  // must be validated
			None,
			None,
			&<Test as Config>::config().clone(),
		);
		assert!(res.is_err());
	});
}

#[test]
fn eip3607_transaction_from_contract() {
	new_test_ext().execute_with(|| {
		// external transaction
		match <Test as Config>::Runner::call(
			// Contract address.
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::from(1u32),
			1000000,
			None,
			None,
			None,
			Vec::new(),
			Vec::new(),
			true,  // transactional
			false, // not sure be validated
			None,
			None,
			&<Test as Config>::config().clone(),
		) {
			Err(RunnerError {
				error: Error::TransactionMustComeFromEOA,
				..
			}) => (),
			_ => panic!("Should have failed"),
		}

		// internal call
		assert!(<Test as Config>::Runner::call(
			// Contract address.
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			H160::from_str("1000000000000000000000000000000000000001").unwrap(),
			Vec::new(),
			U256::from(1u32),
			1000000,
			None,
			None,
			None,
			Vec::new(),
			Vec::new(),
			false, // non-transactional
			true,  // must be validated
			None,
			None,
			&<Test as Config>::config().clone(),
		)
		.is_ok());
	});
}

#[test]
fn metadata_code_gets_cached() {
	new_test_ext().execute_with(|| {
		let address = H160::repeat_byte(0xaa);

		assert!(crate::Pallet::<Test>::create_account(address, b"Exemple".to_vec(), None).is_ok());

		let metadata = crate::Pallet::<Test>::account_code_metadata(address);
		assert_eq!(metadata.size, 7);
		assert_eq!(
			metadata.hash,
			hex_literal::hex!("e8396a990fe08f2402e64a00647e41dadf360ba078a59ba79f55e876e67ed4bc")
				.into()
		);

		let metadata2 = <AccountCodesMetadata<Test>>::get(address).expect("to have metadata set");
		assert_eq!(metadata, metadata2);
	});
}

#[test]
fn metadata_empty_dont_code_gets_cached() {
	new_test_ext().execute_with(|| {
		let address = H160::repeat_byte(0xaa);

		let metadata = crate::Pallet::<Test>::account_code_metadata(address);
		assert_eq!(metadata.size, 0);
		assert_eq!(
			metadata.hash,
			hex_literal::hex!("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")
				.into()
		);

		assert!(<AccountCodesMetadata<Test>>::get(address).is_none());
	});
}

/// EIP-7939: CLZ (Count Leading Zeros) opcode tests
///
/// The CLZ opcode (0x1e) counts the number of leading zero bits in a 256-bit value.
/// Test vectors from https://eips.ethereum.org/EIPS/eip-7939
mod eip7939_clz_test {
	use super::*;
	use evm::ExitSucceed;
	use fp_evm::CreateInfo;

	// All contracts use: PUSH<n> value, CLZ, PUSH1 0, MSTORE, PUSH1 32, PUSH1 0, RETURN
	// For PUSH1 values (11 bytes runtime): PUSH11 pattern with MSTORE
	// For PUSH32 values (42 bytes runtime): CODECOPY pattern

	// CLZ(0x00) = 256
	const CLZ_ZERO: &str = "6a60001e60005260206000f3600052600b6015f3";

	// CLZ(0x01) = 255
	const CLZ_ONE: &str = "6a60011e60005260206000f3600052600b6015f3";

	// CLZ(0x8000...00) = 0 (2^255, MSB set)
	const CLZ_MSB_SET: &str = "602a600c600039602a6000f37f80000000000000000000000000000000000000000000000000000000000000001e60005260206000f3";

	// CLZ(0xffff...ff) = 0 (MAX_U256, all bits set)
	const CLZ_MAX_U256: &str = "602a600c600039602a6000f37fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1e60005260206000f3";

	// CLZ(0x4000...00) = 1 (2^254)
	const CLZ_SECOND_BIT: &str = "602a600c600039602a6000f37f40000000000000000000000000000000000000000000000000000000000000001e60005260206000f3";

	// CLZ(0x7fff...ff) = 1 (2^255 - 1)
	const CLZ_MSB_UNSET: &str = "602a600c600039602a6000f37f7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1e60005260206000f3";

	fn create_clz_test_contract(
		contract: &str,
		gas_limit: u64,
	) -> Result<CreateInfo, crate::RunnerError<crate::Error<Test>>> {
		<Test as Config>::Runner::create(
			H160::default(),
			hex::decode(contract.trim_end()).expect("Failed to decode contract"),
			U256::zero(),
			gas_limit,
			Some(FixedGasPrice::min_gas_price().0),
			None,
			None,
			Vec::new(),
			Vec::new(),
			true, // transactional
			true, // must be validated
			None,
			Some(0),
			<Test as Config>::config(),
		)
	}

	fn call_contract(
		contract_addr: H160,
		gas_limit: u64,
	) -> Result<CallInfo, crate::RunnerError<crate::Error<Test>>> {
		<Test as Config>::Runner::call(
			H160::default(),
			contract_addr,
			Vec::new(),
			U256::zero(),
			gas_limit,
			Some(FixedGasPrice::min_gas_price().0),
			None,
			None,
			Vec::new(),
			Vec::new(),
			true, // transactional
			true, // must be validated
			None,
			Some(0),
			<Test as Config>::config(),
		)
	}

	fn u256_to_bytes(value: u16) -> Vec<u8> {
		let mut result = vec![0u8; 32];
		result[30] = (value >> 8) as u8;
		result[31] = value as u8;
		result
	}

	fn run_clz_test(contract: &str, expected_clz: u16) {
		new_test_ext().execute_with(|| {
			let gas_limit: u64 = 1_000_000;

			let result = create_clz_test_contract(contract, gas_limit)
				.expect("contract deployment should succeed");
			assert_eq!(
				result.exit_reason,
				crate::ExitReason::Succeed(ExitSucceed::Returned)
			);

			let call_result =
				call_contract(result.value, gas_limit).expect("contract call should succeed");
			assert_eq!(
				call_result.exit_reason,
				crate::ExitReason::Succeed(ExitSucceed::Returned)
			);

			assert_eq!(call_result.value, u256_to_bytes(expected_clz));
		});
	}

	// EIP-7939 Test Vectors

	#[test]
	fn clz_of_0x00_returns_256() {
		run_clz_test(CLZ_ZERO, 256);
	}

	#[test]
	fn clz_of_0x01_returns_255() {
		run_clz_test(CLZ_ONE, 255);
	}

	#[test]
	fn clz_of_0x8000_returns_0() {
		run_clz_test(CLZ_MSB_SET, 0);
	}

	#[test]
	fn clz_of_0xffff_returns_0() {
		run_clz_test(CLZ_MAX_U256, 0);
	}

	#[test]
	fn clz_of_0x4000_returns_1() {
		run_clz_test(CLZ_SECOND_BIT, 1);
	}

	#[test]
	fn clz_of_0x7fff_returns_1() {
		run_clz_test(CLZ_MSB_UNSET, 1);
	}
}
