// This file is part of Frontier.

// Copyright (c) Moonsong Labs.
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

// “secp256r1” is a specific elliptic curve, also known as “P-256” and “prime256v1” curves.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use p256::ecdsa::{signature::hazmat::PrehashVerifier, Signature, VerifyingKey};
use fp_evm::{ExitSucceed, PrecompileResult, Precompile, PrecompileHandle, PrecompileOutput};

pub struct P256Verify;

impl P256Verify {
	/// https://github.com/ethereum/RIPs/blob/master/RIPS/rip-7212.md#precompiled-contract-gas-usage
	const BASE_GAS: u64 = 3_450;
}


/// Implements RIP-7212 P256VERIFY precompile.
/// https://github.com/ethereum/RIPs/blob/master/RIPS/rip-7212.md
impl Precompile for P256Verify {

	fn execute(handle: &mut impl PrecompileHandle) -> PrecompileResult {
		handle.record_cost(Self::BASE_GAS)?;

		let input = handle.input();

		let result = if verify_impl(input).is_some() {
			// If the signature verification process succeeds, it returns 1 in 32 bytes format.
			let mut result = [0u8; 32];
			result[31] = 1;

			result.to_vec()
		} else {
			// If the signature verification process fails, it does not return any output data.
			[0u8; 0].to_vec()
		};

		Ok(PrecompileOutput {
			exit_status: ExitSucceed::Returned,
			output: result.to_vec(),
		})
	}
}

/// (Signed payload) Hash of the original message
/// 32 bytes of the signed data hash
fn message_hash(input: &[u8]) -> &[u8] {
	&input[..32]
}

/// r and s signature components
fn signature(input: &[u8]) -> &[u8] {
	&input[32..96]
}

/// x and y coordinate of the public key
fn public_key(input: &[u8]) -> &[u8] {
	&input[96..160]
}

/// Returns `Some(())` if the signature included in the input byte slice is
/// valid, `None` otherwise.
pub fn verify_impl(input: &[u8]) -> Option<()> {
	// Input data: 160 bytes of data including:
	// - 32 bytes of the signed data hash
	// - 32 bytes of the r component of the signature
	// - 32 bytes of the s component of the signature
	// - 32 bytes of the x coordinate of the public key
	// - 32 bytes of the y coordinate of the public key
	if input.len() != 160 {
		return None;
	}

	let message_hash = message_hash(input);
	let signature = signature(input);
	let public_key = public_key(input);

	let mut uncompressed_pk = [0u8; 65];
	// (0x04) prefix indicates the public key is in its uncompressed from
	uncompressed_pk[0] = 0x04;
	uncompressed_pk[1..].copy_from_slice(public_key);

	// Will only fail if the signature is not exactly 64 bytes
	let signature = Signature::from_slice(signature).ok()?;

	let public_key = VerifyingKey::from_sec1_bytes(&uncompressed_pk).ok()?;

	public_key.verify_prehash(message_hash, &signature).ok()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_empty_input() -> Result<(), PrecompileFailure> {
		let input: [u8; 0] = [];
		let cost: u64 = 1;

		match P256Verify::execute(&input, cost) {
			Ok((_, result)) => {
				if result != [0u8; 32] {
					panic!("Test not expected to pass");
				}
				Ok(())
			}
			Err(e) => {
				assert_eq!(
					e,
					PrecompileFailure::Error {
						exit_status: ExitError::Other("input must contain 128 bytes".into())
					}
				);
				Ok(())
			}
		}
	}
}
