#!/usr/bin/env bash
set -euo pipefail

# Prefer Foundry (foundryup) over an unrelated npm `forge` on PATH.
if [[ -x "${HOME}/.foundry/bin/forge" ]]; then
  export PATH="${HOME}/.foundry/bin:${PATH}"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -f lib/forge-std/src/Script.sol ]]; then
  echo "Installing forge-std (first run only)..."
  forge install foundry-rs/forge-std --no-commit
fi

export DEPLOYER_PRIVATE_KEY="${DEPLOYER_PRIVATE_KEY:-0x5fb92d6e98884f76de468fa3f6278f8807c48bebc13595d45af5bdc4da702133}"
# export DEV_RPC_URL="${DEV_RPC_URL:-http://127.0.0.1:9944}"
export DEV_RPC_URL="${DEV_RPC_URL:-http://127.0.0.1:8800}"

export TRANSFER_WEIGHT="${TRANSFER_WEIGHT:-70}"
export SWAP_WEIGHT="${SWAP_WEIGHT:-20}"
export BLOB_WEIGHT="${BLOB_WEIGHT:-10}"
export BLOB_BYTES="${BLOB_BYTES:-256}"

mkdir -p "${ROOT}/runs"

echo "Deploying LogEmitter to ${DEV_RPC_URL}..."
echo "  weights: transfer=${TRANSFER_WEIGHT} swap=${SWAP_WEIGHT} blob=${BLOB_WEIGHT}"
echo "  blobBytes: ${BLOB_BYTES}"

forge build
CHAIN_ID="$(cast chain-id --rpc-url "${DEV_RPC_URL}")"

forge script script/Deploy.s.sol:Deploy \
  --rpc-url "${DEV_RPC_URL}" \
  --broadcast \
  --legacy \
  --skip-simulation \
  --private-key "${DEPLOYER_PRIVATE_KEY}" \
  -vvv

RUN_JSON="broadcast/Deploy.s.sol/${CHAIN_ID}/run-latest.json"
if [[ ! -f "${RUN_JSON}" ]]; then
  echo "Missing broadcast output: ${RUN_JSON}" >&2
  exit 1
fi

ADDR="$(
  jq -r '
    [ .transactions[]? | select(.contractAddress != null and .contractAddress != "") | .contractAddress ]
    | first
    // (.receipts[0].contractAddress // empty)
  ' "${RUN_JSON}"
)"

if [[ "${ADDR}" == "null" || -z "${ADDR}" ]]; then
  echo "Failed to parse deployed address from ${RUN_JSON}" >&2
  exit 1
fi

echo "Deployed at: ${ADDR}"
echo "${ADDR}" > "${ROOT}/runs/.log-emitter-address"
echo "Wrote address to runs/.log-emitter-address (tx spammer auto-enables on next harness run when this file exists)."
