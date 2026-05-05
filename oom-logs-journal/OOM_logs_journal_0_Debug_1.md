# `emitLogs` contract + deployment for the harness

A minimal but realistic log-emitter that the tx-spammer can drive to produce the kind of log density that actually stresses `logs_journal` (Moonbeam-class: many logs per tx, realistic topic/data sizes). Plus a Foundry-based deploy script and updated harness wiring so you just run one command.

---

## 1. The contract

`contracts/src/LogEmitter.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity .8.24;

/// @title LogEmitter
/// @notice Emits a configurable burst of events per call, with realistic
///         topic/data sizes, for load-testing log infrastructure.
///
/// Three event shapes, chosen to mirror real-world Ethereum traffic:
///   - `Transfer`   : 3 indexed topics + 32 bytes data  (ERC-20-like, the
///                    overwhelmingly common shape)
///   - `Swap`       : 2 indexed topics + 128 bytes data (Uniswap-V2-like)
///   - `Blob`       : 1 indexed topic  + arbitrary data (bridge/oracle-like)
///
/// The `emitBurst` entrypoint emits all three shapes in proportion, so a
/// single tx generates a realistic mix. The mix is tunable via constructor.
contract LogEmitter {
    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId, uint256 value);
    event Swap(
        address indexed sender,
        address indexed recipient,
        int256 amount0,
        int256 amount1,
        uint160 sqrtPriceX96,
        uint128 liquidity,
        int24  tick
    );
    event Blob(bytes32 indexed key, bytes payload);

    // Mix ratios: for every `transferWeight + swapWeight + blobWeight` logs,
    // roughly that proportion is of each type. Defaults tuned to look like
    // a log-heavy EVM chain (Transfer-dominated).
    uint8 public immutable transferWeight;
    uint8 public immutable swapWeight;
    uint8 public immutable blobWeight;
    uint16 public immutable blobBytes;

    uint256 private _nonce;

    constructor(uint8 _transferWeight, uint8 _swapWeight, uint8 _blobWeight, uint16 _blobBytes) {
        require(_transferWeight + _swapWeight + _blobWeight > 0, "zero weights");
        require(_blobBytes <= 4096, "blob too large");
        transferWeight = _transferWeight;
        swapWeight     = _swapWeight;
        blobWeight     = _blobWeight;
        blobBytes      = _blobBytes;
    }

    /// @notice Emit `count` logs, distributed across the three shapes per the
    ///         configured weights. Gas scales roughly linearly with `count`,
    ///         so cap at what fits in a block on your target chain.
    function emitBurst(uint256 count) external {
        uint256 total = uint256(transferWeight) + swapWeight + blobWeight;
        uint256 n = _nonce;

        for (uint256 i = 0; i < count; i++) {
            uint256 slot = (n + i) % total;
            if (slot < transferWeight) {
                _emitTransfer(n + i);
            } else if (slot < transferWeight + swapWeight) {
                _emitSwap(n + i);
            } else {
                _emitBlob(n + i);
            }
        }
        _nonce = n + count;
    }

    /// @notice Fire-and-forget helper matching the harness's current shape.
    function emitLogs(uint256 n) external {
        // Keep this as an alias so the harness config doesn't need changes
        // if you'd prefer the old name.
        this.emitBurst(n);
    }

    // ------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------

    function _emitTransfer(uint256 seed) internal {
        address from = address(uint160(uint256(keccak256(abi.encode(seed, "from")))));
        address to   = address(uint160(uint256(keccak256(abi.encode(seed, "to")))));
        uint256 tokenId = uint256(keccak256(abi.encode(seed, "tok")));
        uint256 value   = uint256(keccak256(abi.encode(seed, "val")));
        emit Transfer(from, to, tokenId, value);
    }

    function _emitSwap(uint256 seed) internal {
        address sender    = address(uint160(uint256(keccak256(abi.encode(seed, "s")))));
        address recipient = address(uint160(uint256(keccak256(abi.encode(seed, "r")))));
        int256  a0 = int256(uint256(keccak256(abi.encode(seed, "a0"))));
        int256  a1 = int256(uint256(keccak256(abi.encode(seed, "a1"))));
        uint160 sp = uint160(uint256(keccak256(abi.encode(seed, "sp"))));
        uint128 lq = uint128(uint256(keccak256(abi.encode(seed, "lq"))));
        int24   tk = int24(int256(uint256(keccak256(abi.encode(seed, "tk")))));
        emit Swap(sender, recipient, a0, a1, sp, lq, tk);
    }

    function _emitBlob(uint256 seed) internal {
        bytes32 key = keccak256(abi.encode(seed, "key"));
        uint256 len = blobBytes;
        bytes memory payload = new bytes(len);
        // Fill with pseudo-random chunks of 32 bytes.
        for (uint256 off = 0; off < len; off += 32) {
            bytes32 chunk = keccak256(abi.encode(seed, off));
            uint256 remaining = len - off;
            uint256 copyLen = remaining < 32 ? remaining : 32;
            for (uint256 j = 0; j < copyLen; j++) {
                payload[off + j] = chunk[j];
            }
        }
        emit Blob(key, payload);
    }
}
```

Why this shape:

- **Three event types** mimic the three log distributions Frontier actually sees in the wild. A test that emits only identical events misses cache/alloc pathologies that hit when topics vary.
- **Topics derived via `keccak256`** ensures varying topic values so Bloom filter logic in `log_matches_filter` actually exercises its per-log path rather than short-circuiting.
- **Blob of configurable bytes** lets you explicitly target `max_bytes_per_entry`. Set `blobBytes = 4096` and emit 10 blobs per tx → one tx produces ~40 KiB of log data, the kind of density that matters for 2.4 and 2.6.
- **Weights tunable at deploy** so you can re-test with different mixes without recompiling or redeploying if you parameterize via a factory. For now: deploy one instance per mix you care about.

Gas note: this contract will burn gas fast. That's fine on a Frontier `--dev` node since gas is free-ish, but size `emitBurst(n)` so `n * worst_case_per_event_gas` stays under block gas limit. Empirically `n = 50` is safe on default Frontier gas limits; `n = 200` usually works; beyond that you'll hit out-of-gas.

---

## 2. Foundry setup

`contracts/foundry.toml`:

```toml
[profile.default]
src = "src"
out = "out"
libs = ["lib"]
solc = "0.8.24"
optimizer = true
optimizer_runs = 200
evm_version = "london"   # widely compatible across Frontier-based chains

[rpc_endpoints]
dev = "${DEV_RPC_URL}"
```

`contracts/.gitignore`:

```
out/
cache/
broadcast/
```

---

## 3. Deploy script

`contracts/script/Deploy.s.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity .8.24;

import "forge-std/Script.sol";
import "../src/LogEmitter.sol";

contract Deploy is Script {
    function run() external {
        uint256 pk = vm.envUint("DEPLOYER_PRIVATE_KEY");
        // Tunables via env; defaults target "Moonbeam-ish" mix.
        uint8  transferWeight = uint8(vm.envOr("TRANSFER_WEIGHT", uint256(70)));
        uint8  swapWeight     = uint8(vm.envOr("SWAP_WEIGHT",     uint256(20)));
        uint8  blobWeight     = uint8(vm.envOr("BLOB_WEIGHT",     uint256(10)));
        uint16 blobBytes      = uint16(vm.envOr("BLOB_BYTES",     uint256(256)));

        vm.startBroadcast(pk);
        LogEmitter le = new LogEmitter(transferWeight, swapWeight, blobWeight, blobBytes);
        vm.stopBroadcast();

        console.log("LogEmitter deployed at:", address(le));
        console.log("  transferWeight=%s swapWeight=%s blobWeight=%s", transferWeight, swapWeight, blobWeight);
        console.log("  blobBytes=%s", blobBytes);
    }
}
```

---

## 4. A one-shot bash wrapper

`contracts/deploy.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Frontier --dev node exposes Alice's Substrate-mapped EVM account. Standard
# dev key across substrate-based EVM chains.
export DEPLOYER_PRIVATE_KEY="${DEPLOYER_PRIVATE_KEY:-0x5fb92d6e98884f76de468fa3f6278f8807c48bebc13595d45af5bdc4da702133}"
export DEV_RPC_URL="${DEV_RPC_URL:-http://127.0.0.1:9944}"

# Defaults: Moonbeam-ish mix. Override via env.
export TRANSFER_WEIGHT="${TRANSFER_WEIGHT:-70}"
export SWAP_WEIGHT="${SWAP_WEIGHT:-20}"
export BLOB_WEIGHT="${BLOB_WEIGHT:-10}"
export BLOB_BYTES="${BLOB_BYTES:-256}"

echo "Deploying LogEmitter to ${DEV_RPC_URL}..."
echo "  weights: transfer=${TRANSFER_WEIGHT} swap=${SWAP_WEIGHT} blob=${BLOB_WEIGHT}"
echo "  blobBytes: ${BLOB_BYTES}"

forge script script/Deploy.s.sol:Deploy \
    --rpc-url "${DEV_RPC_URL}" \
    --broadcast \
    --legacy \
    --skip-simulation \
    -vvv

# Extract the deployed address from the broadcast artifact and write it
# where the TS harness can pick it up.
ADDR=$(jq -r '.receipts[0].contractAddress' \
  "broadcast/Deploy.s.sol/$(cast chain-id --rpc-url "${DEV_RPC_URL}")/run-latest.json")

if [[ "${ADDR}" == "null" || -z "${ADDR}" ]]; then
  echo "Failed to parse deployed address from broadcast artifact" >&2
  exit 1
fi

echo "Deployed at: ${ADDR}"
echo "${ADDR}" > ../runs/.log-emitter-address
echo "Wrote address to runs/.log-emitter-address"
```

Make executable: `chmod +x contracts/deploy.sh`.

The `--legacy` flag matters on Frontier: EIP-1559 support varies by runtime version, and legacy txs are the safest lowest-common-denominator path. `--skip-simulation` avoids a failure mode where `eth_call`-based simulation behaves differently from actual execution on some Frontier versions.

The `cast chain-id` resolution picks the chain ID dynamically; on Frontier dev this is usually `42` but check your runtime constants. If `cast` isn't installed, replace with a hard-coded chain id you know.

---

## 5. Harness wiring updates

### 5.1 `config.ts` — read the deployed address

```typescript
import { readFileSync, existsSync } from "node:fs";

function readDeployedAddress(): string {
  const path = "./runs/.log-emitter-address";
  if (existsSync(path)) {
    return readFileSync(path, "utf8").trim();
  }
  return "0x0000000000000000000000000000000000000000";
}

export const CONFIG = {
  // ... existing fields ...

  load: {
    // ... existing ...

    txSpammer: {
      enabled: true,
      contract: readDeployedAddress() as `0x${string}`,
      privateKey: "0x5fb92d6e98884f76de468fa3f6278f8807c48bebc13595d45af5bdc4da702133",

      // Tx shape: calls LogEmitter.emitBurst(logsPerTx) at the configured rate.
      txPerMinute: 60,
      logsPerTx: 50,

      // Optional: use multiple prefunded accounts in rotation so nonces don't
      // serialize through a single account. Set to [] to disable rotation.
      accounts: [] as `0x${string}`[],
    },
  },
};
```

### 5.2 `src/tx-spammer.ts` — use `emitBurst` with rotation

```typescript
import {
  createWalletClient, createPublicClient, http, parseAbi,
} from "viem";
import { privateKeyToAccount } from "viem/accounts";
import { CONFIG } from "../config.js";

const ABI = parseAbi([
  "function emitBurst(uint256 count)",
  "function emitLogs(uint256 n)",
]);

export class TxSpammer {
  private running = false;
  private sent = 0;
  private failed = 0;

  async start(): Promise<void> {
    if (!CONFIG.load.txSpammer.enabled) return;
    if (CONFIG.load.txSpammer.contract === "0x0000000000000000000000000000000000000000") {
      console.error("[spammer] no contract address; did you run contracts/deploy.sh?");
      return;
    }

    this.running = true;
    const pub = createPublicClient({ transport: http(CONFIG.node.httpRpc) });
    const chainId = await pub.getChainId();

    const keys = CONFIG.load.txSpammer.accounts.length > 0
      ? CONFIG.load.txSpammer.accounts
      : [CONFIG.load.txSpammer.privateKey as `0x${string}`];

    const wallets = keys.map((pk) => {
      const account = privateKeyToAccount(pk);
      return {
        account,
        client: createWalletClient({
          account, transport: http(CONFIG.node.httpRpc),
          // Inline chain descriptor so viem doesn't try to autodetect.
          chain: { id: chainId, name: "frontier-dev", network: "frontier-dev",
                   nativeCurrency: { name: "DEV", symbol: "DEV", decimals: 18 },
                   rpcUrls: { default: { http: [CONFIG.node.httpRpc] },
                              public:  { http: [CONFIG.node.httpRpc] } } } as any,
        }),
      };
    });

    const intervalMs = 60_000 / CONFIG.load.txSpammer.txPerMinute;
    let i = 0;

    // Heartbeat log every 30s so you can see the spammer is alive.
    const heartbeat = setInterval(() => {
      console.log(`[spammer] sent=${this.sent} failed=${this.failed}`);
    }, 30_000);

    while (this.running) {
      const w = wallets[i % wallets.length];
      i++;
      try {
        await w.client.writeContract({
          address: CONFIG.load.txSpammer.contract,
          abi: ABI,
          functionName: "emitBurst",
          args: [BigInt(CONFIG.load.txSpammer.logsPerTx)],
          // Legacy gas to match the deploy script's transport assumptions.
          gas: 20_000_000n,
        });
        this.sent++;
      } catch (e: any) {
        this.failed++;
        // Common failures we don't want to spam-log: nonce races during rotation
        // are self-healing as viem refetches. Log everything else loudly.
        if (!/nonce/i.test(e.message ?? "")) {
          console.error(`[spammer] ${e.message ?? e}`);
        }
      }
      await sleep(intervalMs);
    }

    clearInterval(heartbeat);
    console.log(`[spammer] stopped; sent=${this.sent} failed=${this.failed}`);
  }

  stop() { this.running = false; }
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));
```

Key behaviors:
- **Chain autodetect**: `pub.getChainId()` handles the difference between standard Frontier (42) and custom runtimes without config tweaks.
- **Account rotation**: if you fill `CONFIG.load.txSpammer.accounts`, the spammer cycles through keys so nonce contention doesn't serialize tx submission. With one account at 60 tx/min that's rarely necessary; at 600 tx/min it matters.
- **Heartbeat**: 30-second sent/failed counter so you can see from stdout that the spammer is actually landing txs without tailing node logs.

---

## 6. Sizing: how much log volume does this actually produce?

With defaults (`logsPerTx = 50`, `txPerMinute = 60`, 70/20/10 Transfer/Swap/Blob mix, `blobBytes = 256`):

- Per tx: 50 events. ~35 Transfer (3 topics + 32B data), ~10 Swap (2 topics + 128B data), ~5 Blob (1 topic + 256B data).
- Average event size (topics + data, serialized): `(35 * 128) + (10 * 224) + (5 * 288)` / 50 ≈ **163 bytes/log**.
- Per tx: ~8 KiB of log bytes.
- Per minute: 60 tx × 8 KiB = **~480 KiB/min** of log bytes flowing through `logs_journal`.
- Per hour: **~28 MiB/hr** of log bytes.

Over a 2-hour run that's ~56 MiB of cumulative log bytes — enough to comfortably fill and cycle the journal's VecDeque (512 MiB default, 64 MiB if you applied the 2.5 policy fix), without overwhelming the node. If you want to stress harder, bump `txPerMinute` to 300 (**~140 MiB/hr**) and `logsPerTx` to 100 (**~560 KiB/tx**).

To specifically target the per-entry bounds from #1881:
- `max_bytes_per_entry = 4 MiB` → set `logsPerTx = 4000` (or raise `blobBytes` to 4096 and use `logsPerTx = 200`) on a chain with high gas limit to deliberately trip the incomplete-marker path.
- `max_logs_per_entry = 10_000` → harder to hit in a single tx, easier via a deep reorg against a spammed block range.

---

## 7. Putting it all together — the end-to-end run

One-shot from scratch:

```bash
# Terminal 1: node
./frontier-node --dev --tmp \
  --rpc-external --ws-external --rpc-cors=all \
  --prometheus-external --prometheus-port 9615 \
  --sealing=interval=2000

# Terminal 2: deploy
cd contracts && ./deploy.sh && cd ..

# Terminal 2 (continued): baseline run
pnpm run:master     # writes runs/master-<ts>/

# Rebuild node with patches, restart it, then:
cd contracts && ./deploy.sh && cd ..    # redeploy to the new chain state
pnpm run:patched    # writes runs/patched-<ts>/

# Diff:
pnpm report master-<ts>
pnpm report patched-<ts>
```

A Makefile wrapping this sequence is worth adding:

`Makefile`:

```makefile
.PHONY: deploy smoke baseline patched report-latest

deploy:
	cd contracts && ./deploy.sh

smoke: deploy
	DURATION_MS=$$((15*60*1000)) pnpm run:master

baseline: deploy
	pnpm run:master

patched: deploy
	pnpm run:patched

report-latest:
	@LATEST=$$(ls -1 runs/ | grep -v '^\.' | tail -1); \
	pnpm report $$LATEST
```

(`DURATION_MS` needs a small addition in `config.ts` to read env override — one-liner: `durationMs: Number(process.env.DURATION_MS ?? 2 * 60 * 60 * 1000)`.)

---

## 8. Verifying the spammer is doing what you think

Quick sanity checks before committing to a 2-hour baseline run:

**Check 1: receipts show expected log count.** Query the latest block and count logs:

```bash
cast rpc eth_getLogs '[{"fromBlock":"latest","toBlock":"latest"}]' \
  --rpc-url http://127.0.0.1:9944 | jq 'length'
```

You should see roughly `txPerMinute / 60 × logsPerTx × blockTimeSeconds` logs per block. With defaults and 2s blocks: `60/60 × 50 × 2 = 100` logs/block on average, give or take queuing.

**Check 2: subscriber actually receives them.** In a separate terminal:

```bash
wscat -c ws://127.0.0.1:9944
> {"jsonrpc":"2.0","id":1,"method":"eth_subscribe","params":["logs",{}]}
```

You should see a flood of log notifications at the cadence the spammer is producing.

**Check 3: journal is being exercised.** If you added the optional `frontier_logs_journal_entries_total_bytes` gauge:

```bash
curl -s http://127.0.0.1:9615/metrics | grep logs_journal
```

The `entries_total_bytes` should grow during spammer activity and plateau when bounded.

---

## 9. What to commit vs. what to gitignore

Commit:
- `contracts/src/LogEmitter.sol`
- `contracts/script/Deploy.s.sol`
- `contracts/foundry.toml`
- `contracts/deploy.sh`
- `contracts/.gitignore`
- `Makefile`

Gitignore (via `runs/.gitignore`):

```
*
!.gitignore
```

— so run artifacts stay local per developer while the directory itself is preserved.

---

With this in place, the harness has everything it needs to produce reproducible, realistic load. The whole flow — `forge build`, `./deploy.sh`, `pnpm run:master`, `pnpm run:patched`, `pnpm report` — fits on a single page of copy-pastable commands, which is what you want when handing this to a colleague or the upstream maintainers to verify.
