import {
  createPublicClient,
  createWalletClient,
  http,
  parseAbi,
} from "viem";
import { privateKeyToAccount } from "viem/accounts";
import { CONFIG } from "../config.js";

const ABI = parseAbi([
  "function emitBurst(uint256 count)",
  "function emitLogs(uint256 n)",
]);

/** Frontier template uses `fp_evm::MAX_TRANSACTION_GAS_LIMIT` (EIP-7825, 2^24). */
const FRONTIER_MAX_TX_GAS = 16_777_216n;

export class TxSpammer {
  private running = false;
  private sent = 0;
  private failed = 0;

  async start(): Promise<void> {
    if (!CONFIG.load.txSpammer.enabled) return;
    if (
      CONFIG.load.txSpammer.contract ===
      "0x0000000000000000000000000000000000000000"
    ) {
      console.error(
        "[spammer] no contract address; did you run contracts/deploy.sh?",
      );
      return;
    }

    this.running = true;
    const pub = createPublicClient({ transport: http(CONFIG.node.httpRpc) });
    const chainId = await pub.getChainId();

    const keys =
      CONFIG.load.txSpammer.accounts.length > 0
        ? CONFIG.load.txSpammer.accounts
        : [CONFIG.load.txSpammer.privateKey as `0x${string}`];

    const devChain = {
      id: chainId,
      name: "frontier-dev",
      network: "frontier-dev",
      nativeCurrency: { name: "DEV", symbol: "DEV", decimals: 18 },
      rpcUrls: {
        default: { http: [CONFIG.node.httpRpc] },
        public: { http: [CONFIG.node.httpRpc] },
      },
    } as const;

    const wallets = keys.map((pk) => {
      const account = privateKeyToAccount(pk);
      return {
        account,
        client: createWalletClient({
          account,
          transport: http(CONFIG.node.httpRpc),
          chain: devChain,
        }),
      };
    });

    const intervalMs = 60_000 / CONFIG.load.txSpammer.txPerMinute;
    let i = 0;

    const heartbeat = setInterval(() => {
      console.log(`[spammer] sent=${this.sent} failed=${this.failed}`);
    }, 30_000);

    while (this.running) {
      const w = wallets[i % wallets.length]!;
      i++;
      try {
        const args = [BigInt(CONFIG.load.txSpammer.logsPerTx)] as const;
        let gas: bigint;
        try {
          const est = await pub.estimateContractGas({
            address: CONFIG.load.txSpammer.contract,
            abi: ABI,
            functionName: "emitBurst",
            args,
            account: w.account,
          });
          gas = (est * 115n) / 100n;
          if (gas > FRONTIER_MAX_TX_GAS) gas = FRONTIER_MAX_TX_GAS;
        } catch {
          gas = FRONTIER_MAX_TX_GAS;
        }

        const hash = await w.client.writeContract({
          address: CONFIG.load.txSpammer.contract,
          abi: ABI,
          functionName: "emitBurst",
          args,
          gas,
        });
        await pub.waitForTransactionReceipt({ hash });
        this.sent++;
      } catch (e: unknown) {
        this.failed++;
        const msg = e instanceof Error ? e.message : String(e);
        if (!/nonce|already known/i.test(msg)) {
          console.error(`[spammer] ${msg}`);
        }
      }
      await sleep(intervalMs);
    }

    clearInterval(heartbeat);
    console.log(`[spammer] stopped; sent=${this.sent} failed=${this.failed}`);
  }

  stop(): void {
    this.running = false;
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
