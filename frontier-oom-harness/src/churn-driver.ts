import WebSocket from "ws";
import { CONFIG } from "../config.js";

export class ChurnDriver {
  private running = false;

  async start(): Promise<void> {
    this.running = true;
    while (this.running) {
      await this.runCycle().catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : String(e);
        console.error(`[churn] cycle error: ${msg}`);
      });
      await sleep(CONFIG.load.churnCycleMs);
    }
  }

  stop(): void {
    this.running = false;
  }

  private async runCycle(): Promise<void> {
    const subs: Promise<void>[] = [];
    for (let i = 0; i < CONFIG.load.churnNewHeadsPerCycle; i++) {
      subs.push(openAndClose("newHeads"));
    }
    for (let i = 0; i < CONFIG.load.churnLogsPerCycle; i++) {
      subs.push(openAndClose("logs", CONFIG.logsFilter));
    }
    await Promise.allSettled(subs);
  }
}

async function openAndClose(kind: string, filter?: unknown): Promise<void> {
  const ws = new WebSocket(CONFIG.node.wsRpc);
  await waitOpen(ws);

  const params = filter !== undefined ? [kind, filter] : [kind];
  ws.send(
    JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      method: "eth_subscribe",
      params,
    }),
  );

  await new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, 5_000);
    ws.once("close", () => {
      clearTimeout(timer);
      resolve();
    });
    ws.once("error", () => {
      clearTimeout(timer);
      resolve();
    });
  });

  ws.terminate();
}

function waitOpen(ws: WebSocket): Promise<void> {
  return new Promise((resolve, reject) => {
    ws.once("open", resolve);
    ws.once("error", reject);
  });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
