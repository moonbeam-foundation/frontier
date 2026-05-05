import WebSocket from "ws";
import { CONFIG } from "../config.js";

export class SteadySubscribers {
  private sockets: WebSocket[] = [];

  async start(): Promise<void> {
    const promises: Promise<WebSocket>[] = [];
    for (let i = 0; i < CONFIG.load.steadyNewHeads; i++) {
      promises.push(this.open("newHeads"));
    }
    for (let i = 0; i < CONFIG.load.steadyLogs; i++) {
      promises.push(this.open("logs", CONFIG.logsFilter));
    }
    this.sockets = await Promise.all(promises);
  }

  stop(): void {
    for (const s of this.sockets) s.terminate();
  }

  private open(kind: string, filter?: unknown): Promise<WebSocket> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(CONFIG.node.wsRpc);
      ws.once("open", () => {
        const params = filter !== undefined ? [kind, filter] : [kind];
        ws.send(
          JSON.stringify({
            jsonrpc: "2.0",
            id: 1,
            method: "eth_subscribe",
            params,
          }),
        );
        ws.on("message", () => {});
        resolve(ws);
      });
      ws.once("error", reject);
    });
  }
}
