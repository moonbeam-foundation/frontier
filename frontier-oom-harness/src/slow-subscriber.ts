import type { Socket } from "node:net";
import WebSocket from "ws";
import { CONFIG } from "../config.js";

/**
 * Long-lived WS that subscribes to logs, receives a few messages, then pauses
 * the underlying TCP socket so the server send path backpressures (journal 0 §4.5).
 */
export class SlowSubscriber {
  private ws?: WebSocket;

  async start(): Promise<void> {
    this.ws = new WebSocket(CONFIG.node.wsRpc);
    await new Promise<void>((res, rej) => {
      this.ws!.once("open", res);
      this.ws!.once("error", rej);
    });

    this.ws.send(
      JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "eth_subscribe",
        params: ["logs", CONFIG.logsFilter],
      }),
    );

    let received = 0;
    this.ws.on("message", () => {
      received++;
      if (received >= 10) {
        this.ws!.removeAllListeners("message");
        const sock = (this.ws as unknown as { _socket?: Socket })._socket;
        if (sock && typeof sock.pause === "function") {
          sock.pause();
        }
      }
    });
  }

  stop(): void {
    this.ws?.terminate();
  }
}
