#!/usr/bin/env python3
"""A2A v1.0 server that puts the AgentMart LangGraph ecosystem on the network.

The lab's own transport is ``langgraph-workshop-in-process``: the A2A envelopes in
``agentmart_ecosystem.py`` are real data structures, but they never leave the process,
so an outside agent has nothing to call. This module is the missing front door.

It speaks the same wire format Hermes' ``a2a_call`` tool sends:

    GET  /.well-known/agent-card.json   -> Agent Card (v0.2 ``agent.json`` also answers)
    POST /                              -> JSON-RPC 2.0 ``SendMessage``

A task's text is fed to ``run_agentmart()`` unchanged, so the graph runs exactly as it
does on the CLI -- including the lab's own ``hermes_myshopper_node``. The remote caller
is a second buying agent in front of it, not a replacement for it.

Usage:
    python a2a_server.py                      # 127.0.0.1:9901
    python a2a_server.py --port 9901 --dry-run
    AGENTMART_A2A_TOKEN=secret python a2a_server.py --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from agentmart_ecosystem import (
    DEFAULT_CUSTOMER_ID,
    classify_intent,
    load_hermes_a2a_config,
    run_agentmart,
)
from logging_config import setup_logging

logger = logging.getLogger("agentmart.a2a")

PROTOCOL_VERSION = "1.0"
STATE_COMPLETED = "TASK_STATE_COMPLETED"
STATE_FAILED = "TASK_STATE_FAILED"
ROLE_AGENT = "ROLE_AGENT"

# JSON-RPC / A2A error codes, matching the peer's expectations.
ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603
ERR_UNAUTHORIZED = -32050

# ``SendMessage`` is the v1.0 method name; the pre-1.0 path style still shows up in the wild.
SEND_METHODS = {"SendMessage", "message/send", "tasks/send"}

# Runtime options, set once from the CLI so the handler can read them.
OPTIONS: dict[str, Any] = {
    "dry_run": False,
    "customer_id": DEFAULT_CUSTOMER_ID,
    "channel": "a2a",
    "token": "",
    "show_hops": True,
    "config_path": None,
}


def now_iso() -> str:
    """ISO 8601 UTC with millisecond precision, as A2A v1.0 specifies."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def text_part(text: str) -> dict:
    """A v1.0 text Part -- discriminated by member presence, so no ``kind`` field."""
    return {"text": text, "mediaType": "text/plain"}


def text_message(text: str, context_id: str) -> dict:
    msg: dict[str, Any] = {
        "role": ROLE_AGENT,
        "parts": [text_part(text)],
        "messageId": uuid.uuid4().hex,
    }
    if context_id:
        msg["contextId"] = context_id
    return msg


def build_task(task_id: str, context_id: str, state: str, agent_text: str) -> dict:
    """A v1.0 Task. Artifacts carry the final output; the status message mirrors it."""
    task: dict[str, Any] = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": state, "timestamp": now_iso()},
    }
    if agent_text:
        task["status"]["message"] = text_message(agent_text, context_id)
        if state == STATE_COMPLETED:
            task["artifacts"] = [
                {"artifactId": uuid.uuid4().hex, "parts": [text_part(agent_text)]}
            ]
    return task


def extract_text(params: dict) -> str:
    """Concatenated text from the inbound Message. v1.0, v0.3 (``kind``) and pre-0.3
    (``type``) Parts all carry ``text``, so reading that one key covers every peer."""
    msg = params.get("message") or params
    parts = msg.get("parts", []) if isinstance(msg, dict) else []
    chunks = [p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return "\n".join(chunks).strip()


def extract_context_id(params: dict) -> str:
    """v1.0 puts contextId inside the Message; tolerate a legacy top-level one."""
    msg = params.get("message") or {}
    inner = str(msg.get("contextId") or "") if isinstance(msg, dict) else ""
    return inner or str(params.get("contextId") or "")


def build_agent_card(public_url: str) -> dict:
    """Advertise every AgentMart agent's capabilities as A2A skills."""
    config = load_hermes_a2a_config(OPTIONS["config_path"])
    agentmart = config.get("agentmart", {})
    skills = []
    for agent in agentmart.get("agents", []):
        for capability in agent.get("capabilities", []):
            skills.append(
                {
                    "id": capability,
                    "name": capability.replace("_", " ").title(),
                    "description": f"{agent['id']} -- {capability.replace('_', ' ')}",
                    "tags": [agent["id"]],
                }
            )
    card: dict[str, Any] = {
        "name": agentmart.get("display_name", "AgentMart Agent Ecosystem"),
        "description": (
            "AgentMart's LangGraph agent group: shopping, pricing, inventory, fulfillment, "
            "order and a simulated payment agent. Send a customer request in plain text; "
            "the graph routes it by intent and returns the order agent's answer. "
            "Payments are simulated and contact no payment processor."
        ),
        "url": public_url,
        "version": "1.0.0",
        "provider": {"organization": "AgentMart Workshop", "url": public_url},
        "supportedInterfaces": [
            {
                "url": public_url,
                "protocolBinding": "JSONRPC",
                "protocolVersion": PROTOCOL_VERSION,
            }
        ],
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
            "extendedAgentCard": False,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": skills,
    }
    if OPTIONS["token"]:
        card["securitySchemes"] = {"bearer": {"type": "http", "scheme": "bearer"}}
        card["security"] = [{"bearer": []}]
    return card


def render_reply(state: dict) -> str:
    """The ecosystem's answer as prose. The CLI dumps raw state; a peer agent wants text.

    The last transcript entry is the graph's final word (order or payment agent). The hop
    trail is appended because seeing which agents woke is the point of the lab.
    """
    transcript = state.get("transcript") or []
    if not transcript:
        return "(AgentMart returned no transcript.)"
    final = transcript[-1]
    reply = str(final.get("message") or "").strip() or "(no message)"
    if not OPTIONS["show_hops"]:
        return reply
    hops = [entry.get("agent", "?") for entry in transcript]
    intent = state.get("intent", "?")
    trail = " -> ".join(hops)
    return f"{reply}\n\n---\n[intent: {intent}]\n[A2A hops: {trail}]"


def run_task(message: str) -> str:
    intent = classify_intent(message)
    logger.info("A2A task received: intent=%s text=%r", intent, message[:120])
    state = run_agentmart(
        message,
        channel=OPTIONS["channel"],
        dry_run=OPTIONS["dry_run"],
        config_path=OPTIONS["config_path"],
        customer_id=OPTIONS["customer_id"],
        intent=intent,
    )
    reply = render_reply(state)
    logger.info("A2A task done: %d hop(s)", len(state.get("transcript") or []))
    return reply


class A2AHandler(BaseHTTPRequestHandler):
    server_version = "AgentMartA2A/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # -- helpers ---------------------------------------------------------------

    def _public_url(self) -> str:
        if override := os.getenv("AGENTMART_A2A_PUBLIC_URL"):
            return override.rstrip("/")
        host = self.headers.get("Host") or f"127.0.0.1:{self.server.server_address[1]}"
        return f"http://{host}"

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _rpc_error(self, req_id: Any, code: int, message: str, status: int = 200) -> None:
        self._send_json(
            {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}},
            status=status,
        )

    def _authorized(self) -> bool:
        token = OPTIONS["token"]
        if not token:
            return True
        header = self.headers.get("Authorization", "")
        return header.startswith("Bearer ") and header[7:].strip() == token

    # -- routes ----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/.well-known/agent-card.json", "/.well-known/agent.json"):
            if not self._authorized():
                self._send_json({"error": "unauthorized"}, status=401)
                return
            self._send_json(build_agent_card(self._public_url()))
            return
        if path == "/health":
            self._send_json({"status": "ok", "dry_run": OPTIONS["dry_run"]})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            request = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._rpc_error(None, ERR_PARSE, "Parse error: body is not valid JSON.")
            return
        if not isinstance(request, dict):
            self._rpc_error(None, ERR_INVALID_REQUEST, "Invalid request: expected a JSON object.")
            return

        req_id = request.get("id")
        if not self._authorized():
            self._rpc_error(req_id, ERR_UNAUTHORIZED, "Unauthorized: bearer token required.", status=401)
            return

        method = str(request.get("method") or "")
        if method not in SEND_METHODS:
            self._rpc_error(req_id, ERR_METHOD_NOT_FOUND, f"Method not found: {method!r}.")
            return

        params = request.get("params") or {}
        if not isinstance(params, dict):
            self._rpc_error(req_id, ERR_INVALID_PARAMS, "Invalid params: expected an object.")
            return

        message = extract_text(params)
        if not message:
            self._rpc_error(req_id, ERR_INVALID_PARAMS, "Invalid params: no text part in message.")
            return

        context_id = extract_context_id(params) or "ctx-" + uuid.uuid4().hex[:16]
        task_id = str(req_id or uuid.uuid4().hex)
        try:
            reply = run_task(message)
        except Exception as exc:  # a graph failure is a failed task, not a dead server
            logger.exception("AgentMart task failed")
            task = build_task(task_id, context_id, STATE_FAILED, f"AgentMart failed: {exc}")
            self._send_json({"jsonrpc": "2.0", "id": req_id, "result": {"task": task}})
            return

        task = build_task(task_id, context_id, STATE_COMPLETED, reply)
        self._send_json({"jsonrpc": "2.0", "id": req_id, "result": {"task": task}})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the AgentMart LangGraph ecosystem over the A2A v1.0 protocol."
    )
    parser.add_argument("--host", default=os.getenv("AGENTMART_A2A_HOST", "127.0.0.1"),
                        help="Bind host (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=int(os.getenv("AGENTMART_A2A_PORT", "9901")),
                        help="Bind port (default 9901; Hermes keeps 9900 for its own inbound).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the graph without calling OpenRouter.")
    parser.add_argument("--customer", default=DEFAULT_CUSTOMER_ID,
                        help="Customer id the remote agent shops on behalf of.")
    parser.add_argument("--channel", default="a2a", help="Channel name recorded in the transcript.")
    parser.add_argument("--config", help="Path to Hermes A2A configuration JSON.")
    parser.add_argument("--no-hops", action="store_true",
                        help="Return only the final answer, without the A2A hop trail.")
    parser.add_argument("--verbose", action="store_true", help="Debug logging.")
    parser.add_argument(
        "--log-level",
        default=os.getenv("AGENTMART_LOG_LEVEL", "INFO"),
        help="Console log level for the agentmart logger (DEBUG/INFO/WARNING). "
             "The rotating file always records DEBUG. (default: INFO)",
    )
    args = parser.parse_args()

    # Debug level for the console only when requested; file always gets everything.
    console_level = "DEBUG" if args.verbose else str(args.log_level).upper()
    setup_logging(console_level)

    OPTIONS.update(
        dry_run=args.dry_run,
        customer_id=args.customer,
        channel=args.channel,
        token=os.getenv("AGENTMART_A2A_TOKEN", ""),
        show_hops=not args.no_hops,
        config_path=args.config,
    )

    if args.host != "127.0.0.1" and not OPTIONS["token"]:
        print(
            "Refusing to bind a non-local host without a token. "
            "Set AGENTMART_A2A_TOKEN, or bind 127.0.0.1.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    server = ThreadingHTTPServer((args.host, args.port), A2AHandler)
    base = f"http://{args.host}:{args.port}"
    logger.info("AgentMart A2A server on %s", base)
    logger.info("Agent Card: %s/.well-known/agent-card.json", base)
    logger.info("Auth: %s | dry-run: %s", "bearer token" if OPTIONS["token"] else "none (localhost)", args.dry_run)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
