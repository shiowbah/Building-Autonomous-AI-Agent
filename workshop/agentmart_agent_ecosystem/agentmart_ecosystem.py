from __future__ import annotations

import argparse
import json
import logging
import operator
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from catalog import CatalogNotSeededError, format_product_listing, query_products
from logging_config import bind_agent, setup_logging
from orders import (
    OrderBookNotSeededError,
    OrderNotFoundError,
    checkout_and_pay,
    create_draft_order,
    find_open_draft,
    find_payable_order,
    format_order,
    format_orders,
    get_customer,
    get_order,
    list_orders,
    missing_delivery_slots,
    refund_order,
    update_draft_delivery,
)

# A2A server requests the graph in its own process. Setting up logging at import
# time means every entry point (CLI + a2a_server + test_scenarios) writes to the
# same rotating logs/agentmart.log file from the same configuration.
setup_logging()
log = logging.getLogger("agentmart")


AgentName = Literal[
    "hermes_myshopper",
    "shopping_agent",
    "pricing_agent",
    "inventory_agent",
    "fulfillment_agent",
    "order_agent",
    "payment_agent",
]

# What the customer actually wants. The router is deterministic on purpose:
# the teaching point in Part 7 is capability + heartbeat routing, not intent
# classification, and reproducible routing keeps the scenario suite assertable.
Intent = Literal[
    "browse_catalog",
    "product_advice",
    "purchase_intent",
    "checkout_payment",
    "order_status",
]

DEFAULT_CUSTOMER_ID = "CUST-1001"


class AgentMartState(TypedDict, total=False):
    customer_request: str
    channel: str
    dry_run: bool
    customer_id: str
    intent: Intent
    target_sku: str | None
    target_order_id: str | None
    hermes_a2a_config: dict[str, Any]
    a2a_task: dict[str, Any]
    # Reducers: the parallel agents all append here in the same superstep, so
    # LangGraph needs to be told how to merge their writes instead of rejecting them.
    a2a_log: Annotated[list[dict[str, Any]], operator.add]
    product_listing: str
    order_context: str
    shopping_result: str
    pricing_result: str
    inventory_result: str
    fulfillment_result: str
    order_result: str
    payment_result: str
    draft_order: dict[str, Any]
    payment_receipt: dict[str, Any]
    delivery_details: dict[str, str]
    delivery_updated_slots: list[str]
    missing_delivery_slots: list[str]
    checkout_blocked_reason: str
    intent_only_recorded: bool
    transcript: Annotated[list[dict[str, Any]], operator.add]


@dataclass
class A2AEnvelope:
    """One hop on the A2A stream.

    `correlation_id` ties every hop of a single customer request together, and
    `state` mirrors the task lifecycle from Part 7 of the lecture, so a consumer
    can rebuild the whole conversation by replaying the log.
    """

    task_id: str
    sender: AgentName
    recipient: str
    intent: str
    payload: dict[str, Any]
    correlation_id: str = ""
    state: str = "proposed"
    protocol: str = "agentmart.a2a.v1"
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data["created_at"]:
            data["created_at"] = datetime.now(timezone.utc).isoformat()
        if not data["correlation_id"]:
            data["correlation_id"] = data["task_id"]
        return data


A2A_LIFECYCLE = ("proposed", "accepted", "in_progress", "completed", "failed")


DEFAULT_MODEL = "moonshotai/kimi-k3"

# Cumulative token usage across every model interaction, keyed by agent name.
# complete() feeds it; run_agentmart snapshots a baseline before invoking and
# diffs afterwards, so parallel A2A requests never corrupt a run's totals.
RUN_USAGE: dict[str, dict[str, int]] = {}


class OpenRouterHermesClient:
    """OpenAI-compatible client for OpenRouter, configured for Hermes/MyShopper.

    Settings resolve in this order, first match wins:
      1. environment variables / .env  (OPENROUTER_MODEL, ...)
      2. the model block in hermes_a2a_config.json
      3. the built-in defaults
    """

    def __init__(self, dry_run: bool = False, model_config: dict[str, Any] | None = None) -> None:
        load_dotenv()
        # The API key is deliberately NOT duplicated into the lab .env. It lives
        # only in Hermes' own env file (~/.hermes/.env), which the gateway already
        # loads; pull it in here so the workshop clients and A2A server resolve the
        # same single key instead of carrying a second plaintext copy. Values that
        # the lab DOES set (base_url, model, limits) are never clobbered: both
        # loads use override=False, so the first value already in the environment wins.
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
            hermes_dotenv = os.path.expanduser("~/.hermes/.env")
            if os.path.isfile(hermes_dotenv):
                load_dotenv(hermes_dotenv, override=False)
        config = model_config or {}
        self.dry_run = dry_run
        # OPENAI_* wins when set, so pointing the lab at OpenAI directly needs no rename.
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = (os.getenv("OPENAI_BASE_URL") or
                         os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
        self.model = (os.getenv("OPENAI_MODEL") or os.getenv("OPENROUTER_MODEL")
                      or config.get("default_model") or DEFAULT_MODEL)
        # OpenAI's own endpoint and OpenRouter disagree on three parameters, so the
        # request has to be shaped per endpoint rather than sent one way and hoped for.
        self.openai_native = "api.openai.com" in self.base_url
        self.fallback_models = config.get("fallback_models", [])
        self.temperature = float(os.getenv("OPENROUTER_TEMPERATURE", config.get("temperature", 0.2)))
        self.max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", config.get("max_tokens", 1200)))
        # Kimi K3 reasons before it answers, and reasoning is billed and timed like any
        # other completion token. Capping the effort is the single biggest latency win;
        # set OPENROUTER_REASONING_EFFORT=default to hand the model its full budget back.
        self.reasoning_effort = os.getenv(
            "OPENROUTER_REASONING_EFFORT", config.get("reasoning_effort", "medium")
        )
        self.http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
        self.app_title = os.getenv("OPENROUTER_APP_TITLE", "AgentMart Workshop")

        # Per-interaction usage and elapsed time, captured from the API response
        # for every model call. `last_usage`/`last_elapsed` describe the most
        # recent interaction; `usage_totals` accumulates across this client's
        # lifetime so a run can report what the whole chain cost.
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.last_elapsed: float = 0.0
        self.usage_totals: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0,
        }

    def complete(self, agent_name: str, system_prompt: str, user_prompt: str) -> str:
        started = time.monotonic()
        if self.dry_run or not self.api_key:
            self.last_elapsed = time.monotonic() - started
            self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            return self._dry_run_reply(agent_name, user_prompt)

        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}

        if self.openai_native:
            # The gpt-5.6 family rejects `max_tokens` (wants `max_completion_tokens`),
            # rejects any temperature but the default, and takes reasoning as a
            # top-level `reasoning_effort` rather than OpenRouter's `reasoning` object.
            # Model fallbacks and the attribution headers are OpenRouter features.
            kwargs["max_completion_tokens"] = self.max_tokens
            if self.reasoning_effort and self.reasoning_effort != "default":
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            extra_body: dict[str, Any] = {}
            if self.fallback_models:
                # OpenRouter retries these in order if the primary model is unavailable.
                extra_body["models"] = [self.model, *self.fallback_models]
            if self.reasoning_effort and self.reasoning_effort != "default":
                extra_body["reasoning"] = {"effort": self.reasoning_effort}
            kwargs.update(
                extra_headers={"HTTP-Referer": self.http_referer, "X-Title": self.app_title},
                extra_body=extra_body,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        response = client.chat.completions.create(**kwargs)
        self.last_elapsed = time.monotonic() - started

        # Token usage comes back under `usage` on OpenAI-compatible responses and
        # under `usage` from OpenRouter too; guard the shape defensively so an
        # unexpected provider cannot crash the graph.
        usage = getattr(response, "usage", None)
        self.last_usage = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        } if usage is not None else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for key, value in self.last_usage.items():
            self.usage_totals[key] += value
        self.usage_totals["calls"] += 1
        agent_totals = RUN_USAGE.setdefault(
            agent_name, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            agent_totals[key] += self.last_usage[key]
        agent_totals["calls"] += 1
        log.debug(
            "%s: tokens prompt=%s completion=%s total=%s elapsed=%.2fs",
            agent_name,
            self.last_usage["prompt_tokens"],
            self.last_usage["completion_tokens"],
            self.last_usage["total_tokens"],
            self.last_elapsed,
        )
        return response.choices[0].message.content or ""

    def usage_line(self) -> str:
        """One-line summary of the most recent interaction, for log evidence."""
        return (
            f"tokens={self.last_usage['total_tokens']} "
            f"(prompt {self.last_usage['prompt_tokens']} + completion {self.last_usage['completion_tokens']}) "
            f"elapsed={self.last_elapsed:.2f}s"
        )

    @staticmethod
    def _dry_run_reply(agent_name: str, user_prompt: str) -> str:
        compact_prompt = " ".join(user_prompt.split())
        return f"[dry-run:{agent_name}] {compact_prompt[:260]}"


def load_product_listing(
    category: str | None = None,
    max_price: float | None = None,
    limit: int = 25,
) -> str:
    """Render the seeded AgentMart product listing for the agent prompts."""
    try:
        products = query_products(category=category, max_price=max_price, limit=limit)
    except CatalogNotSeededError as exc:
        return f"(catalog unavailable: {exc})"
    if not products:
        return "(no products in the seeded catalog matched the filters)"
    return format_product_listing(products)


SKU_PATTERN = re.compile(r"\bAM-[A-Z]{3}-\d{4}\b", re.IGNORECASE)
ORDER_ID_PATTERN = re.compile(r"\bAM-ORD-[\w-]+\b", re.IGNORECASE)

# Checked in order: the first rule that matches wins. Order matters here —
# "checkout and pay for my order" contains "my order", so the explicit checkout
# imperative has to outrank the status rule, while a *question* about payment
# ("has my payment gone through?") must stay a status lookup and never charge.
INTENT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        # 0. An imperative to *put values on an order* is an update even when it
        # quotes an earlier read-only result ("The read-only lookup shows ... are
        # still not stored. Please update ... with exactly: ..."). The update verb
        # plus a concrete "with ..." values clause must outrank the read-only marker,
        # or the correction gets downgraded to a no-op lookup. A literal status
        # question ("has anything changed with my order?") carries no such clause.
        "purchase_intent",
        re.compile(
            r"(?:updat\w*|correct\w*|persist\w*|fix\w*|sav\w*|chang\w*|refresh\w*|revis\w*|edit\w*)\b"
            r"[^.?]{0,200}\bwith\s+(?:exactly|these|those|the\s+following|recipient|delivery|address)\b|"
            r"(?:updat\w*|correct\w*|persist\w*|fix\w*|sav\w*|chang\w*|refresh\w*|revis\w*|edit\w*)\b"
            r"[^.?]{0,120}\bdelivery\s+(?:details|address)\b|"
            r"(?:updat\w*|correct\w*|persist\w*|fix\w*|sav\w*|chang\w*|refresh\w*|revis\w*|edit\w*)\b"
            r"[^.?]{0,200}\bwith\s+details?\s*:|"
            r"(?:updat\w*|correct\w*|persist\w*|fix\w*|sav\w*|chang\w*|refresh\w*|revis\w*|edit\w*)\b"
            r"[^.?]{0,160}\b(?:these|those)\s+(?:exact\s+)?(?:missing\s+)?fields?\s*:",
            re.IGNORECASE,
        ),
    ),
    (
        # 1. An explicit settle/pay imperative is CHECKOUT even when the message
        # also asks to read back the result afterwards ("...then perform an
        # order-status lookup"). That trailing verification clause must NOT
        # downgrade a charge to a no-op lookup, or Hermes could say "proceed to
        # checkout and payment" yet never wake the Payment Agent. Guardrail noun
        # phrases like "checkout draft/step" stay drafting, and a bare mention
        # of "checkout." as a noun must not settle anything.
        "checkout_payment",
        re.compile(
            r"\bcheck\s?out\s+and\s+pay(?:ment)?\b|"
            r"\bpay\s+(?:for|now|to)\b|"
            r"\bplace\s+the\s+order\b|"
            r"\bsettle\s+(?:the\s+)?(?:order|bill|up)\b|"
            r"\bproceed\s+(?:with|to)\s+(?:the\s+)?(?:checkout|payment)\b|"
            r"\battempt\s+(?:the\s+)?(?:secure\s+)?(?:payment\s+)?authorization\b|"
            r"\bpay\s+(?:for\s+)?(?:my|this|the)\s+order\b",
            re.IGNORECASE,
        ),
    ),
    (
        # 2. Read-only/verification phrasings never settle anything. Remote peers
        # state their own guardrails inline ("Read-only verification only"), so the
        # markers here mean "lookup", not "charge". A real checkout that happens to
        # verify the cart first is NOT matched (no read-only marker in it), so it
        # still lands on the payment path. A *lookup* phrased as "verify … read
        # back / by looking it up / current status" is the same read-only intent:
        # routing it as a purchase_intent would re-persist junk slots parsed from
        # phrases like "after the address update".
        "order_status",
        re.compile(
            r"read[\s-]?only\b|"
            r"verif\w*\s+only\b|"
            r"\breconcil(?:iation|ing)?\b|"
            r"\baudit\b|"
            r"confirm\s+(?:whether|if)\s+no\b|"
            r"\bverif\w*\b.{0,220}\b(?:read\w*\s*back|by\s+looking\s+it\s+up|current\s+status|order\s+status)\b|"
            r"\b(?:read\w*\s*back|look\s+it\s+up|lookup)\b.{0,220}\b(?:stored|status|order\b|draft\b)",
            re.IGNORECASE,
        ),
    ),
    (
        # 2. An explicit request to *draft* an order outranks every payment word
        # that may trail it ("...just confirm the draft and the next checkout step").
        # The verb list and middle are deliberately loose: remote agents say "create
        # a payable checkout draft", "prepare a checkout draft", "make an order
        # draft", "read/write order draft only", "update/refresh the checkout draft".
        # The word "checkout" alone must NOT win here — "prepare a checkout draft"
        # is drafting, not settling. A bare *creation* request ("create a payable
        # order") is the same capability invite and must also land on the drafting
        # path, never the charging one.
"purchase_intent",
        re.compile(
            r"(?:creat\w*|prepar\w*|mak\w*|writ\w*|set\s*up)\s+(?:a\s+)?(?:new\s+)?"
            r"(?:payable\s+)?(?:checkout\s+|order\s+)?(?:draft|order)\b|"
            r"(?:creat\w*|prepar\w*|mak\w*|writ\w*|set\s*up|updat\w*|refresh\w*|correct\w*|persist\w*|fix\w*|sav\w*|chang\w*)\b[^.]{0,80}\bdraft\b|"
            r"(?:updat\w*|correct\w*|persist\w*|fix\w*|sav\w*|chang\w*|refresh\w*|revis\w*|edit\w*)\b[^.?]{0,100}"
            r"\b(?:with\s+(?:exactly|these|those|the\s+following|the\s+new|delivery)|"
            r"delivery\s+(?:details|address)|recipient|postal|contact\s+number)\b|"
            r"(?:draft|checkout)\s+order\b",
            re.IGNORECASE,
        ),
    ),
    (
        # 2. Unambiguous checkout imperatives. "checkout draft" is a noun phrase,
        # never a settle-and-pay instruction.
        "checkout_payment",
        re.compile(
            r"check\s?out\b(?!\s*(?:step|steps|process|flow|page|link|option|details|draft))|"
            r"\bpay\s+now\b|place\s+the\s+order|settle\s+(up|the\s+bill)",
            re.IGNORECASE,
        ),
    ),
    (
        # 3. Status questions, including questions *about* a payment.
        "order_status",
        re.compile(
            r"order\s+status|order\s+summary|order[_\s]status[_\s]lookup|"
            r"status\s+of\s+(my|the|order)|where\s+is\s+my|"
            r"track(ing)?\b|my\s+orders?\b|delivery\s+status|has\s+it\s+shipped|"
            r"payment\s+.*(gone\s+through|received|cleared|succeed)",
            re.IGNORECASE,
        ),
    ),
    (
        # 4. Weaker payment wording, only once a status reading is ruled out.
        "checkout_payment",
        re.compile(r"\bpay\b|\bpaying\b|payment", re.IGNORECASE),
    ),
    (
        "purchase_intent",
        re.compile(
            r"(?:want|wants|wish|would\s+like|going)\s+to\s+buy|i.?ll\s+buy|"
            r"\bbuy\s+(?:this|it|the|sku)\b|purchase\s+(?:this|it|the|sku)|"
            r"add\s+to\s+(the\s+)?cart|i.?ll\s+take|take\s+(it|this)|order\s+(this|the)",
            re.IGNORECASE,
        ),
    ),
    (
        "browse_catalog",
        re.compile(
            r"list\s+(me|the|all|available)|show\s+me|what\s+(do\s+you\s+have|is\s+available|"
            r"products?\s+are)|browse|catalog(ue)?|available\s+product",
            re.IGNORECASE,
        ),
    ),
    (
        # 5. A bare delivery tuple with no verb at all ("David Neo, 25 Heng Mui
        # Keng Terrace Singapore, 119615, 65162093") is the customer supplying the
        # slots, so record it on the draft. Kept last: every explicit checkout,
        # status, draft, buy or browse marker above wins first, so a checkout
        # request that also carries values is still a checkout.
        "purchase_intent",
        re.compile(
            r"[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,2}\s*,\s*"
            r"\d{1,6}\s+[A-Za-z][A-Za-z]{3,}(?:[ -][A-Za-z0-9']+)*?\s*,\s*"
            r"\d{6}\s*,\s*[2-9]\d{7}\b",
            re.IGNORECASE,
        ),
    ),
)


# A remote agent states its guardrails in the request itself -- "do not charge or
# capture payment", "without taking any refund action". Those clauses name the very
# capability they are forbidding, so matching them verbatim routes a prohibition to
# the Payment Agent. Drop each one up to its clause boundary before any rule runs.
PROHIBITION_CLAUSE = re.compile(
    r"\b(?:do\s+not|do\s?n['\u2019]t|does\s+not|never|without|avoid|no\s+need\s+to)\b[^.;:()\n]*",
    re.IGNORECASE,
)


def strip_prohibitions(text: str) -> str:
    """Remove negated clauses so a forbidden action cannot be read as a requested one."""
    return PROHIBITION_CLAUSE.sub(" ", text)


# Read-only markers. The Payment Agent treats any request with one of these as a
# lookup, never a charge, even if the request happens to mention a payment id.
READ_ONLY_MARKER = re.compile(
    r"read[\s-]?only\b|"
    r"verif\w*\s+only\b|"
    r"\breconcil(?:iation|ing)?\b|"
    r"\baudit\b|"
    r"confirm\s+(?:whether|if)\s+no\b",
    re.IGNORECASE,
)

# What a genuine settle-and-pay request actually says. "prepare/update a checkout
# draft" is cared for by routing and must never satisfy this test.
PAYMENT_IMPERATIVE = re.compile(
    r"check\s?-?\s?out\b(?!\s*(?:draft|step|summary|details|flow|process|page|"
    r"link|option|review|confirmation|required|below))\b|"
    r"\bproceed(?:ing)?\s+with\s+checkout\b|"
    r"\bpay\s+(?:now|for|the\s+order|this|it|my\s+order)\b|"
    r"\bplace\s+the\s+order\b|"
    r"\bsettle\b|"
    r"\bsubmit\s+(?:the\s+)?(?:payment|charge)\b",
    re.IGNORECASE,
)

# A request that records a purchase intent WITHOUT drafting. "Record the purchase
# intent and prepare a checkout draft" is the opposite and must not match.
INTENT_ONLY_RECORD = re.compile(
    r"\bpurchase\s+intent\s+only\b|"
    r"record\w*\s+(?:this|the)\s+.{0,20}\bintent\b|"
    r"record(?:ing)?\s+.*\bintent\b.{0,40}\bonly\b",
    re.IGNORECASE,
)
INTENT_ONLY_BLOCK = re.compile(
    r"(?:do\s+not|do\s?n['\u2019]t|never|without|avoid)\b[^.;:\n]{0,60}"
    r"\b(?:create|open|place|make|modify|reserve|touch)\b[^.;:\n]{0,50}\border\b|"
    r"\b(?:order\s+book|any\s+order|prior\s+order)\b.{0,40}\b(?:do\s+not|never|without)\b",
    re.IGNORECASE,
)
INTENT_ONLY_DRAFT = re.compile(
    r"(?:creat\w*|prepar\w*|mak\w*|writ\w*|set\s*up|updat\w*|refresh\w*|correct\w*|persist\w*|fix\w*|sav\w*|chang\w*)\b[^.]{0,80}\bdraft\b",
    re.IGNORECASE,
)


def is_read_only_request(text: str) -> bool:
    """True when the request is a verification/lookup, never an action."""
    return bool(READ_ONLY_MARKER.search(text or ""))


def _explicit_payment_imperative(text: str) -> bool:
    """True when the request really asks to settle. Guardrails are dropped so that
    'do not pay' cannot satisfy the pay imperative."""
    return bool(PAYMENT_IMPERATIVE.search(strip_prohibitions(text or "")))


def intent_only_request(text: str) -> bool:
    """True when the peer wants intent recorded but explicitly NOT an order.

    Requires all three: an intent-only phrasing, a prohibition on creating/modifying
    an order, and no draft-creation request. 'Record the purchase intent and prepare
    a checkout draft' fails the last test and still drafts.
    """
    text = text or ""
    return bool(
        INTENT_ONLY_RECORD.search(text)
        and INTENT_ONLY_BLOCK.search(text)
        and not INTENT_ONLY_DRAFT.search(text)
    )


def classify_intent(customer_request: str) -> Intent:
    """Deterministic intent routing.

    Kept rule-based so a scenario run is reproducible and the assertions in
    `test_scenarios.py` mean something: the agents are the model-driven part,
    the routing is not.
    """
    customer_request = strip_prohibitions(customer_request)
    for intent, pattern in INTENT_RULES:
        if pattern.search(customer_request):
            return intent  # type: ignore[return-value]
    # A request naming an existing order is about that order, whatever else it says.
    # Falling through to product_advice here would wake Shopping and Pricing to answer
    # a question about an order already in the book.
    if extract_order_id(customer_request):
        return "order_status"
    return "product_advice"


def extract_sku(customer_request: str) -> str | None:
    match = SKU_PATTERN.search(customer_request)
    return match.group(0).upper() if match else None


def extract_order_id(customer_request: str) -> str | None:
    match = ORDER_ID_PATTERN.search(customer_request)
    return match.group(0).upper() if match else None


# Delivery slots a checkout draft should collect before it is paid. The Order
# Agent may still confirm a draft while some are missing, but it must NOT call
# the Payment Agent short of a full set -- it has to ask for the rest instead.
REQUIRED_DELIVERY_SLOTS = ("recipient", "address", "postal", "contact")

DELIVERY_SLOT_LABELS = {
    "recipient": "recipient name",
    "address": "full delivery address",
    "postal": "postal code",
    "contact": "contact number",
}


def extract_delivery_details(text: str) -> dict[str, str]:
    """Pull the customer-supplied delivery slots out of a request.

    Remote peers phrase these loosely ("address line '3 Pine Grove'", "recipient
    Ang Chin Tiong", "contact number 97492736", "postal code 597590"), so each
    slot is captured from a keyword up to the next comma, then trimmed of the
    surrounding quotes the peer used.
    """
    details: dict[str, str] = {}

    # A positional list introduces the slots in the order the Order Agent asks for
    # them ("update ... with exactly: ang chin tiong; 3 pine grove singapore; 597590;
    # 97492736"). Parse it first: keyword matching cannot, because the request often
    # *quotes* the earlier read-back ("recipient name, postal code, and contact number
    # are still not stored") and the quote is loaded with junk values ("postal code").
    positional = re.search(
        r"\bwith\s+exactly\s*:\s*(.+?)(?:\.\s|$)",
        text,
        re.IGNORECASE,
    )
    if positional:
        tokens = [t.strip().strip("'\" ") for t in positional.group(1).split(";") if t.strip()]
        postal = next((t for t in tokens if re.fullmatch(r"\d{6}", t)), "")
        contact = next((t for t in tokens if re.fullmatch(r"[2-9]\d{7}", t)), "")
        address = next(
            (t for t in tokens if t not in (postal, contact) and re.search(r"\d", t) and len(t) > 3),
            "",
        )
        remaining = [t for t in tokens if t not in (postal, contact, address)]
        if remaining:
            details["recipient"] = remaining[0]
        if address:
            details["address"] = address
        if postal:
            details["postal"] = postal
        if contact:
            details["contact"] = contact
        return details

    # A labelled "save these exact missing fields: ..." list. The peer's own
    # readback quote ahead of it is loaded with junk ("recipient name, postal
    # code, and contact number are missing"), so keyword matching on the whole
    # request would persist "postal code" as the recipient. The fields list is
    # authoritative: parse the semicolon-separated label/value pairs it names.
    fields_list = re.search(
        r"\b(?:these|those)\s+(?:exact\s+)?(?:missing\s+)?fields?\s*:\s*(.+?)(?:\.\s|$)",
        text,
        re.IGNORECASE,
    )
    if fields_list:
        for token in fields_list.group(1).split(";"):
            token = token.strip().strip("'\" ")
            if not token:
                continue
            slot = next(
                (
                    s
                    for s in ("recipient", "postal", "contact")
                    if re.match(rf"^{re.escape(DELIVERY_SLOT_LABELS[s])}\s*:?\s*", token, re.IGNORECASE)
                ),
                None,
            )
            if not slot:
                if re.match(r"^(?:full\s+)?(?:delivery\s+)?address\b", token, re.IGNORECASE):
                    slot = "address"
            if not slot:
                continue
            value = re.sub(
                rf"^{re.escape(DELIVERY_SLOT_LABELS[slot])}\s*:?\s*", "", token, flags=re.IGNORECASE
            ).strip().strip("'\" ")
            if slot == "postal":
                match = re.search(r"([0-9]{6})", value)
                value = match.group(1) if match else value
            elif slot == "contact":
                match = re.search(r"([2-9][0-9]{7})", value)
                value = match.group(1) if match else value
            elif slot == "recipient":
                value = re.sub(r"^\s*(?:full\s+)?name\s*:\s*", "", value, flags=re.IGNORECASE).strip()
            if not value:
                continue
            # A readback enumeration is a list of the *labels themselves* ("recipient
            # full name, complete delivery address, postal code, contact number, and
            # delivery method are still missing"), so each _after(label) capture lands
            # on the NEXT label, not on data. Persisting it stores a slot label as a
            # slot value (recipient => "complete delivery address", postal => "contact
            # number"), which is exactly the shift an earlier peer observed. Reject any
            # captured value that is itself one of the slot labels / label phrases.
            if any(
                re.search(rf"\b{re.escape(label)}\b", value, re.IGNORECASE)
                for label in (
                    *DELIVERY_SLOT_LABELS.values(),
                    "recipient full name",
                    "recipient name",
                    "full name",
                    "complete delivery address",
                    "full delivery address",
                    "delivery address",
                    "address",
                    "postal code",
                    "postal",
                    "postcode",
                    "contact number",
                    "contact",
                    "delivery method",
                )
            ):
                continue
            details[slot] = value
        return details

    def _after(keyword: str) -> str:
        match = re.search(
            rf"\b{keyword}\b[^A-Za-z0-9]{{0,8}}([A-Za-z0-9][^,;]*)",
            text,
            re.IGNORECASE,
        )
        if not match:
            return ""
        value = match.group(1).strip()
        value = re.split(r"\.\s+\w", value, maxsplit=1)[0]
        value = value.split(" and ")[0]
        value = re.sub(r"[^A-Za-z0-9\s'\"-]+$", "", value).strip().strip("'\" ").strip()
        if len(value) > 48:
            value = value[:48].rsplit(" ", 1)[0]
        # A labelled list the peer read back as still-missing ("...recipient full
        # name, complete delivery address, postal code, contact number, and
        # delivery method are missing from the draft") assigns each keyword's
        # _after() capture the NEXT slot label, not a value: recipient lands on
        # "complete delivery address", postal on "contact number", contact on
        # "delivery method". That is a label quote, not deliverable data, so
        # reject any capture whose value STARTS with a slot label or label
        # fragment ("full name", "code", "and delivery method ..."). Real values
        # never start with one: an address always begins with its house number
        # (already enforced down the loop), and recipient/postal/contact data
        # never begins with a label word. Let the caller's `if not raw: continue`
        # skip the slot entirely.
        if re.match(
            r"\s*(?:and\s+)?(?:recipient\b|(?:full\s+)?name\b|complete\s+delivery\s+"
            r"address\b|full\s+delivery\s+address\b|delivery\s+address\b|address\b|"
            r"postal\s+code\b|postal\b|postcode\b|code\b|contact\s+number\b|contact\b|"
            r"number\b|delivery\s+method\b|delivery\s+details\b|delivery\b|method\b|"
            r"details\b|details\b)",
            value,
            re.IGNORECASE,
        ):
            return ""
        return value

    for slot, keywords in (
        ("recipient", ("recipient full name", "recipient name", "recipient", "full name", "deliver to", "ship to", "send to")),
        ("address", ("delivery address", "address line", "shipping to", "send to", "address")),
        ("contact", ("contact number", "contact", "phone number", "phone")),
        ("postal", ("postal code", "postal", "postcode")),
    ):
        raw = ""
        for keyword in keywords:
            raw = _after(keyword)
            if raw:
                break
        if not raw:
            continue
        if slot == "recipient":
            # "recipient full name: ang chin tiong" (and "recipient name: x") are
            # labelled with the slot word itself -- drop that label so the persisted
            # value is the name, not "full name: ang chin tiong".
            value = re.sub(r"^\s*(?:full\s+)?name\s*:\s*", "", raw, flags=re.IGNORECASE).strip()
            details["recipient"] = value if value else raw
        elif slot == "contact":
            match = re.search(r"([2-9][0-9]{7})", raw)
            details[slot] = match.group(1) if match else raw
        elif slot == "postal":
            match = re.search(r"([0-9]{6})", raw)
            details[slot] = match.group(1) if match else raw
        elif slot == "address":
            value = raw
            # Strip leading filler the peer may have used ("delivery address
            # exactly: 3 pine grove" => "3 pine grove"), then drop junk values
            # that are really about the request itself ("after the address update",
            # "on draft AM-ORD-2026..."). A real street always leads with a house
            # number here, so require one before accepting the slot.
            value = re.sub(
                r"^(?:exactly|precisely|just|the|a|an|of|below|following|updated|new|current)\s*[:]?\s*",
                "",
                value,
                flags=re.IGNORECASE,
            )
            value = re.sub(r"[^A-Za-z0-9\s'\"-]+$", "", value).strip().strip("'\" ").strip()
            if len(value) > 48:
                value = value[:48].rsplit(" ", 1)[0]
            if re.search(r"\b(?:draft|update|updated|sku|order)\b", value, re.IGNORECASE):
                continue
            if not re.search(r"\d{1,6}\s+[A-Za-z]", value):
                continue
            details[slot] = value
            # A labelled "full delivery address 25 Heng Mui Keng Terrace,
            # Singapore" truncates at the comma even though the area is part of
            # the address. Re-attach a single capitalised area word that directly
            # follows the captured line and is closed by a slot separator.
            area = re.search(
                rf"\b{re.escape(value)}\s*,\s*([A-Z][A-Za-z'-]+)\b([;,]|(?:\s+(?:postal|contact|recipient)\b)|$)",
                text,
                re.IGNORECASE,
            )
            if area and len(value) + len(area.group(1)) + 2 <= 48:
                details["address"] = value + ", " + area.group(1)
        else:
            details[slot] = raw

    if "recipient" not in details:
        match = re.search(r"\bfor ([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)(?:\s+[A-Z][A-Za-z'-]+)?)\b", text)
        if match:
            details["recipient"] = match.group(1)
    if "recipient" not in details:
        # "update ... with exactly: ang chin tiong; 3 pine grove singapore; 597590; 97492736"
        # is an unlabelled positional list in the order the Order Agent asks for slots.
        match = re.search(r"\bwith\s+exactly\s*:\s*([^;.;]{2,64})\s*;", text, re.IGNORECASE)
        if match:
            details["recipient"] = match.group(1).strip().strip("'\" ") or details.get("recipient", "")
    if "contact" not in details:
        # Singleton numbers in a bare request; never take a digit-run that is
        # glued to an identifier ("AM-ORD-20260919-5E10" carries "-20260919").
        match = re.search(r"(?<![\w-])([2-9][0-9]{7})(?![\w-])", text)
        if match:
            details["contact"] = match.group(1)
    if "postal" not in details:
        match = re.search(r"\b([0-9]{6})\b", text)
        if match:
            details["postal"] = match.group(1)

    # Peers sometimes skip the slot labels entirely and just give an ordered list
    # ("Use delivery details: Ang Chin Tiong, 3 Pine Grove, Singapore 597590,
    # contact number 97492736"). Fall back to positional/structural patterns:
    # the recipient is whatever leads the "delivery details:" list, and the
    # address is the first house-numbered line in the request.
    if "recipient" not in details:
        match = re.search(
            r"delivery\s+details?\s*:\s*"
            r"(?!(?:postal|address|contact|phone|recipient|name|code)\b)"
            r"([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,2})",
            text,
            re.IGNORECASE,
        )
        if match:
            details["recipient"] = match.group(1).strip()
    if "address" not in details:
        # A house-numbered street line. The digits must start a fresh value: a
        # SKU/quantity readback like "1 x AM-EAR-1002 Nimbus Air 2" embeds
        # "1002" right after a hyphen, so "1002 Nimbus Air 2" must never be
        # captured as an address.
        match = re.search(
            r"(?<![\w-])(\d{1,6}\s+[A-Za-z][A-Za-z]{3,}(?:[ -][A-Za-z0-9']+)*?)(?:,|;|\s{2,}|$)",
            text,
            re.IGNORECASE,
        )
        if match:
            details["address"] = (
                match.group(1).strip().strip("'\" ").strip()
            )

    # A bare comma-separated ordered list in the order the Order Agent asks for
    # the slots ("David Neo, 25 Heng Mui Keng Terrace Singapore, 119615,
    # 65162093"). Fill anything the keyword passes above could not parse: the
    # name and the 6-start contact number are the usual losses.
    if not {"recipient", "address", "postal", "contact"}.issubset(details):
        tuple_match = re.search(
            r"([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,2})\s*,\s*"
            r"(\d{1,6}\s+[A-Za-z][A-Za-z]{3,}(?:[ -][A-Za-z0-9']+)*?)\s*,\s*"
            r"(\d{6})\s*,\s*([2-9]\d{7})\b",
            text,
        )
        if tuple_match:
            details.setdefault("recipient", tuple_match.group(1).strip())
            details.setdefault("address", tuple_match.group(2).strip().strip("'\" ").strip())
            details.setdefault("postal", tuple_match.group(3))
            details.setdefault("contact", tuple_match.group(4))

    return details


def load_order_context(customer_id: str, order_id: str | None = None) -> str:
    """Render the customer's order book for the Order and Payment agent prompts."""
    try:
        if order_id:
            return format_order(get_order(order_id))
        customer = get_customer(customer_id)
        header = (
            f"customer {customer['customer_id']} | {customer['name']}"
            f" | {customer['channel']} {customer['channel_handle']}"
            f" | ships to {customer['shipping_address']}"
            f" | default payment {customer['default_payment_method']}"
            if customer
            else f"(no customer record for {customer_id})"
        )
        return header + "\n" + format_orders(list_orders(customer_id=customer_id))
    except OrderNotFoundError as exc:
        return f"(order not found: {exc})"
    except OrderBookNotSeededError as exc:
        return f"(order book unavailable: {exc})"


def emit_envelope(
    state: AgentMartState,
    sender: AgentName,
    recipient: str,
    intent: str,
    payload: dict[str, Any],
    lifecycle: str = "in_progress",
) -> dict[str, Any]:
    """Build one A2A hop. Pure: parallel branches share a state snapshot, so the
    caller collects the returned envelopes and hands them back as an ``a2a_log`` delta."""
    task = state.get("a2a_task") or {}
    envelope = A2AEnvelope(
        task_id=str(uuid4()),
        sender=sender,
        recipient=recipient,
        intent=intent,
        payload=payload,
        correlation_id=task.get("correlation_id") or task.get("task_id") or str(uuid4()),
        state=lifecycle,
        protocol=task.get("protocol", "agentmart.a2a.v1"),
    ).to_dict()
    log.debug(
        "a2a hop %s -> %s intent=%s lifecycle=%s correlation=%s",
        sender,
        recipient,
        intent,
        lifecycle,
        envelope["correlation_id"],
    )
    return envelope


def load_hermes_a2a_config(config_path: str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else Path(__file__).with_name("hermes_a2a_config.json")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def model_config_from_state(state: AgentMartState) -> dict[str, Any]:
    """Read the Hermes model block (provider, model, temperature) from graph state."""
    config = state.get("hermes_a2a_config") or {}
    return config.get("hermes_agent", {}).get("model", {})


def transcript_entry(
    agent: AgentName,
    message: str,
    envelope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One transcript row. Nodes return ``{"transcript": [entry]}`` and the reducer
    concatenates -- returning the whole list would double it under a reducer."""
    return {"agent": agent, "message": message, "a2a": envelope}


def make_agent_node(
    agent: AgentName,
    system_prompt: str,
    prompt_builder: Callable[[AgentMartState], str],
    output_key: str,
    capability: str = "",
) -> Callable[[AgentMartState], AgentMartState]:
    def node(state: AgentMartState) -> AgentMartState:
        envelope = emit_envelope(
            state,
            sender="hermes_myshopper",
            recipient=agent,
            intent=capability or output_key,
            payload={"intent": state.get("intent"), "capability": capability or output_key},
            lifecycle="accepted",
        )
        client = OpenRouterHermesClient(
            dry_run=state.get("dry_run", False),
            model_config=model_config_from_state(state),
        )
        agent_log = bind_agent(agent)
        started = time.monotonic()
        log.info(
            "%s: starting (intent=%s, dry_run=%s, model=%s)",
            agent,
            state.get("intent"),
            state.get("dry_run", False),
            client.model,
        )
        result = client.complete(agent, system_prompt, prompt_builder(state))
        elapsed = time.monotonic() - started
        log.info(
            "%s: completed in %.2fs (%d chars) | %s",
            agent,
            elapsed,
            len(result),
            client.usage_line(),
        )
        agent_log.debug("reply: %s", result)
        done = emit_envelope(
            state,
            sender=agent,
            recipient="hermes_myshopper",
            intent=capability or output_key,
            payload={"result_chars": len(result)},
            lifecycle="completed",
        )
        return {
            "transcript": [transcript_entry(agent, result, envelope)],
            "a2a_log": [envelope, done],
            output_key: result,
        }

    return node


def hermes_myshopper_node(state: AgentMartState) -> AgentMartState:
    customer_request = state["customer_request"]
    config = state.get("hermes_a2a_config") or load_hermes_a2a_config()
    hermes_config = config["hermes_agent"]
    a2a_config = config["a2a_connection"]
    agentmart_config = config["agentmart"]

    intent = state.get("intent") or classify_intent(customer_request)
    customer_id = state.get("customer_id") or DEFAULT_CUSTOMER_ID
    task_id = str(uuid4())

    envelope = A2AEnvelope(
        task_id=task_id,
        correlation_id=task_id,
        state="proposed",
        sender=hermes_config["id"],
        recipient=agentmart_config["id"],
        intent=intent,
        payload={
            "customer_request": customer_request,
            "customer_id": customer_id,
            "channel": state.get("channel", "webchat"),
            "target_sku": state.get("target_sku") or extract_sku(customer_request),
            "target_order_id": state.get("target_order_id") or extract_order_id(customer_request),
            "hermes_agent": hermes_config,
            "a2a_connection": a2a_config,
            "target_agents": agentmart_config["agents"],
            "constraints": {
                "representing": "customer",
                "ecosystem": agentmart_config["display_name"],
            },
        },
        protocol=a2a_config["protocol"],
    ).to_dict()

    client = OpenRouterHermesClient(
        dry_run=state.get("dry_run", False),
        model_config=hermes_config.get("model", {}),
    )
    started = time.monotonic()
    log.info(
        "hermes_myshopper: routing '%.100s' -> intent=%s customer=%s task=%s model=%s",
        customer_request,
        intent,
        customer_id,
        task_id,
        client.model,
    )
    message = client.complete(
        "hermes_myshopper",
        (
            "You are Hermes/MyShopper, a personal buying agent. "
            "You represent the customer, not AgentMart. "
            "Create a short handoff note for the AgentMart agent ecosystem."
        ),
        json.dumps(envelope, indent=2),
    )
    log.info(
        "hermes_myshopper: handoff written in %.2fs | %s",
        time.monotonic() - started,
        client.usage_line(),
    )

    next_state: AgentMartState = {
        "transcript": [transcript_entry("hermes_myshopper", message, envelope)],
        "a2a_log": [envelope],
        "a2a_task": envelope,
        "intent": intent,
        "customer_id": customer_id,
        "target_sku": envelope["payload"]["target_sku"],
        "target_order_id": envelope["payload"]["target_order_id"],
    }
    # Order and Payment agents read the customer's real order book.
    if intent in ("order_status", "checkout_payment", "purchase_intent"):
        # A status lookup gets the FULL book (plus nails the named order) so a
        # peer that says "check order X" still can't hide the customer's other
        # orders, like a recent draft. Checkout/payment narrow to one order.
        order_context = load_order_context(
            customer_id,
            None if intent == "order_status" else envelope["payload"]["target_order_id"],
        )
        if intent == "order_status" and envelope["payload"]["target_order_id"]:
            requested = envelope["payload"]["target_order_id"]
            in_book = requested in order_context
            order_context += "\n(requested order id: " + requested + ")"
            if not in_book:
                order_context += "\n**Note: order " + requested + " does not appear in the customer's order book.**"
        next_state["order_context"] = order_context
    return next_state


shopping_agent_node = make_agent_node(
    "shopping_agent",
    (
        "You are the AgentMart Shopping Agent. Find candidate products that match the customer's intent. "
        "Only recommend SKUs that appear in the AgentMart product listing you are given."
    ),
    lambda state: json.dumps(
        {
            "a2a_task": state["a2a_task"],
            "product_listing": state.get("product_listing", ""),
            "instruction": (
                "Pick 3 candidate SKUs from the product listing and give a short reason for each. "
                "Cite the SKU and price exactly as listed. Do not invent products."
            ),
        },
        indent=2,
    ),
    "shopping_result",
    capability="product_search",
)


pricing_agent_node = make_agent_node(
    "pricing_agent",
    "You are the AgentMart Pricing Agent. Evaluate price, value, and budget fit.",
    lambda state: json.dumps(
        {
            "a2a_task": state["a2a_task"],
            "shopping_result": state.get("shopping_result", "(agent not on this path)"),
            "product_listing": state.get("product_listing", ""),
            "instruction": (
                "Rank the candidates by value using the listed price and list price. "
                "Note any discount, budget overrun, or price risk."
            ),
        },
        indent=2,
    ),
    "pricing_result",
    capability="price_check",
)


inventory_agent_node = make_agent_node(
    "inventory_agent",
    "You are the AgentMart Inventory Agent. Check stock assumptions and availability risks.",
    lambda state: json.dumps(
        {
            "a2a_task": state["a2a_task"],
            "shopping_result": state.get("shopping_result", "(agent not on this path)"),
            "pricing_result": state.get("pricing_result", "(agent not on this path)"),
            "product_listing": state.get("product_listing", ""),
            "instruction": (
                "Use the stock lines in the product listing. Mark which options are in stock now, "
                "which depend on a restock, and what needs confirmation."
            ),
        },
        indent=2,
    ),
    "inventory_result",
    capability="stock_check",
)


fulfillment_agent_node = make_agent_node(
    "fulfillment_agent",
    "You are the AgentMart Fulfillment Agent. Check delivery, pickup, and fulfillment constraints.",
    lambda state: json.dumps(
        {
            "a2a_task": state["a2a_task"],
            "inventory_result": state.get("inventory_result", "(agent not on this path)"),
            "product_listing": state.get("product_listing", ""),
            "instruction": (
                "Recommend a fulfillment path using the delivery methods, ETAs, and costs in the listing. "
                "Mention which warehouse ships the item."
            ),
        },
        indent=2,
    ),
    "fulfillment_result",
    capability="delivery_options",
)


ORDER_AGENT_SYSTEM_PROMPT = (
    "You are the AgentMart Order Agent. You speak to Hermes/MyShopper, which relays to the "
    "customer, so your reply is what a customer reads. Be warm, courteous, and human: write "
    "like a helpful shop assistant talking to a person, address the customer as 'you', and "
    "never dumps raw field labels or log lines at them. Work only from the order book and "
    "agent results you are given. Never invent an order id, tracking reference, amount, or "
    "delivery date. If something is missing, say what is missing and ask for it kindly."
)


def self_consistent_a2a_task(state: AgentMartState) -> dict[str, Any]:
    """Hand the Order Agent a copy of the A2A envelope whose request is unambiguous.

    A peer that says "record this as purchase intent only... do not create or modify
    an order... confirm only that the intent was recorded" is using guardrail phrasing
    even though it named a SKU -- i.e. it wants to buy. The Order Agent has already
    materialised the draft. If the raw wrapper text reached the model as the operative
    instruction it would make the reply claim nothing was created, so replace it with a
    factual statement once a draft exists.
    """
    task = state.get("a2a_task") or {}
    if not (state.get("draft_order") or {}).get("order_id"):
        return {"a2a_task": task}
    if not intent_only_request(state.get("customer_request") or ""):
        return {"a2a_task": task}
    draft_id = state["draft_order"]["order_id"]
    clean = json.loads(json.dumps(task))
    payload = clean.get("payload") or {}
    payload["customer_request"] = (
        f"The customer asked to buy 1 x {state.get('target_sku') or 'the item in the draft'}. "
        f"Draft order {draft_id} has been prepared and is awaiting payment. Any "
        f"'record intent only / do not create an order' wording in the peer's message is "
        f"guardrail phrasing, not the actual outcome."
    )
    return {"a2a_task": clean}


def message_for_drafted_purchase(state: AgentMartState) -> str:
    """The Order Agent instruction once a draft order is on the books.

    The ask for delivery details must read like a person, not like form labels:
    no bare "recipient / address / postal / contact" bullets. Name each missing
    slot in plain words inside a natural, friendly sentence.

    When this turn just persisted delivery slots (an "update ... with exactly:"
    or "report whether the details were accepted" request), the instruction must
    state the outcome explicitly: the model once reported the update as failed
    even though the slots had landed, because the guardrail-wording of the
    peer's request ("do not authorize payment, capture funds, reserve
    inventory") read like nothing had happened.
    """
    if state.get("delivery_updated_slots"):
        slots = state["delivery_updated_slots"]
        saved_plain = ", ".join(
            DELIVERY_SLOT_LABELS[key] for key in REQUIRED_DELIVERY_SLOTS if key in slots
        )
        draft_id = (state.get("draft_order") or {}).get("order_id")
        confirm = (
            f"The delivery details supplied in this turn HAVE been accepted and saved to "
            f"draft {draft_id}: {saved_plain}. However the request is phrased -- even when "
            f"it forbids charging, reserving, or placing an order -- these slots were "
            f"persisted and this is a fact of the order book. State clearly and warmly that "
            f"the update was accepted and stored; NEVER claim the update failed, was not "
            f"accepted, could not be processed, or was not persisted. If the customer asked "
            f"whether the details were accepted, answer that they were."
        )
    else:
        confirm = ""
    missing = state.get("missing_delivery_slots") or []
    if missing:
        plain_labels = [
            DELIVERY_SLOT_LABELS[key] for key in REQUIRED_DELIVERY_SLOTS
        ]
        missing_plain = ", ".join(
            DELIVERY_SLOT_LABELS[key] for key in REQUIRED_DELIVERY_SLOTS if key in missing
        )
        given = ", ".join(
            DELIVERY_SLOT_LABELS[key] for key in REQUIRED_DELIVERY_SLOTS if key not in missing
        ) or "none yet"
        ask = (
            "Keep the draft summary short (two or three lines): item and quantity, "
            "total, fulfillment path, and that nothing is charged until the customer "
            "confirms checkout.\n"
            f"No payment is charged yet, but the draft cannot be finalised without "
            f"delivery details. What the customer has already given: {given}. Still "
            f"needed: {missing_plain}.\n"
            "End the reply by ASKING for the still-missing delivery details in one warm, "
            "natural sentence or two, addressed to the customer in the first person -- for "
            "example: 'Could you share your full name, delivery address, postal code, and "
            "contact number so I can finalise your order? Nothing will be charged until "
            "you confirm.' Name every still-missing slot in plain words (use the labels "
            "above), and never list them as bare form labels such as recipient/address/"
            "postal/contact or say 'please provide exactly'."
        )
    else:
        ask = (
            "The draft has complete delivery details and is ready for the customer to "
            "confirm checkout. Keep the reply short (two or three lines): item and "
            "quantity, total, delivery path, and a friendly first-person line inviting "
            "the customer to confirm so the payment can go through. Nothing is charged "
            "until they confirm. Do not ask for delivery details again."
        )
    lead = confirm + (" " if confirm else "")
    if not intent_only_request(state.get("customer_request") or ""):
        return lead + ask + (
            " The original customer_request is authoritative: report the draft and ask "
            "for any missing delivery details conversationally."
        )
    return lead + ask + (
        " IMPORTANT: peer_requested_intent_only is true. The requesting peer wrapped the "
        "purchase in guardrail language ('record purchase intent only', 'do not create or "
        "modify an order', 'confirm only that the intent was recorded'). That is the peer "
        "being cautious, NOT a statement of fact: the draft order WAS created and is "
        "awaiting payment. Ignore that wrapper completely in your reply. NEVER claim no "
        "order was created, modified, or charged, and never claim only the intent was "
        "recorded. Report the draft order id, its contents and total, and ask for any "
        "missing delivery details conversationally."
    )


def _order_agent_prompt(state: AgentMartState) -> str:
    intent = state.get("intent", "product_advice")
    common = {
        "a2a_task": state["a2a_task"],
        "intent": intent,
        "customer_id": state.get("customer_id"),
    }

    if intent == "order_status":
        return json.dumps(
            {
                **common,
                "order_book": state.get("order_context", ""),
                "instruction": (
                    "Answer the customer's order-status question from the order book in a warm, "
                    "friendly, human voice. Address the customer as 'you' and write like an "
                    "assistant talking to a person, never like a log file or a form.\n"
                    "Open with one short courtesy line ('Thanks for checking in!', 'Here's how "
                    "your order looks so far'). Then give the essential facts in flowing prose, "
                    "not a rigid list of labels: what was ordered, its current status, the "
                    "total, and what happens next. A short styled list is fine when it genuinely "
                    "aids reading, but prefer natural sentences.\n"
                    "If the request names a specific order id, focus on it, but still report "
                    "every OTHER order in the book in one line each. If no order id is named, "
                    "every order in the book is relevant: report each one with order id, items, "
                    "status, total, tracking/ETA when present, and what the customer should do "
                    "next (for example, a draft awaiting payment needs a checkout step). Zero in "
                    "on any order that contains a SKU or product the customer mentioned. Never "
                    "skip or merge orders just because one order id was emphasized in the "
                    "request. Close with a friendly question or next step, and keep anything "
                    "uncertain plainly honest ('I couldn't confirm that from the order book')."
                ),
            },
            indent=2,
        )

    if intent == "purchase_intent":
        if state.get("intent_only_recorded"):
            return json.dumps(
                {
                    **common,
                    "intent_only_recorded": True,
                    "instruction": (
                        "The customer asked to RECORD the purchase intent only, and "
                        "explicitly forbade creating or modifying an order, reserving "
                        "stock, or taking any payment action. No order was created, "
                        "modified, or charged. Confirm the product and quantity named in "
                        "the request (use the original customer_request), state plainly "
                        "that no order was created and no payment was made, and do not "
                        "present any draft, total, or checkout step. Say it warmly and "
                        "reassuringly, as if to a person: a short friendly line confirming "
                        "they're all set whenever they want to go ahead, without any "
                        "checkout prompt."
                    ),
                },
                indent=2,
            )
        return json.dumps(
            {
                **common,
                **self_consistent_a2a_task(state),
                "draft_order": state.get("draft_order", {}),
                "delivery_details": state.get("delivery_details", {}),
                "missing_delivery_slots": state.get("missing_delivery_slots", []),
                "peer_requested_intent_only": intent_only_request(state.get("customer_request") or ""),
                "inventory_result": state.get("inventory_result", "(agent not on this path)"),
                "fulfillment_result": state.get("fulfillment_result", "(agent not on this path)"),
                "instruction": (
                    message_for_drafted_purchase(state)
                ),
            },
            indent=2,
        )

    if intent == "checkout_payment":
        blocked = state.get("checkout_blocked_reason") or state.get("missing_delivery_slots")
        return json.dumps(
            {
                **common,
                "order_book": state.get("order_context", ""),
                "order_to_settle": state.get("target_order_id"),
                "delivery_details": state.get("delivery_details", {}),
                "missing_delivery_slots": state.get("missing_delivery_slots", []),
                "checkout_blocked_reason": state.get("checkout_blocked_reason", ""),
                "instruction": (
                    "Resolve the single order the customer wants to settle and restate its "
                    "total, items, and payment method for confirmation in a warm, friendly, "
                    "human voice. Address the customer as 'you', write in natural sentences "
                    "rather than a dry form, and reassure them that nothing is charged until "
                    "they confirm. Do not claim payment has happened: the Payment Agent runs "
                    "next.\n"
                    "If checkout_blocked_reason is set, the order may NOT be settled yet: tell "
                    "the customer kindly and conversationally that payment cannot proceed "
                    "because the delivery details are incomplete, NAME in plain words exactly "
                    "what is still missing (recipient name, full delivery address, postal "
                    "code, contact number), and warmly invite them to share it so the order "
                    "can move forward. There is no charge until they provide them."
                ),
            },
            indent=2,
        )

    # browse_catalog / product_advice: the original recommendation summary
    return json.dumps(
        {
            **common,
            "shopping_result": state.get("shopping_result", "(agent not on this path)"),
            "pricing_result": state.get("pricing_result", "(agent not on this path)"),
            "inventory_result": state.get("inventory_result", "(agent not on this path)"),
            "fulfillment_result": state.get("fulfillment_result", "(agent not on this path)"),
            "instruction": (
                "Produce a final recommendation that Hermes/MyShopper can send back to the customer. "
                "Do not pretend an order was placed; summarize the recommended next action."
            ),
        },
        indent=2,
    )


def _persist_delivery_details(
    state: AgentMartState,
    draft: dict[str, Any],
    customer_id: str,
) -> tuple[dict[str, Any], list[str]]:
    """Extract delivery slots from the incoming request and persist them.

    Returns ``(refreshed_draft, slots_written)``. Anything the customer supplied
    is written even if slots are still missing; the caller records what is
    outstanding so the prompt can tell the Order Agent to ask for it. ``slots_written``
    names the slots this turn actually persisted, empty when nothing was supplied or
    the write failed -- the prompt needs it so an update is never reported as failed
    when it in fact landed.
    """
    request_text = state.get("customer_request") or ""
    if not request_text:
        task = state.get("a2a_task") or {}
        request_text = (task.get("payload") or {}).get("customer_request") or ""
    provided = extract_delivery_details(request_text)
    if not provided:
        return draft, []
    try:
        refreshed = update_draft_delivery(draft["order_id"], provided, db_path=None)
        return refreshed, [key for key in REQUIRED_DELIVERY_SLOTS if provided.get(key)]
    except Exception as exc:  # never fail the hop because a slot could not be written
        log.error("order_agent: delivery persist failed for %s: %s", draft["order_id"], exc)
        return draft, []


def order_agent_node(state: AgentMartState) -> AgentMartState:
    """Order Agent. Reads the order book; creates a draft order on a purchase intent."""
    intent = state.get("intent", "product_advice")
    customer_id = state.get("customer_id", DEFAULT_CUSTOMER_ID)

    envelope = emit_envelope(
        state,
        sender="hermes_myshopper",
        recipient="order_agent",
        intent=intent,
        payload={"capability": "order_summary", "customer_id": customer_id},
        lifecycle="accepted",
    )

    next_state: AgentMartState = {**state}

    # A purchase intent materialises a real draft order before the model speaks.
    # The one exception is a request that ONLY records intent and names no
    # product at all: there is nothing to draft yet. A named SKU is an express
    # wish to buy, so even a guardrail-y "record purchase intent only, do not
    # create an order" wrapper must still yield the draft and ask for delivery
    # details -- otherwise the customer can never get to checkout.
    if intent == "purchase_intent":
        sku = state.get("target_sku")
        draft: dict[str, Any] | None = None
        if not sku and intent_only_request(state.get("customer_request") or ""):
            log.info("order_agent: intent-only request with no SKU - order book left untouched")
            next_state["intent_only_recorded"] = True
        elif not sku:
            open_drafts = sorted(
                (
                    order for order in list_orders(customer_id=customer_id)
                    if order["status"] in ("draft", "awaiting_payment") and not order["is_paid"]
                ),
                key=lambda o: o["placed_at"],
                reverse=True,
            )
            if open_drafts:
                draft = open_drafts[0]
                log.info(
                    "order_agent: refreshed newest open draft %s (no sku named, customer=%s)",
                    draft["order_id"], customer_id,
                )
            else:
                next_state["draft_order"] = {"error": "no SKU identified and no open draft to update"}
                log.warning(
                    "order_agent: purchase_intent with no target_sku and no open draft (customer=%s)",
                    customer_id,
                )
        else:
            try:
                existing = find_open_draft(customer_id, sku)
                if existing:
                    draft = existing
                    log.info(
                        "order_agent: refreshed draft %s for sku=%s customer=%s",
                        draft["order_id"], sku, customer_id,
                    )
                else:
                    draft = create_draft_order(
                        customer_id=customer_id,
                        items=[{"sku": sku, "quantity": 1}],
                        warehouse="SG-CENTRAL",
                        fulfillment_method="standard_delivery",
                    )
                    log.info("order_agent: drafted %s for sku=%s customer=%s", draft["order_id"], sku, customer_id)
            except (CatalogNotSeededError, OrderBookNotSeededError, ValueError) as exc:
                next_state["draft_order"] = {"error": f"{type(exc).__name__}: {exc}"}
                log.error("order_agent: draft failed for sku=%s: %s", sku, exc)

        if draft:
                draft, updated_slots = _persist_delivery_details(
                    state, draft, customer_id=customer_id
                )
                next_state["draft_order"] = draft
                next_state["target_order_id"] = draft["order_id"]
                next_state["order_context"] = format_order(draft)
                next_state["delivery_details"] = {
                    key: draft.get(f"delivery_{key}")
                    for key in REQUIRED_DELIVERY_SLOTS
                    if draft.get(f"delivery_{key}")
                }
                next_state["missing_delivery_slots"] = [
                    key for key in REQUIRED_DELIVERY_SLOTS
                    if not (draft.get(f"delivery_{key}") or "").strip()
                ]
                if updated_slots:
                    next_state["delivery_updated_slots"] = updated_slots
                log.info(
                    "order_agent: draft %s delivery_see=%s missing=%s",
                    draft["order_id"],
                    ",".join(next_state["delivery_details"]) or "-",
                    ",".join(next_state["missing_delivery_slots"]) or "-",
                )

    # A bare "checkout and pay" resolves to the newest open draft (or, when a
    # SKU is named, the newest open draft FOR that SKU -- creating one if the
    # purchase intent never materialised), then delivery completeness gates
    # whether the Payment Agent may run.
    if intent == "checkout_payment":
        target_id = next_state.get("target_order_id")
        if not target_id:
            sku = state.get("target_sku")
            try:
                if sku:
                    draft = find_open_draft(customer_id, sku)
                    if not draft:
                        draft = create_draft_order(
                            customer_id=customer_id,
                            items=[{"sku": sku, "quantity": 1}],
                            warehouse="SG-CENTRAL",
                            fulfillment_method="standard_delivery",
                        )
                        log.info(
                            "order_agent: drafted %s for sku=%s during checkout (customer=%s)",
                            draft["order_id"], sku, customer_id,
                        )
                    target_id = draft["order_id"]
                    next_state["target_order_id"] = target_id
                    next_state["order_context"] = format_order(draft)
                else:
                    payable = find_payable_order(customer_id)
                    if payable:
                        target_id = payable["order_id"]
                        next_state["target_order_id"] = target_id
                        next_state["order_context"] = format_order(payable)
            except (CatalogNotSeededError, OrderBookNotSeededError, ValueError) as exc:
                next_state["order_context"] = f"(order book unavailable: {exc})"

        if target_id:
            try:
                target = get_order(target_id)
            except OrderNotFoundError:
                target = None
            if target:
                # The checkout turn itself may carry the missing delivery slots;
                # persist them before deciding whether settlement may proceed.
                updated, _ = _persist_delivery_details(
                    state, target, customer_id=customer_id
                )
                if updated:
                    target = updated
                next_state["order_context"] = format_order(target)
                provided = {
                    key: target.get(f"delivery_{key}")
                    for key in REQUIRED_DELIVERY_SLOTS
                    if (target.get(f"delivery_{key}") or "").strip()
                }
                missing = missing_delivery_slots(target)
                next_state["delivery_details"] = provided
                next_state["missing_delivery_slots"] = missing
                if missing:
                    next_state["checkout_blocked_reason"] = (
                        "checkout refused: order " + target_id
                        + " is missing required delivery details: "
                        + ", ".join(DELIVERY_SLOT_LABELS[key] for key in missing)
                        + ". The customer must provide them before any payment is attempted."
                    )
                    log.info(
                        "order_agent: checkout blocked for %s (missing=%s)",
                        target_id, ",".join(missing),
                    )
                else:
                    next_state["checkout_blocked_reason"] = ""

    client = OpenRouterHermesClient(
        dry_run=state.get("dry_run", False),
        model_config=model_config_from_state(state),
    )
    started = time.monotonic()
    log.info("order_agent: starting (intent=%s, order=%s, model=%s)", intent, next_state.get("target_order_id"), client.model)
    result = client.complete("order_agent", ORDER_AGENT_SYSTEM_PROMPT, _order_agent_prompt(next_state))
    log.info(
        "order_agent: completed in %.2fs (%d chars) | %s",
        time.monotonic() - started,
        len(result),
        client.usage_line(),
    )
    bind_agent("order_agent").debug("reply: %s", result)

    done = emit_envelope(
        next_state,
        sender="order_agent",
        recipient="hermes_myshopper",
        intent=intent,
        payload={
            "order_id": next_state.get("target_order_id"),
            "draft_created": bool(next_state.get("draft_order", {}).get("order_id")),
        },
        lifecycle="completed",
    )

    # Only the keys this node actually decided; transcript/a2a_log go back as deltas.
    delta: AgentMartState = {
        k: v for k, v in next_state.items()
        if k in ("draft_order", "target_order_id", "order_context", "delivery_details", "delivery_updated_slots", "missing_delivery_slots", "checkout_blocked_reason", "intent_only_recorded")
    }
    delta["transcript"] = [transcript_entry("order_agent", result, envelope)]
    delta["a2a_log"] = [envelope, done]
    delta["order_result"] = result
    return delta


PAYMENT_AGENT_SYSTEM_PROMPT = (
    "You are the AgentMart Payment Agent. Payments in this workshop are SIMULATED: "
    "the receipt you are given was written to a local database and no payment processor "
    "was contacted. Confirm the settled order back to Hermes/MyShopper using only the "
    "receipt values, and state plainly that this was a simulated payment. Speak warmly "
    "and reassure the customer: thank them for their order, confirm what was received, "
    "and only then note the simulation in a soft, honest way.\n"
    "If the receipt has status 'blocked', NO payment was made: no authorization, no "
    "capture, nothing. Report the blocking reason from the receipt and the missing "
    "delivery details the customer still owes kindly, and say there is no charge yet. "
    "Never describe a blocked receipt as a completed or simulated payment.\n"
    "If the receipt has status 'readonly_no_charge', NO payment was attempted either: "
    "the request was a read-only verification or lookup, so the receipt simply reports "
    "the order's current state from the order_book. Say plainly, in a friendly tone, "
    "that nothing was charged, captured, or authorized."
)


def _payment_request_text(state: AgentMartState) -> str:
    return (state.get("customer_request") or "").strip()


def payment_agent_node(state: AgentMartState) -> AgentMartState:
    """Payment Agent. Authorizes and captures a SIMULATED payment, then reports back."""
    order_id = state.get("target_order_id")
    request_text = _payment_request_text(state)

    envelope = emit_envelope(
        state,
        sender="order_agent",
        recipient="payment_agent",
        intent="checkout_payment",
        payload={"capability": "payment_capture", "order_id": order_id, "simulated": True},
        lifecycle="accepted",
    )

    next_state: AgentMartState = {**state}
    receipt: dict[str, Any]
    if not order_id:
        receipt = {"error": "no payable order found for this customer"}
        log.warning("payment_agent: checkout_payment with no target_order_id")
    elif is_read_only_request(request_text) or not _explicit_payment_imperative(request_text):
        # Defense in depth: routing sends read-only/verification phrasings to the
        # Order Agent, but if one still reaches this node it must NEVER charge.
        receipt = {
            "status": "readonly_no_charge",
            "order_id": order_id,
            "reason": (
                "the request did not clearly instruct payment; it read as "
                "read-only verification or a status lookup. Nothing was charged."
            ),
            "order_status": None,
            "simulated": True,
        }
        try:
            lookup = get_order(order_id)
            if lookup:
                receipt["order_status"] = lookup["status"]
        except Exception:  # noqa: BLE001 - a lookup failure must not turn into a charge
            receipt["order_status"] = "unknown"
        log.info(
            "payment_agent: refused to settle %s — read-only / no explicit payment instruction",
            order_id,
        )
    elif state.get("checkout_blocked_reason"):
        receipt = {
            "status": "blocked",
            "order_id": order_id,
            "reason": state["checkout_blocked_reason"],
            "missing_delivery_slots": state.get("missing_delivery_slots", []),
            "simulated": True,
        }
        log.info(
            "payment_agent: refused to settle %s — delivery details incomplete",
            order_id,
        )
    else:
        try:
            settled = checkout_and_pay(order_id)
            receipt = {
                "simulated": True,
                "payment_id": settled["payment"]["payment_id"],
                "processor_ref": settled["payment"]["processor_ref"],
                "status": settled["payment"]["status"],
                "amount_usd": settled["payment"]["amount_usd"],
                "method_id": settled["payment"]["method_id"],
                "order_id": order_id,
                "order_status": settled["order"]["status"],
            }
            log.info(
                "payment_agent: SIMULATED capture ok order=%s payment=%s amount_usd=%s",
                order_id,
                receipt["payment_id"],
                receipt["amount_usd"],
            )
        except (OrderNotFoundError, OrderBookNotSeededError, ValueError) as exc:
            receipt = {"error": f"{type(exc).__name__}: {exc}", "order_id": order_id}
            log.error("payment_agent: capture failed for %s: %s", order_id, exc)

    next_state["payment_receipt"] = receipt

    client = OpenRouterHermesClient(
        dry_run=state.get("dry_run", False),
        model_config=model_config_from_state(state),
    )
    started = time.monotonic()
    log.info("payment_agent: explaining receipt (order=%s, simulated=True)", order_id)
    result = client.complete(
        "payment_agent",
        PAYMENT_AGENT_SYSTEM_PROMPT,
        json.dumps(
            {
                "a2a_task": state["a2a_task"],
                "receipt": receipt,
                "order_book": state.get("order_context", ""),
            },
            indent=2,
        ),
    )
    log.info(
        "payment_agent: completed in %.2fs (%d chars) | %s",
        time.monotonic() - started,
        len(result),
        client.usage_line(),
    )
    bind_agent("payment_agent").debug("reply: %s", result)

    done = emit_envelope(
        next_state,
        sender="payment_agent",
        recipient="hermes_myshopper",
        intent="checkout_payment",
        payload={k: v for k, v in receipt.items() if k != "error"} or {"error": receipt.get("error")},
        lifecycle="failed" if "error" in receipt else "completed",
    )

    return {
        "transcript": [transcript_entry("payment_agent", result, envelope)],
        "a2a_log": [envelope, done],
        "payment_receipt": receipt,
        "payment_result": result,
    }


# Which AgentMart agents each intent actually visits. Routing on capability is
# the Part 7 teaching point: a status question must not wake the whole ecosystem.
INTENT_PATHS: dict[str, tuple[str, ...]] = {
    "browse_catalog": ("shopping_agent", "pricing_agent", "inventory_agent", "order_agent"),
    "product_advice": (
        "shopping_agent",
        "pricing_agent",
        "inventory_agent",
        "fulfillment_agent",
        "order_agent",
    ),
    # Stops at the Order Agent on purpose: a purchase intent leaves a draft in
    # `awaiting_payment`. Settling it is a separate, explicit checkout turn.
    "purchase_intent": ("inventory_agent", "fulfillment_agent", "order_agent"),
    "checkout_payment": ("order_agent", "payment_agent"),
    "order_status": ("order_agent",),
}


# Pricing, Inventory and Fulfillment all read the Shopping Agent's shortlist and
# none reads another's output, so running them in sequence only bought latency.
# They now share one superstep and the Order Agent joins on all of them.
PARALLEL_AGENTS = frozenset({"pricing_agent", "inventory_agent", "fulfillment_agent"})


def _stages(intent: str) -> list[str | list[str]]:
    """The intent's path, with consecutive independent agents grouped into one stage.

    product_advice becomes:
        shopping_agent -> [pricing, inventory, fulfillment] -> order_agent
    """
    stages: list[str | list[str]] = []
    for agent in INTENT_PATHS[intent]:
        if agent in PARALLEL_AGENTS and stages and isinstance(stages[-1], list):
            stages[-1].append(agent)
        elif agent in PARALLEL_AGENTS:
            stages.append([agent])
        else:
            stages.append(agent)
    return stages


def route_from_hermes(state: AgentMartState) -> str | list[str]:
    """First AgentMart stage for this intent. A list fans out in one superstep."""
    stage = _stages(state.get("intent", "product_advice"))[0]
    _debug_stage("hermes_myshopper", state.get("intent", "product_advice"), stage, "first")
    return stage


def _next_after(node: str) -> Callable[[AgentMartState], str | list[str]]:
    """Follow this intent's stages; fall off to END when the stage is the last one.

    Every member of a parallel stage returns the same next stage, which is what makes
    the Order Agent a join: LangGraph runs it once, after the whole stage completes.
    """

    def router(state: AgentMartState) -> str | list[str]:
        stages = _stages(state.get("intent", "product_advice"))
        for index, stage in enumerate(stages):
            members = stage if isinstance(stage, list) else [stage]
            if node in members:
                nxt = stages[index + 1] if index + 1 < len(stages) else END
                _debug_stage(node, state.get("intent", "product_advice"), nxt, "next")
                return nxt
        _debug_stage(node, state.get("intent", "product_advice"), END, "end")
        return END

    return router


def _debug_stage(src: str, intent: str, dest: str | list[str], stage_kind: str) -> None:
    """One-line graph-traversal evidence: which node handed control to which stage."""
    target = ",".join(dest) if isinstance(dest, list) else dest
    log.debug("graph: %s -> %s [%s stage, intent=%s]", src, target, stage_kind, intent)


def build_graph():
    graph = StateGraph(AgentMartState)
    graph.add_node("hermes_myshopper", hermes_myshopper_node)
    graph.add_node("shopping_agent", shopping_agent_node)
    graph.add_node("pricing_agent", pricing_agent_node)
    graph.add_node("inventory_agent", inventory_agent_node)
    graph.add_node("fulfillment_agent", fulfillment_agent_node)
    graph.add_node("order_agent", order_agent_node)
    graph.add_node("payment_agent", payment_agent_node)

    graph.add_edge(START, "hermes_myshopper")

    agent_nodes = [
        "shopping_agent",
        "pricing_agent",
        "inventory_agent",
        "fulfillment_agent",
        "order_agent",
        "payment_agent",
    ]
    graph.add_conditional_edges(
        "hermes_myshopper",
        route_from_hermes,
        {name: name for name in agent_nodes},
    )
    for name in agent_nodes:
        graph.add_conditional_edges(
            name,
            _next_after(name),
            {**{other: other for other in agent_nodes if other != name}, END: END},
        )
    return graph.compile()


def _hop_agent(hop: dict[str, Any]) -> str:
    """The AgentMart agent a hop concerns, whichever side of the exchange it sits on."""
    return hop["recipient"] if hop["sender"] == "hermes_myshopper" else hop["sender"]


def normalize_ordering(result: AgentMartState) -> AgentMartState:
    """Re-sort the parallel stage's rows back into the intent's declared path order.

    A superstep finishes in whatever order the network returns, which would make the
    transcript non-deterministic. Sorting by the intent path keeps the replay stable
    and the A2A log readable; the sort is stable, so each agent's accepted/completed
    pair keeps its relative order. The work still happened concurrently.
    """
    order = ["hermes_myshopper", *INTENT_PATHS[result.get("intent", "product_advice")]]
    rank = {name: index for index, name in enumerate(order)}
    result["transcript"] = sorted(
        result.get("transcript", []), key=lambda e: rank.get(e["agent"], len(rank))
    )
    # The opening handoff is addressed to 'agentmart' itself, so it sorts ahead of everything.
    result["a2a_log"] = sorted(
        result.get("a2a_log", []), key=lambda h: rank.get(_hop_agent(h), -1)
    )
    return result


def run_agentmart(
    customer_request: str,
    channel: str = "webchat",
    dry_run: bool = False,
    config_path: str | None = None,
    category: str | None = None,
    max_price: float | None = None,
    customer_id: str = DEFAULT_CUSTOMER_ID,
    intent: Intent | None = None,
) -> AgentMartState:
    app = build_graph()
    log.info("run_agentmart: invoke request=%r channel=%s customer=%s", customer_request, channel, customer_id)
    usage_baseline = dict(RUN_USAGE)
    started = time.monotonic()
    result = normalize_ordering(app.invoke(
        {
            "customer_request": customer_request,
            "channel": channel,
            "dry_run": dry_run,
            "customer_id": customer_id,
            "intent": intent or classify_intent(customer_request),
            "hermes_a2a_config": load_hermes_a2a_config(config_path),
            "product_listing": load_product_listing(category=category, max_price=max_price),
            "transcript": [],
            "a2a_log": [],
        }
    ))
    elapsed = time.monotonic() - started
    log.info(
        "run_agentmart: done in %.2fs intent=%s agents_woken=%s dry_run=%s",
        elapsed,
        result.get("intent"),
        ",".join(e.get("agent", "?") for e in result.get("transcript", [])),
        dry_run,
    )
    _log_usage_summary(usage_baseline)
    return result


def _usage_delta(baseline: dict[str, dict[str, int]], agent: str) -> dict[str, int]:
    before = baseline.get(agent, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    after = RUN_USAGE.get(agent, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return {key: after.get(key, 0) - before.get(key, 0) for key in before}


def _log_usage_summary(baseline: dict[str, dict[str, int]]) -> None:
    """Per-agent token + latency accounting for the run that just finished."""
    rows = [
        (agent, _usage_delta(baseline, agent))
        for agent in sorted(RUN_USAGE)
    ]
    for agent, usage in rows:
        if usage["calls"] > 0:
            log.info(
                "usage %s: calls=%d tokens=%d (prompt %d + completion %d)",
                agent,
                usage["calls"],
                usage["total_tokens"],
                usage["prompt_tokens"],
                usage["completion_tokens"],
            )
    total_tokens = sum(u["total_tokens"] for _, u in rows if u["calls"] > 0)
    total_calls = sum(u["calls"] for _, u in rows if u["calls"] > 0)
    if total_calls > 0:
        log.info("usage run: total tokens=%d across %d interactions", total_tokens, total_calls)


def check_model_connection(config_path: str | None = None) -> int:
    """Verify the key, endpoint and model before running the full graph."""
    config = load_hermes_a2a_config(config_path)
    model_config = config["hermes_agent"].get("model", {})
    client = OpenRouterHermesClient(model_config=model_config)

    endpoint = "OpenAI" if client.openai_native else "OpenRouter"
    print("Hermes model configuration")
    print(f"  endpoint    : {endpoint}")
    print(f"  base_url    : {client.base_url}")
    print(f"  model       : {client.model}")
    # Model fallbacks and temperature are OpenRouter features; say so rather than
    # printing settings that this endpoint will silently ignore.
    if client.openai_native:
        print(f"  fallbacks   : n/a (OpenRouter only)")
        print(f"  temperature : n/a (model default)")
        print(f"  max_tokens  : {client.max_tokens} (sent as max_completion_tokens)")
    else:
        print(f"  fallbacks   : {', '.join(client.fallback_models) or 'none'}")
        print(f"  temperature : {client.temperature}")
        print(f"  max_tokens  : {client.max_tokens}")
    print(f"  reasoning   : {client.reasoning_effort or 'default'}")
    print(f"  api_key     : {'set' if client.api_key else 'MISSING'}")

    if not client.api_key:
        key_var = "OPENAI_API_KEY" if client.openai_native else "OPENROUTER_API_KEY"
        print(f"\n{key_var} is not set. Add it to .env, then re-run.")
        return 1

    print(f"\nCalling {endpoint}...")
    try:
        reply = client.complete(
            "hermes_myshopper",
            "You are Hermes/MyShopper. Reply with a single short sentence.",
            "Confirm that the Hermes model connection is working.",
        )
    except Exception as exc:  # noqa: BLE001 - surface any client/transport error to the workshop user
        print(f"Connection failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"Reply: {reply.strip()[:300]}")
    print("\nConnection OK.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AgentMart LangGraph workshop demo.")
    parser.add_argument(
        "request",
        nargs="?",
        help="Customer buying request to send through Hermes/MyShopper.",
    )
    parser.add_argument("--channel", default="webchat", help="Customer channel name.")
    parser.add_argument(
        "--customer",
        default=DEFAULT_CUSTOMER_ID,
        help=f"Customer id from the seeded order book (default: {DEFAULT_CUSTOMER_ID}).",
    )
    parser.add_argument(
        "--intent",
        choices=list(INTENT_PATHS),
        help="Force an intent instead of routing on the request text.",
    )
    parser.add_argument("--config", help="Path to Hermes A2A configuration JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Run without calling OpenRouter.")
    parser.add_argument("--category", help="Limit the seeded product listing to a category, e.g. audio/earbuds.")
    parser.add_argument("--max-price", type=float, help="Limit the seeded product listing by maximum price.")
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="Print the Hermes model settings and test the OpenRouter connection, then exit.",
    )
    args = parser.parse_args()

    if args.check_model:
        raise SystemExit(check_model_connection(args.config))

    if not args.request:
        parser.error("a customer request is required (or use --check-model)")

    result = run_agentmart(
        args.request,
        channel=args.channel,
        dry_run=args.dry_run,
        config_path=args.config,
        category=args.category,
        max_price=args.max_price,
        customer_id=args.customer,
        intent=args.intent,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
