"""End-to-end scenario suite for Hermes/MyShopper and the AgentMart A2A ecosystem.

Each scenario sends one customer message through the real LangGraph workflow and
then checks what actually happened: which intent the router picked, which agents
were woken, what A2A envelopes crossed the wire, and what landed in the order
book. Scenarios run in dry-run by default, so the whole suite passes with no
OpenRouter key: everything asserted here is the deterministic part of the system
(routing, A2A envelope chain, order and payment state). `--live` sends the same
scenarios through the model as well.

Payments are SIMULATED throughout. No payment processor is contacted.

Usage:
    python test_scenarios.py                      # run everything, dry-run
    python test_scenarios.py --list               # list scenario names
    python test_scenarios.py -s order-status      # run one scenario
    python test_scenarios.py --verbose            # show agent replies + A2A log
    python test_scenarios.py --live               # call OpenRouter for real
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agentmart_ecosystem import INTENT_PATHS, classify_intent, run_agentmart
from orders import get_order, list_orders
from seed_data import seed

Check = tuple[bool, str]


# ---------------------------------------------------------------------------
# assertion helpers
# ---------------------------------------------------------------------------
def agents_visited(result: dict[str, Any]) -> list[str]:
    return [entry["agent"] for entry in result.get("transcript", [])]


def expect(label: str, condition: bool) -> Check:
    return bool(condition), label


def check_path(result: dict[str, Any], intent: str) -> list[Check]:
    """Router picked `intent`, and exactly that intent's agents ran."""
    expected = ["hermes_myshopper", *INTENT_PATHS[intent]]
    actual = agents_visited(result)
    return [
        expect(f"intent is {intent}", result.get("intent") == intent),
        expect(f"agent path is {' -> '.join(expected)}", actual == expected),
    ]


def check_a2a_chain(result: dict[str, Any]) -> list[Check]:
    """Every hop shares one correlation id and carries a valid lifecycle state."""
    log = result.get("a2a_log", [])
    correlation_ids = {hop["correlation_id"] for hop in log}
    states = {hop["state"] for hop in log}
    valid = {"proposed", "accepted", "in_progress", "completed", "failed"}
    protocols = {hop["protocol"] for hop in log}
    return [
        expect("A2A log is not empty", len(log) > 0),
        expect(f"one correlation id across {len(log)} hops", len(correlation_ids) == 1),
        expect("every hop has a valid lifecycle state", states <= valid),
        expect("every hop uses agentmart.a2a.v1", protocols == {"agentmart.a2a.v1"}),
        expect("task opens as 'proposed'", bool(log) and log[0]["state"] == "proposed"),
    ]


def check_no_error_text(result: dict[str, Any]) -> list[Check]:
    joined = " ".join(entry["message"] for entry in result.get("transcript", []))
    return [expect("no traceback leaked into an agent reply", "Traceback" not in joined)]


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    request: str
    intent: str
    describes: str
    channel: str = "webchat"
    customer_id: str = "CUST-1001"
    setup: Callable[[], None] | None = None
    checks: list[Callable[[dict[str, Any]], list[Check]]] = field(default_factory=list)


def _seeded_awaiting_payment_order() -> str:
    """The order the seed data leaves unpaid for CUST-1001."""
    return "AM-ORD-20260915-0003"


def check_order_status(result: dict[str, Any]) -> list[Check]:
    context = result.get("order_context", "")
    orders = list_orders(customer_id="CUST-1001")
    return [
        expect("order book was loaded into the Order Agent prompt", bool(context)),
        expect("all 3 seeded orders for CUST-1001 are visible", len(orders) == 3),
        expect(
            "the in-transit order and its tracking ref are in context",
            "AM-ORD-20260912-0002" in context and "SGX4417900456" in context,
        ),
        expect(
            "the unpaid order is flagged awaiting_payment",
            "awaiting_payment" in context,
        ),
        expect("only the Order Agent was woken", agents_visited(result) == ["hermes_myshopper", "order_agent"]),
    ]


def check_specific_order(result: dict[str, Any]) -> list[Check]:
    context = result.get("order_context", "")
    return [
        expect("order id parsed out of the message", result.get("target_order_id") == "AM-ORD-20260912-0002"),
        expect("the requested order appears in context", "AM-ORD-20260912-0002" in context),
        expect("status in_transit surfaced", "in_transit" in context),
    ]


def check_unknown_order(result: dict[str, Any]) -> list[Check]:
    context = result.get("order_context", "")
    return [
        expect("unknown order id was still parsed", result.get("target_order_id") == "AM-ORD-9999-9999"),
        expect("missing order is flagged in context", "does not appear" in context.lower() or "not found" in context.lower()),
        expect("graph still completed", "order_result" in result),
    ]


def check_listing(result: dict[str, Any]) -> list[Check]:
    listing = result.get("product_listing", "")
    return [
        expect("seeded catalog reached the agents", "AM-EAR-1001" in listing),
        expect("listing carries real prices", "price $" in listing),
        expect("listing carries stock lines", "stock:" in listing),
        expect("Fulfillment Agent is skipped when only browsing", "fulfillment_agent" not in agents_visited(result)),
    ]


def check_draft_order(result: dict[str, Any]) -> list[Check]:
    draft = result.get("draft_order", {})
    order_id = draft.get("order_id")
    checks = [
        expect("SKU parsed from the request", result.get("target_sku") == "AM-EAR-1002"),
        expect("a draft order was created", bool(order_id)),
        expect("draft is awaiting_payment, not paid", draft.get("status") == "awaiting_payment"),
        expect("line item is the requested SKU", [i["sku"] for i in draft.get("items", [])] == ["AM-EAR-1002"]),
        expect("priced from the catalog ($89.00 + $3.50 shipping)", draft.get("total_usd") == 92.5),
        expect("nothing was charged", draft.get("amount_paid_usd", 0) == 0),
        expect("Payment Agent did NOT run", "payment_agent" not in agents_visited(result)),
    ]
    if order_id:
        persisted = get_order(order_id)
        checks.append(expect("draft is persisted in the order book", persisted["status"] == "awaiting_payment"))
    return checks


def check_checkout(result: dict[str, Any]) -> list[Check]:
    receipt = result.get("payment_receipt", {})
    order_id = receipt.get("order_id")
    checks = [
        expect("an order was selected to settle", bool(order_id)),
        expect("payment captured", receipt.get("status") == "captured"),
        expect("payment is flagged simulated", receipt.get("simulated") is True),
        expect("reference is a simulated one", str(receipt.get("processor_ref", "")).startswith("sim_")),
        expect("customer's default method was used", receipt.get("method_id") == "PM-VISA-4417"),
        expect("amount matches the order total ($93.00)", receipt.get("amount_usd") == 93.0),
    ]
    if order_id:
        persisted = get_order(order_id)
        checks += [
            expect("order moved to paid in the order book", persisted["status"] == "paid"),
            expect("order reads as fully paid", persisted["is_paid"] is True),
        ]
    return checks


def check_full_advice_path(result: dict[str, Any]) -> list[Check]:
    return [
        expect("all five AgentMart agents ran", len(agents_visited(result)) == 6),
        expect("catalog reached the Shopping Agent", "AM-EAR-" in result.get("product_listing", "")),
        expect("a final recommendation came back", bool(result.get("order_result"))),
    ]


SCENARIOS: list[Scenario] = [
    Scenario(
        name="order-status",
        request="What is my order status?",
        intent="order_status",
        channel="telegram",
        describes="A bare status question must wake only the Order Agent and read the real order book.",
        checks=[lambda r: check_path(r, "order_status"), check_order_status, check_a2a_chain],
    ),
    Scenario(
        name="order-status-specific",
        request="Where is my order AM-ORD-20260912-0002?",
        intent="order_status",
        channel="whatsapp",
        describes="An order id in the message scopes the Order Agent to that single order.",
        checks=[lambda r: check_path(r, "order_status"), check_specific_order, check_a2a_chain],
    ),
    Scenario(
        name="order-status-unknown",
        request="Where is my order AM-ORD-9999-9999?",
        intent="order_status",
        describes="An order id that does not exist degrades gracefully instead of crashing the graph.",
        checks=[lambda r: check_path(r, "order_status"), check_unknown_order, check_no_error_text],
    ),
    Scenario(
        name="list-products",
        request="List me the available products.",
        intent="browse_catalog",
        describes="Browsing routes to Shopping/Pricing/Inventory and skips Fulfillment.",
        checks=[lambda r: check_path(r, "browse_catalog"), check_listing, check_a2a_chain],
    ),
    Scenario(
        name="buy-this",
        request="I want to buy this AM-EAR-1002.",
        intent="purchase_intent",
        channel="telegram",
        describes="A purchase intent creates a real draft order that stops at awaiting_payment.",
        checks=[lambda r: check_path(r, "purchase_intent"), check_draft_order, check_a2a_chain],
    ),
    Scenario(
        name="checkout-and-pay",
        request="Checkout and pay for my order.",
        intent="checkout_payment",
        channel="telegram",
        describes="Checkout settles the oldest unpaid order with a simulated payment.",
        checks=[lambda r: check_path(r, "checkout_payment"), check_checkout, check_a2a_chain],
    ),
    Scenario(
        name="product-advice",
        request="Find me wireless earbuds under $120 with good battery life.",
        intent="product_advice",
        describes="The original Day 3 flow: the full five-agent recommendation pipeline.",
        checks=[lambda r: check_path(r, "product_advice"), check_full_advice_path, check_a2a_chain],
    ),
]

SCENARIOS_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


# ---------------------------------------------------------------------------
# chained scenario: buy, then pay for what you just bought
# ---------------------------------------------------------------------------
def run_buy_then_checkout(dry_run: bool, verbose: bool) -> tuple[str, list[Check], dict[str, Any]]:
    """Two turns on one order: purchase creates the draft, checkout settles it."""
    first = run_agentmart("I want to buy this AM-WCH-3001.", dry_run=dry_run, channel="telegram")
    draft_id = first.get("draft_order", {}).get("order_id")

    second = run_agentmart(
        f"Checkout and pay for order {draft_id}.", dry_run=dry_run, channel="telegram"
    )
    receipt = second.get("payment_receipt", {})

    checks = [
        expect("turn 1 created a draft order", bool(draft_id)),
        expect("turn 1 left it unpaid", first.get("draft_order", {}).get("status") == "awaiting_payment"),
        expect("turn 2 targeted the same order", receipt.get("order_id") == draft_id),
        expect("turn 2 captured the payment", receipt.get("status") == "captured"),
        expect("payment flagged simulated", receipt.get("simulated") is True),
        expect(
            "total is the catalog price plus shipping ($149.00 + $3.50)",
            receipt.get("amount_usd") == 152.5,
        ),
    ]
    if draft_id:
        persisted = get_order(draft_id)
        checks.append(expect("order book shows it paid", persisted["status"] == "paid"))

    # The two turns are separate A2A tasks, so they must NOT share a correlation id.
    ids = {hop["correlation_id"] for hop in first.get("a2a_log", [])} | {
        hop["correlation_id"] for hop in second.get("a2a_log", [])
    }
    checks.append(expect("each turn is its own A2A task (2 correlation ids)", len(ids) == 2))

    return "buy-then-checkout", checks, second


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def print_detail(result: dict[str, Any]) -> None:
    print("    A2A hops:")
    for hop in result.get("a2a_log", []):
        print(
            f"      {hop['state']:<11} {hop['sender']} -> {hop['recipient']}"
            f"  ({hop['intent']})"
        )
    print("    agent replies:")
    for entry in result.get("transcript", []):
        message = " ".join(entry["message"].split())
        print(f"      [{entry['agent']}] {message[:160]}")


# Routing regression. A remote agent phrases a request very differently from a
# person: long, hedged, and stating its own guardrails inline. Those guardrails name
# the capability they forbid, so "do not capture payment" once routed straight to the
# Payment Agent and settled an unrelated order. These are the real strings a Hermes
# peer sent over A2A, plus the short human phrasings they must not break.
ROUTING_CASES: tuple[tuple[str, str], ...] = (
    # Short human phrasings -- the original contract.
    ("What is my order status?", "order_status"),
    ("Where is my order AM-ORD-20260912-0002?", "order_status"),
    ("List me the available products.", "browse_catalog"),
    ("I want to buy this AM-EAR-1002.", "purchase_intent"),
    ("Checkout and pay for my order.", "checkout_payment"),
    ("Find me wireless earbuds under $120 with good battery life.", "product_advice"),
    ("pay now", "checkout_payment"),
    # A question *about* payment stays a status read and must never charge.
    ("has my payment gone through?", "order_status"),
    # Prohibitions must not be read as instructions.
    ("Create a draft order for AM-EAR-1002. Do not charge or capture payment; "
     "just confirm the draft order details and next checkout step.", "purchase_intent"),
    ("Please provide an order summary for AM-ORD-20260915-0003. Do not take any "
     "further payment, refund, or fulfillment action.", "order_status"),
    ("Please proceed to fulfillment for existing order AM-ORD-20260915-0003. Do not "
     "take any payment, authorization, capture, refund, or cancellation action.", "order_status"),
    # No keyword survives the strip -- the order id alone must keep it off product_advice.
    ("Please proceed to fulfillment for existing order AM-ORD-20260915-0003.", "order_status"),
    # "prepare a checkout draft" is drafting, never settling -- checkout must not win.
    ("The user wants to buy AM-EAR-1002 (Nimbus Air 2). Record the purchase intent "
     "and prepare a checkout draft for quantity 1. Do not charge, capture, or call "
     "the payment agent.", "purchase_intent"),
    # "where to buy" is advice; "wants to buy" is intent.
    ("Find wireless earbuds under $120 and include a link or where to buy.", "product_advice"),
    ("The customer wants to buy SKU AM-EAR-1002 (Nimbus Air 2).", "purchase_intent"),
)


def check_routing() -> list[Check]:
    """Every request routes where it should -- especially the ones that forbid an action."""
    checks: list[Check] = []
    for text, want in ROUTING_CASES:
        got = classify_intent(text)
        checks.append(expect(f"{want:16s} <- {text[:58]}", got == want))
    return checks


def run_scenario(scenario: Scenario, dry_run: bool, verbose: bool) -> list[Check]:
    result = run_agentmart(
        scenario.request,
        channel=scenario.channel,
        dry_run=dry_run,
        customer_id=scenario.customer_id,
    )
    checks: list[Check] = []
    for check in scenario.checks:
        checks.extend(check(result))
    if verbose:
        print_detail(result)
    return checks


def report(name: str, request: str, describes: str, checks: list[Check]) -> bool:
    failed = [label for ok, label in checks if not ok]
    mark = "PASS" if not failed else "FAIL"
    print(f"\n[{mark}] {name}")
    print(f"    request : {request}")
    print(f"    checks  : {len(checks) - len(failed)}/{len(checks)} passed")
    if failed:
        print(f"    why     : {describes}")
        for label in failed:
            print(f"      x {label}")
    return not failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Hermes + A2A scenario suite against the AgentMart agents."
    )
    parser.add_argument("-s", "--scenario", action="append", help="Run only this scenario (repeatable).")
    parser.add_argument("--list", action="store_true", help="List scenario names and exit.")
    parser.add_argument("--live", action="store_true", help="Call OpenRouter instead of dry-run.")
    parser.add_argument("--verbose", action="store_true", help="Print the A2A log and agent replies.")
    parser.add_argument("--no-reseed", action="store_true", help="Do not reset the order book between scenarios.")
    args = parser.parse_args()

    if args.list:
        for scenario in SCENARIOS:
            print(f"{scenario.name:<24} {scenario.intent:<17} {scenario.request}")
        print(f"{'buy-then-checkout':<24} {'(chained)':<17} purchase a SKU, then settle that order")
        print(f"{'intent-routing':<24} {'(routing)':<17} 14 phrasings, human and agent-generated")
        return 0

    dry_run = not args.live
    selected = SCENARIOS
    run_chained = True
    if args.scenario:
        names = set(args.scenario)
        unknown = names - set(SCENARIOS_BY_NAME) - {"buy-then-checkout", "intent-routing"}
        if unknown:
            print(f"Unknown scenario(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"Available: {', '.join(SCENARIOS_BY_NAME)}, buy-then-checkout, intent-routing", file=sys.stderr)
            return 2
        selected = [s for s in SCENARIOS if s.name in names]
        run_chained = "buy-then-checkout" in names

    mode = "LIVE (OpenRouter)" if args.live else "dry-run (no model calls)"
    print(f"AgentMart scenario suite — {mode}")
    print("Payments are simulated; no payment processor is contacted.")

    results: list[bool] = []
    if not args.scenario or "intent-routing" in set(args.scenario):
        results.append(
            report(
                "intent-routing",
                f"{len(ROUTING_CASES)} phrasings, human and agent-generated",
                "Prohibitions are not instructions; a payment question is not a payment.",
                check_routing(),
            )
        )

    for scenario in selected:
        if not args.no_reseed:
            seed()  # each scenario starts from the same order book
        try:
            checks = run_scenario(scenario, dry_run, args.verbose)
        except Exception as exc:  # noqa: BLE001 - a crash is a scenario failure, not a stack trace
            checks = [expect(f"scenario raised {type(exc).__name__}: {exc}", False)]
        results.append(report(scenario.name, scenario.request, scenario.describes, checks))

    if run_chained:
        if not args.no_reseed:
            seed()
        try:
            name, checks, result = run_buy_then_checkout(dry_run, args.verbose)
            if args.verbose:
                print_detail(result)
        except Exception as exc:  # noqa: BLE001
            name, checks = "buy-then-checkout", [expect(f"scenario raised {type(exc).__name__}: {exc}", False)]
        results.append(
            report(
                name,
                "I want to buy this AM-WCH-3001. -> Checkout and pay for order <id>.",
                "Two turns on one order id: draft then settle.",
                checks,
            )
        )

    if not args.no_reseed:
        seed()  # leave the lab in its seeded state

    passed = sum(results)
    print(f"\n{'=' * 60}")
    print(f"{passed}/{len(results)} scenarios passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
