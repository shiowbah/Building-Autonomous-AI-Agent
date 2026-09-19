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

from agentmart_ecosystem import INTENT_PATHS, classify_intent, extract_delivery_details, run_agentmart
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
        expect("priced from the catalog ($89.00, standard delivery is free)", draft.get("total_usd") == 89.0),
        expect("nothing was charged", draft.get("amount_paid_usd", 0) == 0),
        expect("Payment Agent did NOT run", "payment_agent" not in agents_visited(result)),
    ]
    if order_id:
        persisted = get_order(order_id)
        checks.append(expect("draft is persisted in the order book", persisted["status"] == "awaiting_payment"))
    return checks


def check_checkout_blocks_missing_delivery(result: dict[str, Any]) -> list[Check]:
    """A bare checkout must stop when the target order has no delivery details."""
    receipt = result.get("payment_receipt", {})
    return [
        expect("an order was selected to settle", bool(receipt.get("order_id"))),
        expect("payment was NOT captured (blocked)", receipt.get("status") == "blocked"),
        expect("the gate says delivery details are missing", bool(receipt.get("missing_delivery_slots"))),
        expect("no simulated capture reference was issued", "processor_ref" not in receipt),
        expect("the order is still unpaid in the order book", bool(receipt.get("order_id")) and not get_order(receipt["order_id"])["is_paid"]),
    ]


def check_checkout(result: dict[str, Any]) -> list[Check]:
    receipt = result.get("payment_receipt", {})
    order_id = receipt.get("order_id")
    checks = [
        expect("an order was selected to settle", bool(order_id)),
        expect("payment captured", receipt.get("status") == "captured"),
        expect("payment is flagged simulated", receipt.get("simulated") is True),
        expect("reference is a simulated one", str(receipt.get("processor_ref", "")).startswith("sim_")),
        expect("customer's default method was used", receipt.get("method_id") == "PM-VISA-4417"),
    ]
    if order_id:
        persisted = get_order(order_id)
        checks += [
            expect("order moved to paid in the order book", persisted["status"] == "paid"),
            expect("order reads as fully paid", persisted["is_paid"] is True),
        ]
    return checks


def check_read_only_verify(result: dict[str, Any]) -> list[Check]:
    return [
        expect("payment agent was NOT woken", "payment_agent" not in agents_visited(result)),
        expect("no payment receipt was produced", not result.get("payment_receipt")),
        expect("the named order is in the Order Agent's context", "AM-ORD-20260915-0003" in result.get("order_context", "")),
        expect("nothing was charged (order untouched)", bool(result.get("order_context"))),
    ]


def check_buy_intent_drafts(result: dict[str, Any]) -> list[Check]:
    return [
        expect("a draft order materialised for the named SKU", bool(result.get("draft_order") and result.get("draft_order", {}).get("order_id"))),
        expect("the draft targets the named product", "AM-EAR-1002" in result.get("order_context", "")),
        expect("delivery details were asked for", bool(result.get("missing_delivery_slots"))),
        expect("payment agent was NOT woken", "payment_agent" not in agents_visited(result)),
        expect("no payment receipt was produced", not result.get("payment_receipt")),
        expect("nothing was charged", result.get("amount_paid", 0) == 0),
    ]


def check_checkout_inline_delivery(result: dict[str, Any]) -> list[Check]:
    """A checkout turn carrying unlabelled delivery details settles the right order."""
    receipt = result.get("payment_receipt", {})
    checks = [
        expect("routes through the payment agent", "payment_agent" in agents_visited(result)),
        expect("the target order contains the named SKU", "AM-EAR-1002" in result.get("order_context", "")),
        expect("no delivery slot was left missing", not result.get("missing_delivery_slots")),
        expect("payment was captured", receipt.get("status") == "captured"),
        expect("no charge was blocked", not result.get("checkout_blocked_reason")),
    ]
    if receipt.get("order_id"):
        checks.append(expect("receipt order == target order", receipt.get("order_id") == result.get("target_order_id")))
        try:
            persisted = get_order(receipt["order_id"])
            checks.append(expect("delivery persisted from the inline message", persisted.get("delivery_recipient") == "Ang Chin Tiong"
                           and persisted.get("delivery_address") == "3 Pine Grove"))
        except Exception:  # noqa: BLE001 - the failed expectation is reported below
            checks.append(expect("order row was readable", False))
    return checks


def check_intent_only(result: dict[str, Any]) -> list[Check]:
    return [
        expect("no draft order materialised", not result.get("draft_order")),
        expect("intent-only flag set for the Order Agent prompt", result.get("intent_only_recorded") is True),
        expect("payment agent was NOT woken", "payment_agent" not in agents_visited(result)),
        expect("no payment receipt was produced", not result.get("payment_receipt")),
    ]


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
        describes="A bare checkout is refused until the target order has complete delivery details.",
        checks=[lambda r: check_path(r, "checkout_payment"), check_checkout_blocks_missing_delivery, check_a2a_chain],
    ),
    Scenario(
        name="read-only-verify",
        request=(
            "Read-only verification only; do not modify, cancel, refund, pay, or create anything. "
            "Verify whether order AM-ORD-20260915-0003 contains AM-EAR-1002 (Nimbus Air 2), and "
            "report its canonical item(s), status, amount, and whether the reported simulated "
            "payment was real."
        ),
        intent="order_status",
        describes="A read-only verification is a lookup: it must wake only the Order Agent and never charge.",
        checks=[lambda r: check_path(r, "order_status"), check_read_only_verify, check_no_error_text],
    ),
    Scenario(
        name="buy-intent-record",
        request=(
            "Tell AgentMart: the user wants to buy AM-EAR-1002 (Nimbus Air 2), quantity 1. "
            "Record this as purchase intent only. Do not create or modify an order, reserve stock, "
            "authorize payment, capture funds, or use any prior order. Confirm only that the intent "
            "was recorded."
        ),
        intent="purchase_intent",
        describes=(
            "Even a guardrail-y 'record intent only, do not create an order' wrapper still names a "
            "SKU: the customer wants to buy, so a real draft must materialise and delivery details "
            "must be asked for -- no payment, no reservation."
        ),
        checks=[lambda r: check_path(r, "purchase_intent"), check_buy_intent_drafts, check_no_error_text],
    ),
    Scenario(
        name="intent-only-no-sku",
        request=(
            "Record this as purchase intent only: the customer wants to buy a laptop but "
            "has not chosen a specific model yet. Do not create or modify an order, reserve "
            "stock, authorize payment, capture funds, or use any prior order. Confirm only "
            "that the intent was recorded."
        ),
        intent="purchase_intent",
        describes=(
            "Without a concrete SKU there is nothing to draft: a pure intent-only request must "
            "leave the order book untouched and never charge."
        ),
        checks=[lambda r: check_path(r, "purchase_intent"), check_intent_only, check_no_error_text],
    ),
    Scenario(
        name="checkout-with-delivery-inline",
        request=(
            "Checkout and pay for exactly 1 \u00d7 Nimbus Air 2 (SKU AM-EAR-1002), using the "
            "current purchase intent. Use delivery details: Ang Chin Tiong, 3 Pine Grove, "
            "Singapore 597590, contact number 97492736, standard delivery. Verify the cart "
            "contains only AM-EAR-1002 and report the final total and order ID. If required "
            "delivery or payment details are unavailable, stop without charging. Do not use or "
            "modify any other order."
        ),
        intent="checkout_payment",
        describes=(
            "Hermes sends the delivery slots inline and unlabelled ('Use delivery details: "
            "Ang Chin Tiong, 3 Pine Grove, Singapore 597590, contact number 97492736'). The "
            "checkout turn must persist them onto the Nimbus draft and settle it."
        ),
        checks=[lambda r: check_path(r, "checkout_payment"), check_checkout_inline_delivery, check_no_error_text],
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
def run_buy_then_checkout(dry_run: bool, verbose: bool) -> tuple[list[Check], dict[str, Any]]:
    """Two turns on one order: purchase (with delivery details) creates the draft, then checkout settles it."""
    first = run_agentmart(
        ("I want to buy this AM-WCH-3001. recipient 'Wei Ling Tan',"
         " address '3 Pine Grove', postal 597590, contact '97492736'."),
        dry_run=dry_run,
        channel="telegram",
    )
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
            "total is the catalog price, standard delivery is free ($149.00)",
            receipt.get("amount_usd") == 149.0,
        ),
    ]
    if draft_id:
        persisted = get_order(draft_id)
        checks.append(expect("order book shows it paid", persisted["status"] == "paid"))
        checks.append(
            expect("delivery details persisted on the draft",
                   persisted.get("delivery_recipient") == "Wei Ling Tan"
                   and persisted.get("delivery_postal") == "597590"),
        )

    # The two turns are separate A2A tasks, so they must NOT share a correlation id.
    ids = {hop["correlation_id"] for hop in first.get("a2a_log", [])} | {
        hop["correlation_id"] for hop in second.get("a2a_log", [])
    }
    checks.append(expect("each turn is its own A2A task (2 correlation ids)", len(ids) == 2))

    return checks, second


def run_draft_refresh(dry_run: bool, verbose: bool) -> tuple[list[Check], dict[str, Any]]:
    """Re-quoting a draft reused the same open draft and never charges."""
    first = run_agentmart("I want to buy this AM-EAR-1002.", dry_run=dry_run, channel="telegram")
    draft_id = first.get("draft_order", {}).get("order_id")

    second = run_agentmart(
        ("Update the checkout draft with the user-provided delivery details: address line "
         "'3 Pine Grove', recipient 'Ang Chin Tiong', contact number '97492736', postal code "
         "597590, Singapore. Keep quantity 1 and standard delivery as the current method unless "
         "unavailable. Do not place the order, reserve stock, authorize payment, or capture "
         "funds. Return the updated total and any remaining required confirmation."),
        dry_run=dry_run,
        channel="telegram",
    )

    checks = [
        expect("turn 1 created a draft order", bool(draft_id)),
        expect("turn 1 left it unpaid", first.get("draft_order", {}).get("status") == "awaiting_payment"),
        expect("turn 2 routed to drafting, not checkout_payment", second.get("intent") == "purchase_intent"),
        expect("turn 2 reused the same draft id", second.get("draft_order", {}).get("order_id") == draft_id),
        expect("turn 2 did NOT run the payment agent", "payment_agent" not in agents_visited(second)),
        expect("no payment receipt was produced", not second.get("payment_receipt")),
    ]
    if draft_id:
        persisted = get_order(draft_id)
        checks.append(expect("the draft is still awaiting_payment", persisted["status"] == "awaiting_payment"))
        checks.append(
            expect(
                "delivery slots persisted on the draft",
                (
                    persisted.get("delivery_recipient") == "Ang Chin Tiong"
                    and persisted.get("delivery_address") == "3 Pine Grove"
                    and persisted.get("delivery_postal") == "597590"
                    and persisted.get("delivery_contact") == "97492736"
                ),
            )
        )
        checks.append(
            expect(
                "no required delivery slot is outstanding after the full update",
                not second.get("missing_delivery_slots"),
            )
        )

    return checks, second


def run_draft_needs_fields(dry_run: bool, verbose: bool) -> tuple[list[Check], dict[str, Any]]:
    """Partial delivery details persist, name what is still missing, then clear."""
    first = run_agentmart(
        "I want to buy this AM-EAR-1002. recipient 'Ang Chin Tiong', address '3 Pine Grove'.",
        dry_run=dry_run,
        channel="telegram",
    )
    draft_id = first.get("draft_order", {}).get("order_id")

    missing_first = sorted(first.get("missing_delivery_slots") or [])

    second = run_agentmart(
        ("Update the draft order with the remaining delivery details: postal code "
         "597590, contact number '97492736'. Do not place the order, authorize "
         "payment, or capture funds."),
        dry_run=dry_run,
        channel="telegram",
    )

    checks = [
        expect("turn 1 created a draft order", bool(draft_id)),
        expect("turn 1 persisted the slots that were given", bool(first.get("delivery_details"))),
        expect(
            "turn 1 named the missing required slots",
            sorted(first.get("missing_delivery_slots") or []) == ["contact", "postal"],
        ),
    ]
    if draft_id:
        first_persisted = get_order(draft_id)
        checks.append(
            expect(
                "partial slots were saved on the draft",
                first_persisted.get("delivery_recipient") == "Ang Chin Tiong"
                and first_persisted.get("delivery_address") == "3 Pine Grove",
            )
        )
        checks.append(
            expect(
                "the still-missing slots were not fabricated",
                missing_first == ["contact", "postal"],
            )
        )
        checks.append(
            expect("turn 2 routed to drafting, not checkout_payment", second.get("intent") == "purchase_intent"),
        )
        checks.append(
            expect("turn 2 reused the same draft id", second.get("draft_order", {}).get("order_id") == draft_id),
        )
        checks.append(
            expect("turn 2 did NOT run the payment agent", "payment_agent" not in agents_visited(second)),
        )
        checks.append(
            expect(
                "turn 2 cleared the outstanding slots",
                not second.get("missing_delivery_slots"),
            )
        )
        second_persisted = get_order(draft_id)
        checks.append(
            expect(
                "full delivery details are now persisted",
                second_persisted.get("delivery_recipient") == "Ang Chin Tiong"
                and second_persisted.get("delivery_address") == "3 Pine Grove"
                and second_persisted.get("delivery_postal") == "597590"
                and second_persisted.get("delivery_contact") == "97492736",
            )
        )
        checks.append(
            expect(
                "no payment was captured while updating",
                first.get("draft_order", {}).get("status") == "awaiting_payment"
                and second_persisted["status"] == "awaiting_payment",
            )
        )

    return checks, second


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
    # Updating/refreshing a checkout draft is drafting, never settling -- even though
    # it literally says "checkout draft", the payment path must not fire.
    ("Update the checkout draft with the user-provided delivery details: address "
     "line '3 Pine Grove', recipient 'Ang Chin Tiong', contact number '97492736', "
     "postal code 597590, Singapore. Keep quantity 1 and standard delivery as the "
     "current method unless unavailable. Do not place the order, reserve stock, "
     "authorize payment, or capture funds. Return the updated total and any "
     "remaining required confirmation.", "purchase_intent"),
    ("Refresh the checkout draft for AM-EAR-1002 with quantity 1 and standard "
     "delivery. Do not charge or authorize payment.", "purchase_intent"),
    # "where to buy" is advice; "wants to buy" is intent.
    ("Find wireless earbuds under $120 and include a link or where to buy.", "product_advice"),
    ("The customer wants to buy SKU AM-EAR-1002 (Nimbus Air 2).", "purchase_intent"),
    # Read-only verification / reconciliation / audit phrasings are LOOKUPS. They
    # mention payment ids and orders but must never wake the Payment Agent.
    ("Read-only verification only; do not modify, cancel, refund, pay, or create anything. "
     "Verify whether order AM-ORD-20260915-0003 contains AM-EAR-1002 (Nimbus Air 2), and "
     "report its canonical item(s), status, amount, and whether the reported simulated "
     "payment was real.", "order_status"),
    ("Read-only order-status lookup: check the user's current order status, especially order "
     "AM-ORD-20260915-0003 from the previous Nimbus Air 2 checkout. Do not create, modify, "
     "cancel, refund, or place any order.", "order_status"),
    ("Read-only reconciliation for order AM-ORD-20260915-0003: the prior lookup returned "
     "payment ID PAY-20260918-B085, while an earlier lookup returned PAY-20260918-E10D. "
     "Verify the single canonical order record and report the current order status. Do not "
     "create, modify, cancel, refund, or place anything.", "order_status"),
    ("Read-only verification only. Do not modify, pay, cancel, refund, or create anything. "
     "Verify whether the updated draft AM-ORD-20260918-2AC7 contains the user-provided "
     "address and Nimbus Air 2, and report its canonical status. Also verify whether the "
     "newly reported order AM-ORD-20260915-0003 / payment PAY-20260918-AA8D is a real "
     "external charge or only a simulated local record.", "order_status"),
    ("Read-only verification for order AM-ORD-20260915-0003: item SKU(s), amount, "
     "fulfillment, and whether it is the requested AM-EAR-1002. Do not modify, cancel, "
     "refund, pay, or create anything.", "order_status"),
    # A GLOBAL CHECKOUT that verifies the cart before paying is still a checkout and
    # must keep charging (the verification-only rule must not hijack it).
    ("Checkout and pay for exactly 1 x Nimbus Air 2 (SKU AM-EAR-1002) for customer "
     "CUST-1001, using the provided delivery details: 3 Pine Grove, Singapore 597590; "
     "recipient Ang Chin Tiong; contact number 97492736; standard delivery. Before payment, "
     "verify the cart contains only this SKU and return the final item price, shipping, "
     "taxes/fees, total, and payment method identifier. Do not substitute items.", "checkout_payment"),
    # Recording intent only, with an express ban on creating orders, still routes to
    # purchase_intent; the Order Agent then decides draft/no-draft from whether a SKU
    # is named (a named SKU means the customer wants to buy, so it drafts and asks
    # for delivery; only a SKU-less intent stays purely recorded).
    ("Tell AgentMart: the user wants to buy AM-EAR-1002 (Nimbus Air 2), quantity 1. Record "
     "this as purchase intent only. Do not create or modify an order, reserve stock, "
     "authorize payment, capture funds, or use any prior order. Confirm only that the intent "
     "was recorded.", "purchase_intent"),
    # Recording intent WHILE asking to prepare a checkout draft still drafts.
    ("The user wants to buy AM-EAR-1002 (Nimbus Air 2). Record the purchase intent and "
     "prepare a checkout draft for quantity 1 using the previously provided destination "
     "(Singapore 597590). Do not place the order, reserve stock, authorize payment, or "
     "capture funds. Return the draft status and any missing checkout details.", "purchase_intent"),
    # A peer correctness message ("correct and persist the delivery address on draft X")
    # is an UPDATE, not a read-only lookup that falls back on the order id. It must
    # route to purchase_intent even though the SKU/value words are embedded mid-sentence.
    ("Correct and persist the delivery address on draft AM-ORD-20260919-4F77. Use exactly: "
     "3 pine grove, astor green, singapore. Keep the previously provided recipient name ang "
     "chin tiong, postal code 597590, contact number 97492736, SKU AM-EAR-1002, quantity 1, "
     "and standard delivery. Do not authorize payment, capture funds, or place the order. "
     "Return the updated draft state.", "purchase_intent"),
    # Verify/read-back messages about a draft are LOOKUPS, even when they reference an
    # order id and mention address changes, and must NOT ride the purchase_intent path
    # (which would re-persist junk extracted from "after the address update").
    ("Verify the exact draft order AM-ORD-20260919-4F77 after the address update. Read back "
     "its current status, SKU, quantity, delivery method, total, payment status, inventory "
     "reservation status, and any taxes/fees or remaining missing details. Do not modify, "
     "authorize, capture, or place it.", "order_status"),
    ("Verify the exact draft order AM-ORD-20260919-4F77 by looking it up. Return its current "
     "status, SKU, quantity, payment status, total, and whether inventory is reserved. Do not "
     "modify, authorize, capture, or place anything.", "order_status"),
    ("Verify draft order AM-ORD-20260919-4F77 after this address update. Read back the exact "
     "current status, SKU, quantity, delivery details, total, ETA, payment status, and inventory "
     "reservation status. Do not modify, authorize, capture, or place anything.", "order_status"),
    # A read-back phrased without the word "verify" is still a LOOKUP: it must outrank the
    # weak "payment" marker, or the checkout branch would persist junk parsed from the
    # lookup's own list of field names ("recipient name, full delivery address, postal code,
    # contact number, SKU ...").
    ("Perform a fresh order/draft lookup for AM-ORD-20260919-23BE and read back the stored "
     "recipient name, full delivery address, postal code, contact number, SKU, quantity, total, "
     "payment status, inventory reservation status, fulfillment method, and order placement "
     "status. Do not modify anything.", "order_status"),
    # A correction that quotes the earlier read-only result is still an UPDATE: the quoted
    # "read-only lookup shows ..." must not downgrade it, and "with exactly:" must survive
    # strip_prohibitions even after a parenthesised "do not create a duplicate".
    ("The read-only lookup shows recipient name, postal code, and contact number are still not "
     "stored. Please update the existing AM-ORD-20260919-23BE only (do not create a duplicate) "
     "with exactly: ang chin tiong; 3 pine grove singapore; 597590; 97492736. Do not authorize "
     "payment, capture funds, reserve inventory, or place the order. Return whether each field "
     "was persisted and the resulting status.", "purchase_intent"),
)


def check_routing() -> list[Check]:
    """Every request routes where it should -- especially the ones that forbid an action."""
    checks: list[Check] = []
    for text, want in ROUTING_CASES:
        got = classify_intent(text)
        checks.append(expect(f"{want:16s} <- {text[:58]}", got == want))
    return checks


UNLABELLED_DELIVERY = (
    "Checkout and pay for exactly 1 \u00d7 Nimbus Air 2 (SKU AM-EAR-1002), using the current "
    "purchase intent. Use delivery details: Ang Chin Tiong, 3 Pine Grove, Singapore 597590, "
    "contact number 97492736, standard delivery. Verify the cart contains only AM-EAR-1002 and "
    "report the final total and order ID. If required delivery or payment details are "
    "unavailable, stop without charging. Do not use or modify any other order."
)


def check_extract() -> list[Check]:
    """Delivery slots parse from Hermes's labelled AND unlabelled phrasings."""
    unlabelled = extract_delivery_details(UNLABELLED_DELIVERY)
    labelled = extract_delivery_details(
        "I want to buy this AM-EAR-1002. recipient 'Ang Chin Tiong', address "
        "'3 Pine Grove', postal 597590, contact '97492736'."
    )
    expected = {
        "recipient": "Ang Chin Tiong",
        "address": "3 Pine Grove",
        "postal": "597590",
        "contact": "97492736",
    }
    update_msg = extract_delivery_details(
        "Update draft order AM-ORD-20260919-4F77 with these delivery details exactly as "
        "provided: recipient name: ang chin tiong; full delivery address: 3 pine grove; "
        "postal code: 597590; contact number: 97492736."
    )
    correct_msg = extract_delivery_details(
        "Correct and persist the delivery address on draft AM-ORD-20260919-4F77. Use exactly: "
        "3 pine grove, astor green, singapore. Keep the previously provided recipient name ang "
        "chin tiong, postal code 597590, contact number 97492736, SKU AM-EAR-1002, quantity 1, "
        "and standard delivery."
    )
    verify_msg = extract_delivery_details(
        "Verify the exact draft order AM-ORD-20260919-4F77 after the address update. Read back "
        "its current status, SKU, quantity, delivery method, total, payment status."
    )
    fullname_msg = extract_delivery_details(
        "Update the staged checkout draft/order AM-ORD-20260919-23BE with these delivery details "
        "exactly as provided: recipient full name: ang chin tiong; full delivery address: 3 pine "
        "grove singapore; postal code: 597590; contact number: 97492736. Do not authorize payment, "
        "capture funds, reserve inventory, or place the order."
    )
    positional_msg = extract_delivery_details(
        "The read-only lookup shows recipient name, postal code, and contact number are still not "
        "stored. Please update the existing AM-ORD-20260919-23BE only (do not create a duplicate) "
        "with exactly: ang chin tiong; 3 pine grove singapore; 597590; 97492736. Do not authorize "
        "payment, capture funds, reserve inventory, or place the order."
    )
    update_expected = {
        "recipient": "ang chin tiong",
        "address": "3 pine grove",
        "postal": "597590",
        "contact": "97492736",
    }
    fullname_expected = {
        "recipient": "ang chin tiong",
        "address": "3 pine grove singapore",
        "postal": "597590",
        "contact": "97492736",
    }
    return [
        expect("unlabelled 'Use delivery details:' phrasing parses all four slots", unlabelled == expected),
        expect("labelled recipient/address/postal/contact phrasing parses all four slots", labelled == expected),
        expect("colon-labelled 'recipient name:'/address phrasing parses cleanly", update_msg == update_expected),
        expect("'correct and persist ... on draft' extracts the real slots", correct_msg == update_expected),
        expect("a pure verify lookup extracts nothing to persist", verify_msg == {}),
        expect("'recipient full name:' labelled phrasing parses cleanly", fullname_msg == fullname_expected),
        expect("'with exactly:' positional list parses cleanly despite a leading quote", positional_msg == fullname_expected),
    ]


def run_update_then_verify(dry_run: bool, verbose: bool) -> tuple[list[Check], dict[str, Any]]:
    """The live failure: an update lands, then a read-back must not clobber it.

    Reproduces the exact Hermes phrasing that corrupted AM-ORD-20260919-4F77 (the
    verify reply now routes to order_status and extracts nothing, so the address the
    update persisted is left untouched).
    """
    buy = run_agentmart("I want to buy this AM-EAR-1002.", dry_run=dry_run, channel="telegram")
    draft_id = buy.get("draft_order", {}).get("order_id")

    update = run_agentmart(
        ("Update draft order {0} with these delivery details exactly as provided: "
         "recipient name: ang chin tiong; full delivery address: 3 pine grove; "
         "postal code: 597590; contact number: 97492736. Keep quantity 1, SKU "
         "AM-EAR-1002, and standard delivery. Do not place the order, authorize "
         "payment, or capture funds.").format(draft_id or "AM-ORD-TBD"),
        dry_run=dry_run,
        channel="telegram",
    )

    verify = run_agentmart(
        ("Verify the exact draft order {0} after the address update. Read back its "
         "current status, SKU, quantity, delivery method, total, payment status, "
         "inventory reservation status, and any taxes/fees or remaining missing "
         "details. Do not modify, authorize, capture, or place it.").format(draft_id or "AM-ORD-TBD"),
        dry_run=dry_run,
        channel="telegram",
    )

    checks = [
        expect("turn 1 created a draft order", bool(draft_id)),
        expect("turn 2 routed to drafting, not checkout_payment", update.get("intent") == "purchase_intent"),
        expect("turn 3 verify routed to a read-only lookup", verify.get("intent") == "order_status"),
        expect("turn 3 did not run the payment agent", "payment_agent" not in agents_visited(verify)),
        expect("turn 3 did not draft or refresh anything", not verify.get("draft_order")),
    ]
    if draft_id:
        persisted = get_order(draft_id)
        checks.append(
            expect(
                "delivery slots persisted on the draft by the update",
                (
                    persisted.get("delivery_recipient") == "ang chin tiong"
                    and persisted.get("delivery_address") == "3 pine grove"
                    and persisted.get("delivery_postal") == "597590"
                    and persisted.get("delivery_contact") == "97492736"
                ),
            )
        )
    return checks, verify


def run_quote_correct_and_reupdate(dry_run: bool, verbose: bool) -> tuple[list[Check], dict[str, Any]]:
    """The full live failure loop: update -> lookup (misroute hazard) -> re-update.

    A read-back that named its own fields once corrupted the order via the checkout
    branch; a re-update that quoted the read-only result then got downgraded and never
    persisted. Both must now be inert or effective respectively.
    """
    buy = run_agentmart("I want to buy this AM-EAR-1002.", dry_run=dry_run, channel="telegram")
    draft_id = buy.get("draft_order", {}).get("order_id")
    oid = draft_id or "AM-ORD-TBD"

    update = run_agentmart(
        ("Update the staged checkout draft/order {0} with these delivery details exactly as "
         "provided: recipient full name: ang chin tiong; full delivery address: 3 pine grove "
         "singapore; postal code: 597590; contact number: 97492736. Do not authorize payment, "
         "capture funds, reserve inventory, or place the order.").format(oid),
        dry_run=dry_run,
        channel="telegram",
    )

    lookup = run_agentmart(
        ("Perform a fresh order/draft lookup for {0} and read back the stored recipient name, "
         "full delivery address, postal code, contact number, SKU, quantity, total, payment "
         "status, inventory reservation status, fulfillment method, and order placement status. "
         "Do not modify anything.").format(oid),
        dry_run=dry_run,
        channel="telegram",
    )

    before = get_order(draft_id) if draft_id else {}
    recipient_before = before.get("delivery_recipient")
    address_before = before.get("delivery_address")

    reupdate = run_agentmart(
        ("The read-only lookup shows recipient name, postal code, and contact number are still "
         "not stored. Please update the existing {0} only (do not create a duplicate) with "
         "exactly: ang chin tiong; 3 pine grove singapore; 597590; 97492736. Do not authorize "
         "payment, capture funds, reserve inventory, or place the order. Return whether each "
         "field was persisted and the resulting status.").format(oid),
        dry_run=dry_run,
        channel="telegram",
    )
    after = get_order(draft_id) if draft_id else {}

    checks = [
        expect("turn 1 created a draft order", bool(draft_id)),
        expect("turn 2 routed to drafting, not checkout_payment", update.get("intent") == "purchase_intent"),
        expect("turn 3 lookup stayed read-only (not checkout_payment)", lookup.get("intent") == "order_status"),
        expect("turn 3 did not run the payment agent", "payment_agent" not in agents_visited(lookup)),
        expect("the lookup did not clobber the persisted recipient", recipient_before == "ang chin tiong"),
        expect(
            "the lookup did not clobber the persisted address",
            address_before == "3 pine grove singapore",
        ),
        expect("turn 4 re-update routed to drafting despite the quoted 'read-only'", reupdate.get("intent") == "purchase_intent"),
        expect("turn 4 did not run the payment agent", "payment_agent" not in agents_visited(reupdate)),
    ]
    if draft_id:
        checks.append(
            expect(
                "all four slots persisted intact after the full sequence",
                (
                    after.get("delivery_recipient") == "ang chin tiong"
                    and after.get("delivery_address") == "3 pine grove singapore"
                    and after.get("delivery_postal") == "597590"
                    and after.get("delivery_contact") == "97492736"
                ),
            )
        )
    return checks, reupdate


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
        print(f"{'draft-refresh':<24} {'(chained)':<17} re-quote a draft: reused, never charged")
        print(f"{'draft-needs-fields':<24} {'(chained)':<17} partial delivery: ask, then complete the draft")
        print(f"{'update-then-verify':<24} {'(chained)':<17} update a draft, then verify it read-only")
        print(f"{'quote-correct-and-reupdate':<24} {'(chained)':<17} update -> lookup -> quoted correction")
        print(f"{'intent-routing':<24} {'(routing)':<17} phrasings, human and agent-generated")
        return 0

    dry_run = not args.live
    selected = SCENARIOS
    run_chained: set[str] = set()
    if args.scenario:
        names = set(args.scenario)
        known_chained = {"buy-then-checkout", "draft-refresh", "draft-needs-fields", "update-then-verify", "quote-correct-and-reupdate"}
        unknown = names - set(SCENARIOS_BY_NAME) - known_chained - {"intent-routing"}
        if unknown:
            print(f"Unknown scenario(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            print(
                f"Available: {', '.join(SCENARIOS_BY_NAME)}, "
                f"{', '.join(known_chained)}, intent-routing",
                file=sys.stderr,
            )
            return 2
        selected = [s for s in SCENARIOS if s.name in names]
        run_chained = names & known_chained

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
        results.append(
            report(
                "delivery-extract",
                "labelled and unlabelled delivery phrasings",
                "Hermes and a human label their delivery slots differently; both must land on the order.",
                check_extract(),
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

    chained = [
        (
            "buy-then-checkout",
            run_buy_then_checkout,
            "I want to buy this AM-WCH-3001. -> Checkout and pay for order <id>.",
            "Two turns on one order id: draft then settle.",
        ),
        (
            "draft-refresh",
            run_draft_refresh,
            "Buy AM-EAR-1002 -> update the checkout draft with delivery details.",
            "Re-quoting a draft reuses the draft and never charges.",
        ),
        (
            "draft-needs-fields",
            run_draft_needs_fields,
            "Buy AM-EAR-1002 -> complete the delivery details.",
            "Missing delivery slots are named, then a full update clears them.",
        ),
        (
            "update-then-verify",
            run_update_then_verify,
            "Buy AM-EAR-1002 -> update address -> verify (must not clobber).",
            "A read-back after an address update stays read-only.",
        ),
        (
            "quote-correct-and-reupdate",
            run_quote_correct_and_reupdate,
            "Buy AM-EAR-1002 -> update -> lookup -> quoted correction.",
            "A quoted read-only result must not downgrade the corrective update.",
        ),
    ]
    for name, fn, request, describes in chained:
        if not args.scenario or name in run_chained:
            if not args.no_reseed:
                seed()
            try:
                chained_checks, result = fn(dry_run, args.verbose)
                if args.verbose:
                    print_detail(result)
                results.append(report(name, request, describes, chained_checks))
            except Exception as exc:  # noqa: BLE001
                results.append(
                    report(
                        name,
                        request,
                        describes,
                        [expect(f"scenario raised {type(exc).__name__}: {exc}", False)],
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
