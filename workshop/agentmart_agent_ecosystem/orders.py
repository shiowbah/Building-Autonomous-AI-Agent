"""Read and write helpers over the seeded AgentMart order book.

`catalog.py` covers the product side; this module covers the customer side:
customers, orders, order items, and payments. The Order and Payment agents call
in here so "what is my order status" and "checkout and pay" resolve against real
rows instead of invented ones.

Payments are SIMULATED. `authorize_payment` and `capture_payment` write rows to
the local SQLite database and generate a `sim_` reference. No payment processor
is ever contacted, no card number is stored, and no money moves.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from catalog import DB_PATH, CatalogNotSeededError

# Lifecycle the workshop uses. `awaiting_payment` is the state a draft order
# sits in until the Payment Agent captures it.
ORDER_STATUSES = (
    "draft",
    "awaiting_payment",
    "paid",
    "packed",
    "in_transit",
    "delivered",
    "cancelled",
)

OPEN_STATUSES = ("draft", "awaiting_payment", "paid", "packed", "in_transit")


class OrderBookNotSeededError(RuntimeError):
    """Raised when the order tables are missing or empty."""


class OrderNotFoundError(RuntimeError):
    """Raised when an order id does not exist in the seeded order book."""


def _connect(db_path=None) -> sqlite3.Connection:
    from pathlib import Path

    path = Path(db_path) if db_path else DB_PATH
    if not path.exists():
        raise OrderBookNotSeededError(
            f"Order book not found at {path}. Run: python seed_data.py"
        )
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("SELECT 1 FROM orders LIMIT 1").fetchone()
    except sqlite3.OperationalError as exc:  # table missing on an old database
        conn.close()
        raise OrderBookNotSeededError(
            f"Order tables missing in {path}. Run: python seed_data.py --reset"
        ) from exc
    _ensure_delivery_columns(conn)
    return conn


# Delivery-detail columns were added after the first seed; ALTER an old database
# in place so a draft can actually carry the address fields the customer gave us.
NON_SEED_DELIVERY_COLUMNS = (
    ("delivery_recipient", "TEXT"),
    ("delivery_address", "TEXT"),
    ("delivery_postal", "TEXT"),
    ("delivery_contact", "TEXT"),
)


def _ensure_delivery_columns(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"] for row in conn.execute("PRAGMA table_info(orders)").fetchall()
    }
    for name, kind in NON_SEED_DELIVERY_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE orders ADD COLUMN {name} {kind}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _eta_string(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).date().isoformat()


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------
def get_customer(customer_id: str, db_path=None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_payment_method(method_id: str, db_path=None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM payment_methods WHERE method_id = ?", (method_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _attach_detail(conn: sqlite3.Connection, order: dict[str, Any]) -> dict[str, Any]:
    items = conn.execute(
        "SELECT oi.sku, oi.quantity, oi.unit_price_usd, p.name, p.brand"
        " FROM order_items oi LEFT JOIN products p ON p.sku = oi.sku"
        " WHERE oi.order_id = ? ORDER BY oi.sku",
        (order["order_id"],),
    ).fetchall()
    order["items"] = [dict(item) for item in items]

    payments = conn.execute(
        "SELECT * FROM payments WHERE order_id = ? ORDER BY authorized_at",
        (order["order_id"],),
    ).fetchall()
    order["payments"] = [dict(payment) for payment in payments]
    order["amount_paid_usd"] = sum(
        payment["amount_usd"] for payment in order["payments"] if payment["status"] == "captured"
    )
    order["is_paid"] = order["amount_paid_usd"] >= order["total_usd"] - 0.001
    return order


def get_order(order_id: str, db_path=None) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        if row is None:
            raise OrderNotFoundError(f"No such order: {order_id}")
        return _attach_detail(conn, dict(row))
    finally:
        conn.close()


def list_orders(
    customer_id: str | None = None,
    status: str | None = None,
    open_only: bool = False,
    limit: int = 20,
    db_path=None,
) -> list[dict[str, Any]]:
    sql = ["SELECT * FROM orders WHERE 1 = 1"]
    params: list[Any] = []
    if customer_id:
        sql.append("AND customer_id = ?")
        params.append(customer_id)
    if status:
        sql.append("AND status = ?")
        params.append(status)
    if open_only:
        placeholders = ",".join("?" for _ in OPEN_STATUSES)
        sql.append(f"AND status IN ({placeholders})")
        params.extend(OPEN_STATUSES)
    sql.append("ORDER BY placed_at DESC LIMIT ?")
    params.append(limit)

    conn = _connect(db_path)
    try:
        rows = conn.execute(" ".join(sql), params).fetchall()
        return [_attach_detail(conn, dict(row)) for row in rows]
    finally:
        conn.close()


def find_payable_order(customer_id: str, db_path=None) -> dict[str, Any] | None:
    """The order a bare 'checkout and pay' should act on: oldest unpaid one."""
    candidates = [
        order
        for order in list_orders(customer_id=customer_id, db_path=db_path)
        if order["status"] in ("draft", "awaiting_payment") and not order["is_paid"]
    ]
    return sorted(candidates, key=lambda o: o["placed_at"])[0] if candidates else None


def find_open_draft(customer_id: str, sku: str, db_path=None) -> dict[str, Any] | None:
    """The customer's newest unpaid draft that already contains ``sku``.

    Lets a "prepare / refresh the checkout draft" turn update the same draft
    instead of minting a second order every time the peer re-quotes.
    """
    for order in list_orders(customer_id=customer_id, db_path=db_path):
        items = order.get("items", [])
        if order["status"] in ("draft", "awaiting_payment") and not order["is_paid"]:
            if any(item.get("sku") == sku for item in items):
                return order
    return None


# --------------------------------------------------------------------------
# writes
# --------------------------------------------------------------------------
def create_draft_order(
    customer_id: str,
    items: list[dict[str, Any]],
    warehouse: str | None = None,
    fulfillment_method: str | None = None,
    shipping_usd: float = 0.0,
    eta_date: str | None = None,
    delivery: dict[str, str] | None = None,
    db_path=None,
) -> dict[str, Any]:
    """Create an `awaiting_payment` order from [{sku, quantity}] using catalog prices.

    When a warehouse + fulfillment method is given, the real shipping cost and
    ETA are taken from the seeded ``fulfillment_options`` table so the draft
    always matches the catalog quote; ``shipping_usd`` is only a fallback if no
    matching option exists. ``delivery`` carries the customer-supplied slots
    (recipient, address, postal, contact) so a draft update actually persists.
    """
    if not items:
        raise ValueError("create_draft_order requires at least one item")

    conn = _connect(db_path)
    try:
        shipping = round(float(shipping_usd), 2)
        eta = eta_date
        if warehouse and fulfillment_method:
            option = conn.execute(
                "SELECT cost_usd, eta_days FROM fulfillment_options"
                " WHERE warehouse = ? AND method = ?",
                (warehouse, fulfillment_method),
            ).fetchone()
            if option:
                shipping = round(float(option["cost_usd"]), 2)
                if eta is None:
                    eta = _eta_string(int(option["eta_days"]))

        priced: list[dict[str, Any]] = []
        for item in items:
            sku = item["sku"]
            row = conn.execute("SELECT price_usd FROM products WHERE sku = ?", (sku,)).fetchone()
            if row is None:
                raise CatalogNotSeededError(f"SKU not in the seeded catalog: {sku}")
            priced.append(
                {
                    "sku": sku,
                    "quantity": int(item.get("quantity", 1)),
                    "unit_price_usd": float(row["price_usd"]),
                }
            )

        subtotal = round(sum(i["unit_price_usd"] * i["quantity"] for i in priced), 2)
        delivery = delivery or {}
        now = _now()
        order_id = f"AM-ORD-{datetime.now(timezone.utc):%Y%m%d}-{uuid4().hex[:4].upper()}"
        order = {
            "order_id": order_id,
            "customer_id": customer_id,
            "status": "awaiting_payment",
            "placed_at": now,
            "updated_at": now,
            "warehouse": warehouse,
            "fulfillment_method": fulfillment_method,
            "tracking_ref": None,
            "eta_date": eta,
            "subtotal_usd": subtotal,
            "shipping_usd": shipping,
            "total_usd": round(subtotal + shipping, 2),
        }
        for key in ("recipient", "address", "postal", "contact"):
            order[f"delivery_{key}"] = (delivery or {}).get(key)

        with conn:
            conn.execute(
                "INSERT INTO orders (order_id, customer_id, status, placed_at, updated_at,"
                " warehouse, fulfillment_method, tracking_ref, eta_date,"
                " subtotal_usd, shipping_usd, total_usd,"
                " delivery_recipient, delivery_address, delivery_postal, delivery_contact)"
                " VALUES (:order_id, :customer_id, :status, :placed_at, :updated_at,"
                " :warehouse, :fulfillment_method, :tracking_ref, :eta_date,"
                " :subtotal_usd, :shipping_usd, :total_usd,"
                " :delivery_recipient, :delivery_address, :delivery_postal, :delivery_contact)",
                order,
            )
            conn.executemany(
                "INSERT INTO order_items (order_id, sku, quantity, unit_price_usd)"
                " VALUES (:order_id, :sku, :quantity, :unit_price_usd)",
                [{"order_id": order_id, **item} for item in priced],
            )
        return _attach_detail(conn, dict(order))
    finally:
        conn.close()


DELIVERY_SLOTS = ("recipient", "address", "postal", "contact")


def update_draft_delivery(
    order_id: str, delivery: dict[str, str], db_path=None
) -> dict[str, Any]:
    """Persist the customer-supplied delivery slots onto an open draft.

    Only slots actually supplied are written; any slot not in ``delivery`` is
    left as it was, so a partial update never blanks data the customer already
    gave. Raises ``OrderNotFoundError`` if there is no open draft for that id.
    """
    conn = _connect(db_path)
    try:
        slots = {key: delivery.get(key) for key in DELIVERY_SLOTS if delivery.get(key)}
        if not slots:
            raise ValueError("update_draft_delivery requires at least one delivery slot")
        assignments = ", ".join(f"delivery_{key} = :delivery_{key}" for key in slots)
        with conn:
            cursor = conn.execute(
                f"UPDATE orders SET updated_at = :updated_at, {assignments}"
                " WHERE order_id = :order_id AND status IN ('draft', 'awaiting_payment')",
                {
                    "updated_at": _now(),
                    "order_id": order_id,
                    **{f"delivery_{key}": value for key, value in slots.items()},
                },
            )
        if cursor.rowcount == 0:
            raise OrderNotFoundError(f"No open draft: {order_id}")
    finally:
        conn.close()
    return get_order(order_id, db_path=db_path)


def set_order_status(order_id: str, status: str, db_path=None) -> dict[str, Any]:
    if status not in ORDER_STATUSES:
        raise ValueError(f"Unknown order status: {status}")
    conn = _connect(db_path)
    try:
        with conn:
            cursor = conn.execute(
                "UPDATE orders SET status = ?, updated_at = ? WHERE order_id = ?",
                (status, _now(), order_id),
            )
        if cursor.rowcount == 0:
            raise OrderNotFoundError(f"No such order: {order_id}")
    finally:
        conn.close()
    return get_order(order_id, db_path=db_path)


def authorize_payment(
    order_id: str,
    method_id: str | None = None,
    db_path=None,
) -> dict[str, Any]:
    """Simulate a payment authorization. Writes a local row; charges nothing."""
    order = get_order(order_id, db_path=db_path)
    if order["is_paid"]:
        raise ValueError(f"Order {order_id} is already paid in full")

    if method_id is None:
        customer = get_customer(order["customer_id"], db_path=db_path)
        method_id = customer["default_payment_method"] if customer else None

    payment = {
        "payment_id": f"PAY-{datetime.now(timezone.utc):%Y%m%d}-{uuid4().hex[:4].upper()}",
        "order_id": order_id,
        "method_id": method_id,
        "status": "authorized",
        "amount_usd": order["total_usd"],
        "authorized_at": _now(),
        "captured_at": None,
        "processor_ref": f"sim_auth_{uuid4().hex[:6]}",
    }
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO payments (payment_id, order_id, method_id, status, amount_usd,"
                " authorized_at, captured_at, processor_ref)"
                " VALUES (:payment_id, :order_id, :method_id, :status, :amount_usd,"
                " :authorized_at, :captured_at, :processor_ref)",
                payment,
            )
    finally:
        conn.close()
    return payment


def capture_payment(payment_id: str, db_path=None) -> dict[str, Any]:
    """Simulate capturing an authorized payment and move the order to `paid`."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM payments WHERE payment_id = ?", (payment_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No such payment: {payment_id}")
        payment = dict(row)
        if payment["status"] == "captured":
            return payment
        if payment["status"] != "authorized":
            raise ValueError(f"Payment {payment_id} is {payment['status']}, not authorized")

        captured_at = _now()
        with conn:
            conn.execute(
                "UPDATE payments SET status = 'captured', captured_at = ? WHERE payment_id = ?",
                (captured_at, payment_id),
            )
        payment.update(status="captured", captured_at=captured_at)
    finally:
        conn.close()

    set_order_status(payment["order_id"], "paid", db_path=db_path)
    return payment


def checkout_and_pay(order_id: str, method_id: str | None = None, db_path=None) -> dict[str, Any]:
    """Authorize then capture in one step. Simulated end to end."""
    authorized = authorize_payment(order_id, method_id=method_id, db_path=db_path)
    captured = capture_payment(authorized["payment_id"], db_path=db_path)
    return {"payment": captured, "order": get_order(order_id, db_path=db_path)}


# --------------------------------------------------------------------------
# formatting for agent prompts
# --------------------------------------------------------------------------
def format_order(order: dict[str, Any]) -> str:
    items = "\n".join(
        f"      {item['quantity']} x {item['sku']}"
        f" {item.get('name') or '(unknown SKU)'} @ ${item['unit_price_usd']:.2f}"
        for item in order.get("items", [])
    ) or "      (no line items)"
    payments = ", ".join(
        f"{payment['status']} ${payment['amount_usd']:.2f} ({payment['processor_ref']})"
        for payment in order.get("payments", [])
    ) or "none"
    delivery_slots = {
        "recipient": order.get("delivery_recipient"),
        "address": order.get("delivery_address"),
        "postal": order.get("delivery_postal"),
        "contact": order.get("delivery_contact"),
    }
    recorded_delivery = ", ".join(
        f"{key} {value}"
        for key, value in delivery_slots.items()
        if value is not None and str(value).strip()
    )
    return (
        f"- {order['order_id']} | status {order['status']}"
        f" | placed {order['placed_at']} | updated {order['updated_at']}\n"
        f"    total ${order['total_usd']:.2f}"
        f" (subtotal ${order['subtotal_usd']:.2f} + shipping ${order['shipping_usd']:.2f})"
        f" | paid ${order.get('amount_paid_usd', 0):.2f}\n"
        f"    fulfillment: {order['fulfillment_method'] or 'not selected'}"
        f" from {order['warehouse'] or 'unassigned'}"
        f" | tracking {order['tracking_ref'] or 'none'}"
        f" | eta {order['eta_date'] or 'tbd'}\n"
        f"    delivery: {recorded_delivery or 'not provided'}\n"
        f"    payments: {payments}\n"
        f"    items:\n{items}"
    )


def format_orders(orders: list[dict[str, Any]]) -> str:
    if not orders:
        return "(no orders found for this customer)"
    return "\n".join(format_order(order) for order in orders)


def format_order_book(db_path=None) -> str:
    """Whole seeded order book, grouped by customer — used by `seed_data.py --list-orders`."""
    conn = _connect(db_path)
    try:
        customers = [dict(r) for r in conn.execute("SELECT * FROM customers ORDER BY customer_id")]
    finally:
        conn.close()

    blocks = []
    for customer in customers:
        orders = list_orders(customer_id=customer["customer_id"], db_path=db_path)
        blocks.append(
            f"{customer['customer_id']} | {customer['name']}"
            f" | {customer['channel']} {customer['channel_handle']}"
            f" | pays with {customer['default_payment_method']}\n"
            f"{format_orders(orders)}"
        )
    return "\n\n".join(blocks)


def order_book_stats(db_path=None) -> dict[str, int]:
    conn = _connect(db_path)
    try:
        counts = {}
        for table in ("customers", "payment_methods", "orders", "order_items", "payments"):
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if counts["orders"] == 0:
            raise OrderBookNotSeededError("Order book is empty. Run: python seed_data.py")
        return counts
    finally:
        conn.close()
