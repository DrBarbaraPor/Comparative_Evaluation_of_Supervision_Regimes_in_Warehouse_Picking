from __future__ import annotations

import pandas as pd

from wms_rpt.cleaning import _can_hold_a_hash, clean_tables
from wms_rpt.schema import SENTINEL_HASHES

SENT = "23C94A5C710DD8ACA6976C0D48474DDF9245EF20FE60433F311CADFE2CF42A9B"
REAL = "D" * 64
SLOT = "E" * 64


def _slot_history():
    return pd.DataFrame({
        "slotid": [SLOT] * 3,
        "itemid": [SENT, REAL, SENT],
        "purchaseorderid": [SENT, "PO1", SENT],
        "latestevent": ["SlotUnassigned", "QuantityDeducted", "SlotUnassigned"],
        "tslatestevent": pd.to_datetime(
            ["2025-12-01 10:00", "2025-12-01 10:05", "2025-12-01 10:10"]),
    })


# --- defect: dropping the rows would have lost 4.6% of slot_history; the
#     unassignment events are real, only the item reference is meaningless.
def test_sentinel_rows_survive():
    clean, _ = clean_tables({"slot_history": _slot_history()})
    assert len(clean["slot_history"]) == 3
    assert clean["slot_history"]["latestevent"].tolist() == [
        "SlotUnassigned", "QuantityDeducted", "SlotUnassigned"]


# --- defect: a frequency-based rule would have nulled a real but rarely
#     ordered SKU. Only crosswalk absence plus no downstream activity counts.
def test_real_rare_sku_untouched():
    df = pd.DataFrame({"sku": [REAL] * 2, "shippingintentline_id": ["L1", "L2"]})
    clean, _ = clean_tables({"shippingintent_sku": df})
    assert clean["shippingintent_sku"]["sku"].notna().all()


# --- defect: the blanket sweep must not touch columns that cannot hold a
#     64-char hash. Nulling a numeric or datetime column would move a median
#     and trip the cleaning notebook's verify assertion.
def test_sweep_leaves_non_id_columns_alone():
    clean, _ = clean_tables({"slot_history": _slot_history()})
    d = clean["slot_history"]
    assert d["tslatestevent"].notna().all()
    assert d["slotid"].notna().all()


# --- defect: the cleartext re-pull made the placeholder a genuine null, so the
#     set is empty and the rule is inert. It stays in the pipeline against a
#     future extract that reintroduces placeholder values. The emptiness is
#     the assertion — a non-empty set means someone refilled it without
#     recording why.
def test_sentinel_set_is_empty_and_immutable():
    assert isinstance(SENTINEL_HASHES, frozenset)
    assert SENTINEL_HASHES == frozenset(), (
        "SENTINEL_HASHES was refilled; see D93 before changing this test")


# --- defect: the column gate read `dtype == "object"`. pandas 3.0 (PDEP-14)
#     made `str` the default dtype for text, so that test became False for
#     every string column and the sweep silently stopped acting — no error, no
#     warning, no line in the cleaning log. The gate asks what a column CANNOT
#     be instead. Pinned per dtype so the next default change fails here.
def test_hash_gate_accepts_every_text_dtype():
    assert _can_hold_a_hash(pd.Series(["a"], dtype=object).dtype)
    assert _can_hold_a_hash(pd.Series(["a"]).dtype)          # str on pandas 3
    assert _can_hold_a_hash(pd.Series(["a"], dtype="string").dtype)
    assert _can_hold_a_hash(pd.Series(["a"]).astype("category").dtype)


def test_hash_gate_rejects_dtypes_that_cannot_hold_one():
    for s in (pd.Series([1]), pd.Series([1.0]), pd.Series([1], dtype="Int64"),
              pd.Series([True]), pd.to_datetime(pd.Series(["2025-12-01"])),
              pd.Series([pd.Timedelta("1h")])):
        assert not _can_hold_a_hash(s.dtype), s.dtype


# --- the sweep must reach a plain text column whatever pandas calls its dtype
def test_sentinel_rule_fires_on_default_string_dtype(monkeypatch):
    monkeypatch.setattr("wms_rpt.cleaning.SENTINEL_HASHES", frozenset({SENT}))
    df = pd.DataFrame({"itemid": [SENT, REAL], "quantity": [1, 2]})
    clean, log = clean_tables({"slot_history": df})
    d = clean["slot_history"]
    assert d["itemid"].isna().tolist() == [True, False], (
        "sweep skipped a text column — check the dtype gate, not the rule")
    assert (log["action"] == "null_sentinel").any(), "action went unlogged"
    assert d["quantity"].tolist() == [1, 2]


# --- the rule itself must still work when given sentinels
def test_sentinel_rule_still_fires_when_configured(monkeypatch):
    monkeypatch.setattr("wms_rpt.cleaning.SENTINEL_HASHES", frozenset({SENT}))
    clean, _ = clean_tables({"slot_history": _slot_history()})
    d = clean["slot_history"]
    assert d["itemid"].isna().tolist() == [True, False, True]
    assert d["purchaseorderid"].isna().tolist() == [True, False, True]
    assert d.loc[1, "itemid"] == REAL