"""Regression tests for wms_rpt.tasks - one per defect I hit building it.

The fixtures here use plaintext on both sides, which is now also what the real
data looks like (D86). When I wrote them the WMS side was hashed and the cobot
side was not, so the crosswalk framing in the comments below is history rather
than current practice - but the normalization problems they describe are all
still live.
"""

from __future__ import annotations

import pandas as pd

from wms_rpt.tasks import identify_fields


# --- defect: identify_fields came back `attribute` for all 24 fields, which
#     is the kind of result that looks like the function is broken and is
#     actually three separate bugs stacked. ref_specs was hardcoded to the
#     hashed WMS columns, so the plaintext domains were never even built;
#     numeric columns skipped domain matching entirely, so float-typed itemno
#     was only ever considered as a quantity; and nothing tried normalization
#     variants, so `217589.0` never met `217589` and `S523365` never met
#     `523365`. Fixing any one of them alone would still have returned
#     `attribute` for everything.
def test_identify_fields_matches_across_normalizations():
    tasks = pd.DataFrame({"itemno": [217589.0, 306105.0],
                          "_slot": ["S523365", "S516293"],
                          "execuser": ["5341", "Temp35"]})
    refs = {"items":   pd.DataFrame({"itemid": ["217589", "306105"]}),
            "slots":   pd.DataFrame({"slotid": ["523365", "516293"]}),
            "workers": pd.DataFrame({"badge": ["5341", "Temp35"]})}
    out = identify_fields(tasks, refs).set_index("column")["identified_as"]
    assert out["itemno"] == "items.itemid_ref"
    assert out["_slot"] == "slots.slotid_ref"
    assert out["execuser"] == "workers.badge_ref"


# --- defect: blanket casefolding cut the execuser match from 125/126 down to
#     89, because badges like `Temp35` carry case. Normalization variants have
#     to be scored per (column, domain) pair rather than applied across the
#     board. Same 36 Temp badges that the reader later ate (D95) - they are a
#     standing hazard, not a one-off.
def test_case_bearing_badges_survive_normalization():
    tasks = pd.DataFrame({"execuser": ["Temp35", "Temp36", "5341"]})
    refs = {"workers": pd.DataFrame({"badge": ["Temp35", "Temp36", "5341"]})}
    out = identify_fields(tasks, refs).set_index("column")["identified_as"]
    assert out["execuser"] == "workers.badge_ref"


# --- defect: passing `picking` whole made every column a candidate domain,
#     and small sequential integers (orderlineid, _task_seq) matched
#     `quantitypicked` at 90%+ by range coincidence. The fix is at the caller:
#     only ID columns are passed as domains. This asserts the contract that
#     makes that fix work — domains come from the columns supplied, nothing
#     else, so restricting the frame restricts the candidate set.
def test_only_supplied_columns_become_domains():
    tasks = pd.DataFrame({"orderlineid": list(range(1, 21))})
    picking = pd.DataFrame({"pickticket_id": [f"PT{i}" for i in range(20)],
                            "quantitypicked": list(range(1, 21))})
    # whole frame: the coincidental match is reachable
    wide = identify_fields(tasks, {"picking": picking})
    assert wide.set_index("column").loc["orderlineid", "identified_as"] \
        == "picking.quantitypicked_ref"
    # ID columns only: it is not
    narrow = identify_fields(tasks, {"picking": picking[["pickticket_id"]]})
    assert narrow.set_index("column").loc["orderlineid", "identified_as"] \
        == "attribute"