"""Regression tests for wms_divergence (wms_rpt 1.5.0).

STATUS: the label these test is retired. wms_divergence needed the slot
crosswalk to reach (ticket, slot) grain, and the cleartext re-pull retired the
crosswalks (D100), so it emits identically zero on live data. It is out of
TARGETS and, since 1.6.0, out of LABEL_COLS as well (D138); the function and
these tests remain for the archived hashed extracts.

Keeping the file. The fixtures are synthetic so they still pass, and the defect
they encode - counting the WMS side on raw picking instead of the deduplicated
spine - is a mistake I could easily make again in another function. If the
label is ever revived these are the tests it needs.

The defect proper: wms_divergence must count WMS picks on the deduplicated
spine, not on raw ``picking``.

Defect. ``wms_divergence`` deduplicated the robot side on ``jobtaskid`` — the
docstring calls that dedupe mandatory, because summing re-emitted tasks
doubles every robot quantity — and then summed ``quantitypicked`` over raw
``picking``, which still contains the byte-identical duplicate pick lines the
SSMS export produces and ``build_spine`` exists to remove. The WMS side was
therefore doubled on any ticket whose rows were exported twice.

Observed on December wh1: 148 tickets carry duplicated rows. On the test
split, 45 tickets showed *every* line divergent while robot and WMS ticket
totals reconciled at exactly 1.000 and the slot crosswalk resolved at 100%.
Reading one ticket's comparison table showed WMS exactly 2x robot on all 18
slots — 2 vs 1, 4 vs 2, no exceptions.

The defect is all-or-nothing per ticket (a ticket is duplicated or it is
not), flat across ticket sizes, and invisible to every aggregate check that
sums after deduplication. Only the slot-grain side-by-side revealed it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wms_rpt.picks import wms_divergence

HASH = {f"S{i:06d}": f"H{i:06d}" for i in range(1, 20)}
XW = {"slot": {k.removeprefix("S"): v for k, v in HASH.items()}}


def _frames(duplicate_ticket: str | None = None):
    """Two tickets, three slots each, robot and WMS in exact agreement."""
    rows = []
    for tk in ("T1", "T2"):
        for i, q in enumerate([1, 2, 1], start=1):
            rows.append({"pickticket_id": tk, "slotid": f"H{i:06d}",
                         "quantitypicked": q, "quantity": q,
                         "tspicked": pd.Timestamp("2025-12-27 14:00") +
                         pd.Timedelta(minutes=i)})
    pk = pd.DataFrame(rows)

    tasks = pd.DataFrame([
        {"pickticket_id": tk, "jobtaskid": f"{tk}-{i}", "tasktype": "PICK",
         "tasklocation": f"S{i:06d}", "taskqty": q, "execqty": q}
        for tk in ("T1", "T2") for i, q in enumerate([1, 2, 1], start=1)])

    spine = pk.copy()
    if duplicate_ticket is not None:
        dup = pk[pk["pickticket_id"] == duplicate_ticket]
        pk = pd.concat([pk, dup], ignore_index=True)   # the export artefact
    return spine.reset_index(drop=True), pk.reset_index(drop=True), tasks


def test_agreement_when_no_duplicates():
    spine, pk, tasks = _frames()
    out = wms_divergence(spine, tasks, pk, XW)
    assert out["wms_divergence"].sum() == 0


def test_duplicated_export_rows_do_not_manufacture_divergence():
    """The defect proper: T1 exported twice must not flag every T1 line."""
    spine, pk, tasks = _frames(duplicate_ticket="T1")
    out = wms_divergence(spine, tasks, pk, XW)
    flagged = spine.loc[out["wms_divergence"] == 1, "pickticket_id"]
    assert len(flagged) == 0, (
        f"{len(flagged)} lines flagged on duplicated export rows; "
        "WMS side is being counted on raw picking, not spine")


def test_real_divergence_still_detected():
    """The fix must not blunt the label: a genuine mismatch still fires."""
    spine, pk, tasks = _frames()
    tasks.loc[tasks.index[0], "execqty"] = 9      # robot 9 vs WMS 1
    out = wms_divergence(spine, tasks, pk, XW)
    assert out["wms_divergence"].sum() == 1


def test_divergence_is_not_all_or_nothing_per_ticket():
    """All lines of a ticket flagging together is the defect's signature."""
    spine, pk, tasks = _frames(duplicate_ticket="T2")
    out = wms_divergence(spine, tasks, pk, XW)
    per_ticket = out["wms_divergence"].groupby(
        spine["pickticket_id"].to_numpy()).agg(["size", "sum"])
    full = per_ticket[(per_ticket["sum"] == per_ticket["size"])
                      & (per_ticket["size"] > 1)]
    assert full.empty, f"whole tickets flagged: {list(full.index)}"


def test_zero_exec_stockout_is_not_divergence():
    """execqty=0 with no pick line is a stockout, excluded by the >0 guard."""
    spine, pk, tasks = _frames()
    tasks = pd.concat([tasks, pd.DataFrame([{
        "pickticket_id": "T1", "jobtaskid": "T1-9", "tasktype": "PICK",
        "tasklocation": "S000009", "taskqty": 1, "execqty": 0}])],
        ignore_index=True)
    out = wms_divergence(spine, tasks, pk, XW)
    assert out["wms_divergence"].sum() == 0


def test_missing_crosswalk_emits_zero_not_error():
    spine, pk, tasks = _frames()
    out = wms_divergence(spine, tasks, pk, {"slot": {}})
    assert (out["wms_divergence"] == 0).all()
