"""Regression tests for wms_rpt.picks

If one of these fails after a change, the change put back a bug that already
cost me a debugging round once. That is the convention across the whole
suite.

Fixtures use cleartext identifiers throughout. They used to use 64-char
hash-shaped values because that is what the July extracts held. The cleartext
re-pull (D86) made those unrepresentative, and made several tests pass for 
the wrong reason.
"""

from __future__ import annotations

import pandas as pd
import pytest

from wms_rpt.picks import (
    attach_cks, build_labels, build_spine, line_grain_cobot, slot_state,
    stockout_events,
)

# D97/D100: the spine carries an 8-char merchandising SKU with a 2-letter
# category prefix, not a 64-char hash. sku_to_itemid strips the prefix to the
# 6-digit itemid the cobot payload supplies as ItemNo.
H1 = "FR115400"
H2 = "FR115401"
S1 = "1"          # D100: cleartext slot code, not a hash
PT = "8480B7DD-B6BD-49BD-B399-7DB6E2CA073E"


def _picking(n_dup_exact=0):
    rows = [
        dict(shippingintentline_id="L1", pickticket_id=PT,
             tspicked=pd.Timestamp("2025-12-01 10:00"), slotid=S1,
             quantity=1, quantitypicked=1),
        dict(shippingintentline_id="L2", pickticket_id=PT,
             tspicked=pd.Timestamp("2025-12-01 10:05"), slotid=S1,
             quantity=2, quantitypicked=2),
    ]
    rows += [dict(rows[0]) for _ in range(n_dup_exact)]
    return pd.DataFrame(rows)


def _map():
    return pd.DataFrame({"shippingintentline_id": ["L1", "L2"],
                         "shippingintent_nbr": ["N1", "N1"]})


def _bridge():
    return pd.DataFrame({"shippingintentline_id": ["L1", "L2"],
                         "sku": [H1, H2]})


# --- defect: merging picking against a map that shares its duplicate line ids
#     cross-joined and inflated 657,540 -> 668,496 rows.
def test_duplicated_map_does_not_cross_join():
    m = pd.concat([_map(), _map()], ignore_index=True)   # map itself duplicated
    s, rep = build_spine(_picking(), m, _bridge())
    assert len(s) == 2
    assert rep.n_exact_dupes_dropped == 0


# --- defect: byte-identical pick lines are an SSMS export artifact.
def test_exact_duplicate_lines_dropped_and_counted():
    s, rep = build_spine(_picking(n_dup_exact=3), _map(), _bridge())
    assert len(s) == 2
    assert rep.n_exact_dupes_dropped == 3


def test_genuine_multi_row_lines_are_kept():
    p = _picking()
    extra = dict(p.iloc[0])
    extra["slotid"] = "2"                          # same line, different slot
    extra["tspicked"] = pd.Timestamp("2025-12-01 10:09")
    p = pd.concat([p, pd.DataFrame([extra])], ignore_index=True)
    s, rep = build_spine(p, _map(), _bridge())
    assert len(s) == 3
    assert rep.n_lines_multi_row == 2


# --- defect: the CK key is (shippingintent_nbr, sku); an earlier attempt
#     joined shippingintent_nbr against shippingintentline_id and got zero.
def test_ck_attaches_on_composite_key():
    s, _ = build_spine(_picking(), _map(), _bridge())
    ck = pd.DataFrame([dict(shippingintent_nbr="N1", sku=H2,
                            ck_ts=pd.Timestamp("2025-12-01 09:00"),
                            ck_found_none=0)])
    att, rep = attach_cks(s, ck)
    assert rep.ck_attached == 1
    assert s.loc[att._pickrow.iloc[0], "sku"] == H2


# --- defect: ~3% of CKs match several pick lines; tie-break must be the
#     nearest tspicked
def test_ambiguous_ck_takes_nearest_pick_and_is_counted():
    p = _picking()
    extra = dict(p.iloc[0])
    extra["tspicked"] = pd.Timestamp("2025-12-01 20:00")
    extra["slotid"] = "2"
    p = pd.concat([p, pd.DataFrame([extra])], ignore_index=True)
    s, _ = build_spine(p, _map(), _bridge())
    ck = pd.DataFrame([dict(shippingintent_nbr="N1", sku=H1,
                            ck_ts=pd.Timestamp("2025-12-01 09:55"),
                            ck_found_none=1)])
    att, rep = attach_cks(s, ck)
    assert rep.ck_ambiguous == 1
    assert s.loc[att._pickrow.iloc[0], "tspicked"] == pd.Timestamp("2025-12-01 10:00")


# --- defect: a total stockout leaves no failed pick line; the pick it
#     attaches to is the later recovery. It must not be a pick-grain anomaly.
def test_stockout_is_recovery_not_shortfall():
    s, _ = build_spine(_picking(), _map(), _bridge())
    ck = pd.DataFrame([
        dict(shippingintent_nbr="N1", sku=H1, ck_ts=pd.Timestamp("2025-12-01 04:00"),
             ck_found_none=1, slotid=S1, ck_quantity=5),
        dict(shippingintent_nbr="N1", sku=H2, ck_ts=pd.Timestamp("2025-12-01 04:00"),
             ck_found_none=0, slotid=S1, ck_quantity=5),
    ])
    att, _ = attach_cks(s, ck)
    lab, _ = build_labels(s, ck, att)
    assert lab["is_recovery_pick"].sum() == 1
    assert lab["ck_shortfall"].sum() == 1
    assert lab.loc[lab.index[0], "is_recovery_pick"] == 1
    assert len(stockout_events(ck)) == 1


# --- defect: an exact-timestamp slot_history match is the pick's own
#     inventory deduction. Including it leaks the outcome into the features.
def test_slot_state_excludes_the_picks_own_event():
    s, _ = build_spine(_picking(), _map(), _bridge())
    # Both spine SKUs get history. The fixture used to cover H1 only, so this
    # ran at a 50% match rate and tripped the low-coverage warning on every
    # run of the suite — a permanent warning is one you stop reading, which
    # is how the workers checksum survived at D95. The coverage guard has its
    # own test below; here it should stay quiet (D143).
    sh = pd.DataFrame([
        dict(slotid=S1, itemid=H1, tslatestevent=pd.Timestamp("2025-12-01 10:00"),
             physicalinventory=0, virtualinventory=0, skumaxcount=1),
        dict(slotid=S1, itemid=H1, tslatestevent=pd.Timestamp("2025-12-01 08:00"),
             physicalinventory=9, virtualinventory=9, skumaxcount=1),
        dict(slotid=S1, itemid=H2, tslatestevent=pd.Timestamp("2025-12-01 08:00"),
             physicalinventory=4, virtualinventory=4, skumaxcount=1),
    ])
    st, rep = slot_state(s, sh)
    assert st.loc[0, "onhand_before"] == 9        # not the 0 written by this pick
    assert rep.state_match_rate == pytest.approx(1.0)


# --- D123: the low-coverage guard. This ran at 4.5% for weeks while the
#     notebook printed "slot-state join coverage: 100.0%" — that print
#     measures column assignment onto the spine index, which is 100% whether
#     the values are real or null. The warning is the only place that knows
#     the truth, so it gets asserted rather than observed in passing.
def test_slot_state_warns_when_coverage_is_low():
    s, _ = build_spine(_picking(), _map(), _bridge())
    sh = pd.DataFrame([                      # H1 only: half the spine matches
        dict(slotid=S1, itemid=H1, tslatestevent=pd.Timestamp("2025-12-01 08:00"),
             physicalinventory=9, virtualinventory=9, skumaxcount=1),
    ])
    with pytest.warns(RuntimeWarning, match="matched only"):
        st, rep = slot_state(s, sh)
    assert rep.state_match_rate == pytest.approx(0.5)
    # the unmatched pick is null, not silently zero
    assert pd.isna(st.loc[1, "onhand_before"])


# --- defect: skuMaxCount is a SKU-slot capacity, so onhand/skuMaxCount
#     reaches 2000+ on Pallet slots. Normalize within size class instead.
def test_onhand_normalized_within_size_class():
    p = pd.DataFrame([
        dict(shippingintentline_id=f"L{i}", pickticket_id=PT,
             tspicked=pd.Timestamp("2025-12-01 10:00"), slotid=f"S{i}",
             quantity=1, quantitypicked=1) for i in range(4)])
    m = pd.DataFrame({"shippingintentline_id": [f"L{i}" for i in range(4)],
                      "shippingintent_nbr": ["N1"] * 4})
    b = pd.DataFrame({"shippingintentline_id": [f"L{i}" for i in range(4)],
                      "sku": [H1] * 4})
    s, _ = build_spine(p, m, b)
    sh = pd.DataFrame([
        dict(slotid=f"S{i}", itemid=H1,
             tslatestevent=pd.Timestamp("2025-12-01 08:00"),
             physicalinventory=v, virtualinventory=v, skumaxcount=1)
        for i, v in enumerate([1, 3, 1000, 3000])])
    sl = pd.DataFrame([dict(slotid=f"S{i}", size=sz, pickzonepk=1, volume=1.0,
                            skumaxcount=1)
                       for i, sz in enumerate(["Shelf", "Shelf", "Pallet", "Pallet"])])
    st, _ = slot_state(s, sh, sl)
    assert "fill_ratio" not in st.columns
    z = st["onhand_z_by_size"]
    # a 3000-unit Pallet is not an outlier; it sits with the other Pallets
    assert abs(z.iloc[3]) < 2
    assert st.loc[2, "slot_size"] == "Pallet"


# --- defect: labels have to attach at (ticket, item), not ticket. At ticket
#     grain a 3.9% rate fans out to 49% because every line inherits the flag.
#     Getting there used to need the item crosswalk; since the cleartext
#     re-pull both sides are 6-digit plaintext and the bridge is gone (D100).
#     The grain requirement is the part that survives.
def test_cobot_labels_attach_at_line_grain():
    s, _ = build_spine(_picking(), _map(), _bridge())
    xw = pd.DataFrame({"itemid": ["115400", "115401"], "hasheditemid": [H1, H2]})
    rt = pd.DataFrame([
        dict(pickticket_id=PT.lower(), itemno="115400.0", taskqty=5, execqty=3,
             exceptioncode="SHORT"),
        dict(pickticket_id=PT.lower(), itemno="115401.0", taskqty=2, execqty=2,
             exceptioncode=""),
    ])
    out = line_grain_cobot(s, rt, xw)
    assert out["did_mismatch"].tolist() == [1, 0]   # only the line that diverged
    assert out["exc_short"].tolist() == [1, 0]


# --- defect: itemno comes off a float-typed console column as "217589.0" and
#     never matched anything until I split on the dot. Still true after the
#     cleartext re-pull - the trailing .0 is a property of the payload, not
#     of the crosswalk I removed.
def test_itemno_float_string_is_normalized():
    s, _ = build_spine(_picking(), _map(), _bridge())
    xw = pd.DataFrame({"itemid": ["115400"], "hasheditemid": [H1]})
    rt = pd.DataFrame([dict(pickticket_id=PT, itemno="115400.0",
                            taskqty=5, execqty=1, exceptioncode="SHORT")])
    assert line_grain_cobot(s, rt, xw)["did_mismatch"].iloc[0] == 1


# --- defect: the cobot writes orderid as a lowercased pickticket_id. I
#     compared case-sensitively and got zero overlap on 42,385 tickets that
#     matched perfectly. Zero overlap is almost never a real finding.
def test_ticket_join_is_case_insensitive():
    s, _ = build_spine(_picking(), _map(), _bridge())
    xw = pd.DataFrame({"itemid": ["115400"], "hasheditemid": [H1]})
    rt = pd.DataFrame([dict(pickticket_id=PT.lower(), itemno="115400",
                            taskqty=5, execqty=1, exceptioncode="")])
    assert line_grain_cobot(s, rt, xw)["did_mismatch"].iloc[0] == 1


def test_labels_are_not_in_the_state_block():
    """Features and labels must stay disjoint — the leakage rule, asserted."""
    s, _ = build_spine(_picking(), _map(), _bridge())
    # Both SKUs covered: this test is about feature/label disjointness and has
    # nothing to say about join coverage, so it should not emit the coverage
    # warning as a side effect (D143).
    sh = pd.DataFrame([
        dict(slotid=S1, itemid=h, tslatestevent=pd.Timestamp("2025-12-01 08:00"),
             physicalinventory=9, virtualinventory=9, skumaxcount=1)
        for h in (H1, H2)])
    st, _ = slot_state(s, sh)
    ck = pd.DataFrame([dict(shippingintent_nbr="N1", sku=H1,
                            ck_ts=pd.Timestamp("2025-12-01 09:00"), ck_found_none=1)])
    att, _ = attach_cks(s, ck)
    lab, _ = build_labels(s, ck, att)
    assert not (set(st.columns) & set(lab.columns))


# ============================== crosswalks, context, rollup =================

from wms_rpt.picks import (  # noqa: E402
    _norm_location, cobot_line_features, rolling_context,
)
from wms_rpt.rollup import assign_windows, rollup, rollup_labels, drivers  # noqa: E402


# --- defect: lstrip('S') strips EVERY leading S, turning STATION001 into
#     TATION001. removeprefix is what's meant.
def test_location_prefix_uses_removeprefix():
    s = pd.Series(["S523365", "STATION001", "PACK01"])
    assert _norm_location(s).tolist() == ["523365", "TATION001", "PACK01"]
    # and the one that matters: a real slot code keeps all its digits
    assert _norm_location(pd.Series(["S107136"])).iloc[0] == "107136"


# --- cobot_line_features bridges nothing now (D100). Cobot TaskLocation with
#     the S stripped IS picking.slotid, 99.6% overlap. xw stays in the
#     signature so callers do not break; passing it empty here means the test
#     exercises what actually runs.
def test_told_location_divergence_is_line_grain():
    s, _ = build_spine(_picking(), _map(), _bridge())
    rt = pd.DataFrame([
        dict(pickticket_id=PT, itemno="115400", tasklocation="S1",
             taskqty=1, execqty=1, exceptioncode=""),
        dict(pickticket_id=PT, itemno="115401", tasklocation="S2",
             taskqty=2, execqty=2, exceptioncode=""),
    ])
    out = cobot_line_features(s, rt, xw={})
    # both picks came from slot 1; the second was told slot 2
    assert out["told_location_divergence"].tolist() == [0, 1]
    assert out["has_told_location"].tolist() == [1, 1]


def test_rolling_context_looks_backward_only():
    p = pd.DataFrame([
        dict(shippingintentline_id=f"L{i}", pickticket_id=PT,
             tspicked=pd.Timestamp("2025-12-01 10:00") + pd.Timedelta(minutes=i),
             slotid=S1, quantity=1, quantitypicked=1,
             picker_employee_id="W1") for i in range(4)])
    m = pd.DataFrame({"shippingintentline_id": [f"L{i}" for i in range(4)],
                      "shippingintent_nbr": ["N1"] * 4})
    b = pd.DataFrame({"shippingintentline_id": [f"L{i}" for i in range(4)],
                      "sku": [H1] * 4})
    s, _ = build_spine(p, m, b)
    ctx = rolling_context(s, windows_min=(5,))
    # the first pick sees nothing before it; the last sees the three prior
    assert ctx["picks_prev_5min"].iloc[0] == 0
    assert ctx["picks_prev_5min"].iloc[3] == 3
    assert pd.isna(ctx["secs_since_worker_prev_pick"].iloc[0])
    assert ctx["secs_since_worker_prev_pick"].iloc[1] == 60
    assert ctx["ordinal_in_ticket"].tolist() == [0, 1, 2, 3]


# --- 15-minute windows at 5-minute stride: each pick lands in exactly three.
def test_window_fanout_is_three_at_15_over_5():
    ts = pd.Series([pd.Timestamp("2025-12-01 10:07")])
    a = assign_windows(ts)
    assert len(a) == 3
    assert a["window_start"].tolist() == [
        pd.Timestamp("2025-12-01 09:55"), pd.Timestamp("2025-12-01 10:00"),
        pd.Timestamp("2025-12-01 10:05")]


def test_rollup_statistics_differ_as_designed():
    ts = pd.Series(pd.date_range("2025-12-01 10:00", periods=20, freq="30s"))
    sc = pd.Series([0.01] * 19 + [9.0])
    r = rollup(sc, ts, pick_threshold=1.0, topk=2)
    w = r.loc[r["n_picks"] == 20].iloc[0]
    assert w["max"] == 9.0
    assert w["mean"] < 0.5          # one anomaly is diluted
    assert w["topk_mean"] > w["mean"]
    assert w["count_over"] == 1


# --- defect: presence-based window labels gave a 96.3% window anomaly rate;
#     min_count=10 still gave 68.2%. Counting selects for volume, not anomaly.
#     Retained after 1.5.0 moved the default to method="binomial": the count
#     rule is still reachable and still behaves as documented, and this test
#     is what keeps the original failure demonstrable rather than folklore.
#     Renamed — it exercises the count rule at two thresholds and never
#     touches the proportion rule, which the old name claimed it did.
def test_count_rule_still_thresholds_as_documented():
    ts = pd.Series(pd.date_range("2025-12-01 10:00", periods=10, freq="30s"))
    lab = pd.DataFrame({"did_mismatch": [1] + [0] * 9})
    loose = rollup_labels(lab, ts, method="count", min_count=1)
    tight = rollup_labels(lab, ts, method="count", min_count=2)
    assert loose["did_mismatch"].max() == 1
    assert tight["did_mismatch"].max() == 0
    assert loose["did_mismatch_n"].max() == 1     # count kept for re-thresholding
    assert "did_mismatch_rate" in loose.columns   # share kept too


# --- defect: the proportional rule was adopted to stop min_count selecting
#     for volume, and inverted the same fault. A proportion ignores how much
#     evidence produced it: 1 event in 7 picks is a 14.3% share and clears
#     any multiple of a ~3% base, while a 400-pick window needs 36 events for
#     the same share. On the December wh1 test split this made the positive
#     class median 7 picks against 341 for the rest, and -n_picks alone
#     separated the classes at AUROC 0.963 — window size outperformed the
#     model it was meant to evaluate. 1.5.0 moved the default to an exact
#     binomial upper tail, which conditions on n.
#
#     This test previously asserted the quiet-but-concentrated window SHOULD
#     flag, under the name "volume invariant". It was neither.
def test_proportional_rule_is_volume_sensitive_binomial_is_not():
    busy = pd.date_range("2025-12-01 10:00", periods=100, freq="6s")
    quiet = pd.date_range("2025-12-01 12:00", periods=10, freq="60s")
    ts = pd.Series(list(busy) + list(quiet))
    v = [1] * 10 + [0] * 90 + [1] * 5 + [0] * 5      # 10% busy, 50% quiet
    lab = pd.DataFrame({"did_mismatch": v})
    base = {"did_mismatch": 0.03}

    prop = rollup_labels(lab, ts, method="proportion", base_rates=base,
                         rate_multiple=3.0).set_index("window_start")
    assert prop.loc[prop["n_picks"] < 20, "did_mismatch"].max() == 1, (
        "the proportional rule must still flag the 10-pick window — this is "
        "the behavior that motivated the binomial replacement")

    binom_ = rollup_labels(lab, ts, method="binomial", base_rates=base,
                           min_picks=5).set_index("window_start")
    small = binom_.loc[binom_["n_picks"] < 20, "did_mismatch"].dropna()
    large = binom_.loc[binom_["n_picks"] >= 50, "did_mismatch"].dropna()
    assert len(small) and len(large), "fixture must produce both window sizes"
    assert large.max() == 1, "100 picks at 10% against a 3% base is significant"
    assert binom_["n_picks"].max() >= 50


def test_rule_arguments_must_match_the_chosen_method():
    ts = pd.Series(pd.date_range("2025-12-01 10:00", periods=4, freq="1min"))
    lab = pd.DataFrame({"did_mismatch": [1, 0, 0, 0]})
    base = {"did_mismatch": 0.03}

    with pytest.raises(ValueError, match="does not use"):
        rollup_labels(lab, ts, method="binomial", base_rates=base, min_count=1)
    with pytest.raises(ValueError, match="does not use"):
        rollup_labels(lab, ts, method="count", min_count=1, rate_multiple=3.0)
    with pytest.raises(ValueError, match="requires min_count"):
        rollup_labels(lab, ts, method="count")
    with pytest.raises(ValueError, match="method must be one of"):
        rollup_labels(lab, ts, method="proportional")


# --- §3.2/§3.10.2: train on known-normal picks only, strictly defined.
def test_known_normal_excludes_any_labelled_pick():
    from wms_rpt.picks import known_normal
    lab = pd.DataFrame({
        "did_mismatch":             [0, 1, 0, 0],
        "ck_shortfall":             [0, 0, 1, 0],
        "told_location_divergence": [0, 0, 0, 1],
    })
    m = known_normal(lab)
    assert m.tolist() == [True, False, False, False]


# --- defect: told_location_divergence is emitted by cobot_line_features
#     beside genuine features, and leaked into the feature matrix.
# --- D138: wms_divergence needs the slot crosswalk D100 retired, so it emits
#     identically zero on live data. Leaving it in LABEL_COLS made every
#     caller filter it out by hand and made known_normal look stricter than it
#     is. If it ever comes back, it comes back with a working join.
def test_wms_divergence_is_not_a_label_column():
    from wms_rpt.picks import LABEL_COLS
    assert "wms_divergence" not in LABEL_COLS
    assert "did_mismatch" in LABEL_COLS


# --- D137: item_xw is vestigial, so it must not gate the primary outcome.
#     Gating on it meant None silently dropped did_mismatch and forced
#     callers to build an empty decoy frame to get past the check.
def test_did_mismatch_survives_a_null_item_xw():
    s, _ = build_spine(_picking(), _map(), _bridge())
    rt = pd.DataFrame([dict(pickticket_id=PT, itemno="115400.0",
                            taskqty=5, execqty=1, exceptioncode="SHORT")])
    ck = pd.DataFrame(columns=["shippingintent_nbr", "sku", "ck_ts",
                               "ck_found_none"])
    att, _ = attach_cks(s, ck)
    lab, _ = build_labels(s, ck, att, result_tasks=rt, item_xw=None)
    assert "did_mismatch" in lab.columns
    assert lab["did_mismatch"].iloc[0] == 1


def test_feature_columns_strips_labels():
    from wms_rpt.picks import feature_columns
    f = pd.DataFrame(columns=["onhand_before", "worker_fte",
                              "told_location_divergence", "did_mismatch"])
    assert feature_columns(f) == ["onhand_before", "worker_fte"]


def test_drivers_names_the_picks_behind_a_window():
    ts = pd.Series(pd.date_range("2025-12-01 10:00", periods=6, freq="1min"))
    sc = pd.Series([0.1, 0.2, 5.0, 0.1, 0.1, 0.1])
    d = drivers(sc, ts, pd.Timestamp("2025-12-01 10:00"), top=1)
    assert d["_row"].iloc[0] == 2
    assert d["score"].iloc[0] == 5.0


# ============================== schema resolution ===========================

from wms_rpt.schema import CROSSWALKS, KEY_ALIASES, PARENTS


# --- D144: pick_tickets is a cobot ticket master, not a ticket master. The
#     four pickticket coverage rows are three causes and no data gap, and two
#     of them sat at `error` severity — a permanent error in a list the
#     cleaning notebook says to clear before modeling is how the workers
#     checksum survived at D95. Each is explained, so each drops to `info`.
def test_pickticket_orphans_are_explained_not_tolerated():
    from wms_rpt.validation import EXPECTED_ORPHANS
    for tbl in ("picking", "cks", "picktickettimer", "cobot_jsons"):
        why = EXPECTED_ORPHANS.get(("pickticket", tbl))
        assert why, f"({'pickticket'!r}, {tbl!r}) has no recorded explanation"
        assert "D144" in why


def test_expected_orphans_demote_to_info():
    """The severity contract: an explained gap is info, not error."""
    from wms_rpt.validation import EXPECTED_ORPHANS, Validator
    pt = pd.DataFrame({"pickticket_id": ["A"]})
    pk = pd.DataFrame({"pickticket_id": ["A", "B"]})   # B is the orphan
    v = Validator()
    v.referential({"pick_tickets": pt, "picking": pk})
    orph = [f for f in v.findings if f.check == "orphan_keys"]
    assert orph, "no orphan finding raised"
    assert all(f.severity == "info" for f in orph), [f.severity for f in orph]
    assert any("EXPECTED" in f.detail for f in orph)
    assert ("pickticket", "picking") in EXPECTED_ORPHANS


def test_cks_has_two_distinct_worker_roles():
    """The 'rename both badge columns' plan was impossible: same row, two people."""
    assert KEY_ALIASES["worker"]["cks"] == "picker_employee_id"
    assert KEY_ALIASES["cker"]["cks"] == "cker_employee_id"
    assert KEY_ALIASES["worker"]["cks"] != KEY_ALIASES["cker"]["cks"]


def test_item_parent_aliases_to_the_cleartext_column():
    """D94: the cleartext re-pull made slot_history.itemid plaintext, so the
    items parent must alias to `itemid`. Pointing it at `hasheditemid` compares
    cleartext children against a hashed parent and yields 100% orphans across
    four child tables."""
    item = KEY_ALIASES["item"]
    assert item["cks"] == "sku"
    assert item["slot_history"] == "itemid"
    assert item["items"] == "itemid"


# Crosswalks are retired for live joins (D100), but CROSSWALKS still has to
# describe them correctly - data/raw/archive_hashed is the only way to reread
# the July extracts and that needs the mapping intact.
def test_crosswalks_declare_plaintext_and_hash_columns():
    for name, (plain, hashed) in CROSSWALKS.items():
        assert plain != hashed
        assert "hash" in hashed or hashed.startswith("anonymized")


def test_ck_shortfall_names_do_not_collide():
    """D75: the cleaning-side COUNT and the picks-side LABEL must not share a name.

    ``cleaning._derive_ck_features`` writes ``ck_shortfall_qty`` (a count, >= 1).
    ``picks.build_labels`` writes ``ck_shortfall`` (binary 0/1). A count of 3 is
    truthy, so if these ever share a name again a cleaning-side frame would pass
    silently wherever the label is expected.
    """
    import inspect
    from wms_rpt import cleaning, picks

    cleaning_src = inspect.getsource(cleaning._derive_ck_features)
    assert 'd["ck_shortfall_qty"]' in cleaning_src
    assert 'd["ck_shortfall"]' not in cleaning_src

    assert "ck_shortfall" in picks.LABEL_COLS
    assert "ck_shortfall_qty" not in picks.LABEL_COLS


def test_asof_enforces_minimum_gap():
    """The pick's own deduction is written ms BEFORE tspicked, not at it, so
    allow_exact_matches=False let it through on 91% of picks."""
    t = pd.Timestamp("2025-12-01 10:00:00")
    s = pd.DataFrame({"slotid": ["A", "A"], "sku": ["1", "1"],
                      "tspicked": [t, t]})
    sh = pd.DataFrame({
        "slotid": ["A", "A"],
        "itemid": ["1", "1"],
        "tslatestevent": [t - pd.Timedelta("10ms"),   # the deduction
                          t - pd.Timedelta("6h")],     # genuine prior state
        "physicalinventory": [0, 12],
        "virtualinventory": [0, 8],
    })
    st, rep = slot_state(s, sh)
    assert (st["onhand_before"] == 12).all(), "matched the deduction, not prior state"
    assert (st["hrs_since_slot_event"] >= 60 / 3600).all()


def test_bare_key_strips_prefix_and_nulls_nan_string():
    from wms_rpt.picks import _bare_key
    out = _bare_key(pd.Series(["FR451294", "451294", None, "CA1"]))
    assert out.tolist()[:2] == ["451294", "451294"]
    assert pd.isna(out.iloc[2]), '"nan" string must not become a join key'


# --- D139: _bare_key used `^[A-Za-z]+`, which strips EVERY leading letter,
#     while sku_to_itemid and validation._item_key strip exactly two. Both run
#     on spine["sku"], so a longer prefix normalized to a different join key
#     depending on which function saw it. Two alpha characters, all three.
def test_bare_key_matches_sku_to_itemid_prefix_rule():
    from wms_rpt.picks import _bare_key, sku_to_itemid
    vals = pd.Series(["FR115400", "115400"])
    assert _bare_key(vals).tolist() == sku_to_itemid(vals).tolist()
    # a longer prefix loses exactly two characters, not all of them
    assert _bare_key(pd.Series(["ABCD115400"])).iloc[0] == "CD115400"


def test_hrs_since_excludes_cobot_pick_events():
    """Source 25 is the cobot pick. Including it made hrs_since_slot_event
    'time since the previous pick at this slot' — picking cadence, which
    correlates with the outcome through slot activity."""
    t = pd.Timestamp("2025-12-01 12:00:00")
    s = pd.DataFrame({"slotid": ["A"], "sku": ["1"], "tspicked": [t]})
    sh = pd.DataFrame({
        "slotid": ["A", "A"],
        "itemid": ["1", "1"],
        "tslatestevent": [t - pd.Timedelta("10min"),   # cobot pick, excluded
                          t - pd.Timedelta("8h")],      # replenishment, kept
        "latesteventsource": [25, 24],
        "physicalinventory": [3, 40],
        "virtualinventory": [0, 30],
    })
    st, rep = slot_state(s, sh)
    # recency skips the pick event
    assert st["hrs_since_slot_event"].iloc[0] == pytest.approx(8, abs=0.01)
    # quantities take the most recent event, including the pick
    assert st["onhand_before"].iloc[0] == 3

# ============================== cobot timing features =======================

from wms_rpt.picks import _dedupe_tasks, _TASK_EVENT_PREFERENCE  # noqa: E402


def _rt_row(itemno, jobtaskid, eventtype, execdate, execrobot="P3-1000",
            execqty=1, **kw):
    """One result-task row at the shape cobot_line_features consumes."""
    return dict(pickticket_id=PT, itemno=itemno, jobtaskid=jobtaskid,
                eventtype=eventtype, execdate=execdate, execrobot=execrobot,
                execqty=execqty, tasklocation="S1", taskqty=1,
                exceptioncode="", **kw)


# --- D155: every result task is emitted twice, once as PICK and once as
#     PICKCOMPLETE. reconcile_with_picking dedupes the same key with
#     keep="last", which is POSITIONAL — it takes whichever emission happens
#     to sit later in the exploded frame. That is harmless while the payloads
#     agree, and wrong for the 99 groups where they do not: a TOTEFULL abort
#     recording execqty 0 on one robot, completed minutes later with execqty 1
#     on another. Selecting by event type is the only rule that survives row
#     order changing.
def test_dedupe_tasks_keeps_pickcomplete_on_a_divergent_pair():
    rt = pd.DataFrame([
        _rt_row("115400", "J1", "PICKCOMPLETE", "2025-12-01T10:05:00-05:00",
                execrobot="P3-7817", execqty=1),
        _rt_row("115400", "J1", "PICK", "2025-12-01T10:00:00-05:00",
                execrobot="P3-4475", execqty=0, ),
    ]).assign(_pt=PT)
    rt.loc[rt["eventtype"] == "PICK", "exceptioncode"] = "TOTEFULL"
    out = _dedupe_tasks(rt)
    assert len(out) == 1
    assert out["eventtype"].iloc[0] == "PICKCOMPLETE"
    assert out["execqty"].iloc[0] == 1, "kept the aborted first attempt"
    assert out["execrobot"].iloc[0] == "P3-7817"


def test_dedupe_tasks_is_not_order_dependent():
    """The defect proper: keep='last' gives a different answer per row order."""
    rows = [
        _rt_row("115400", "J1", "PICKCOMPLETE", "2025-12-01T10:05:00-05:00",
                execqty=1),
        _rt_row("115400", "J1", "PICK", "2025-12-01T10:00:00-05:00", execqty=0),
    ]
    a = _dedupe_tasks(pd.DataFrame(rows).assign(_pt=PT))
    b = _dedupe_tasks(pd.DataFrame(rows[::-1]).assign(_pt=PT))
    assert a["eventtype"].iloc[0] == b["eventtype"].iloc[0] == "PICKCOMPLETE"


# --- D155: 614,222 PICKCOMPLETE against 571,909 PICK, so 42,313 tasks have no
#     paired PICK. A rule written as "keep the PICKCOMPLETE rows" rather than
#     "prefer PICKCOMPLETE within the key" drops every one of them silently.
def test_dedupe_tasks_keeps_a_pick_with_no_pickcomplete():
    rt = pd.DataFrame([
        _rt_row("115400", "J1", "PICK", "2025-12-01T10:00:00-05:00"),
        _rt_row("115401", "J2", "PICKCOMPLETE", "2025-12-01T10:05:00-05:00"),
        _rt_row("115401", "J2", "PICK", "2025-12-01T10:05:00-05:00"),
    ]).assign(_pt=PT)
    out = _dedupe_tasks(rt)
    assert len(out) == 2
    assert set(out["jobtaskid"]) == {"J1", "J2"}
    assert out.set_index("jobtaskid").loc["J1", "eventtype"] == "PICK"


def test_task_event_preference_is_ordered_not_a_set():
    """Order carries the rule; a set would make the preference undefined."""
    assert isinstance(_TASK_EVENT_PREFERENCE, tuple)
    assert _TASK_EVENT_PREFERENCE[0] == "PICKCOMPLETE"


# --- D125 family: execdate is a COMPLETION time, so any window that includes
#     the current task contains that pick's own duration — and a pick that
#     becomes a shortage takes longer to work. The rolling count must be
#     closed="left". An interval feature built here without that boundary ran
#     a 6.5x monotone gradient against did_mismatch and was removed.
def test_robot_task_window_excludes_the_current_task():
    s, _ = build_spine(_picking(), _map(), _bridge())
    rt = pd.DataFrame([
        _rt_row("115400", "J1", "PICKCOMPLETE", "2025-12-01T10:00:00-05:00"),
        _rt_row("115401", "J2", "PICKCOMPLETE", "2025-12-01T10:05:00-05:00"),
    ])
    out = cobot_line_features(s, rt, xw={})
    # first task sees nothing before it; second sees the first, not itself
    assert out["cobot_robot_tasks_prev_15min"].tolist() == [0.0, 1.0]


def test_robot_task_window_forgets_outside_the_interval():
    s, _ = build_spine(_picking(), _map(), _bridge())
    rt = pd.DataFrame([
        _rt_row("115400", "J1", "PICKCOMPLETE", "2025-12-01T09:00:00-05:00"),
        _rt_row("115401", "J2", "PICKCOMPLETE", "2025-12-01T10:05:00-05:00"),
    ])
    out = cobot_line_features(s, rt, xw={})
    assert out["cobot_robot_tasks_prev_15min"].tolist() == [0.0, 0.0], (
        "a task 65 minutes earlier counted inside a 15-minute window")


def test_robot_window_is_per_robot():
    s, _ = build_spine(_picking(), _map(), _bridge())
    rt = pd.DataFrame([
        _rt_row("115400", "J1", "PICKCOMPLETE", "2025-12-01T10:00:00-05:00",
                execrobot="P3-1000"),
        _rt_row("115401", "J2", "PICKCOMPLETE", "2025-12-01T10:05:00-05:00",
                execrobot="P3-2000"),
    ])
    out = cobot_line_features(s, rt, xw={})
    assert out["cobot_robot_tasks_prev_15min"].tolist() == [0.0, 0.0], (
        "counted another robot's task; the window must group on execrobot")
    assert out["cobot_robot_changed"].tolist() == [0, 1]


def test_timing_features_are_ordered_by_execdate_not_row_order():
    """Row order must not change the answer; execdate is the only clock."""
    s, _ = build_spine(_picking(), _map(), _bridge())
    rows = [
        _rt_row("115401", "J2", "PICKCOMPLETE", "2025-12-01T10:05:00-05:00"),
        _rt_row("115400", "J1", "PICKCOMPLETE", "2025-12-01T10:00:00-05:00"),
    ]
    a = cobot_line_features(s, pd.DataFrame(rows), xw={})
    b = cobot_line_features(s, pd.DataFrame(rows[::-1]), xw={})
    assert a["cobot_robot_tasks_prev_15min"].tolist() == [0.0, 1.0]
    assert (a["cobot_robot_tasks_prev_15min"].tolist()
            == b["cobot_robot_tasks_prev_15min"].tolist())


# --- D156: an interval to the robot's previous task ran 1.08% in the shortest
#     decile against 7.15% in the ninth — a smooth 6.5x ramp, not the single
#     blown decile D125 showed. It was reading the pick's own duration through
#     the completion timestamp.
# --- D158: a position-within-ticket ordinal ranked over tasks that COMPLETED,
#     so earlier failures on a ticket lowered its later picks' rank. Alone it
#     showed 1.4x and passed every per-feature check. Paired with the WMS-side
#     ordinal_in_ticket the model recovered their difference, which counts the
#     failures preceding this pick: 1.54% at difference zero against 8.66% at
#     minus one. Neither leaks alone, so this is the first defect in the
#     register that no single-feature diagnostic could have found.
#     Both must stay out, under any name.
def test_leaking_timing_features_stay_out():
    s, _ = build_spine(_picking(), _map(), _bridge())
    rt = pd.DataFrame([
        _rt_row("115400", "J1", "PICKCOMPLETE", "2025-12-01T10:00:00-05:00"),
        _rt_row("115401", "J2", "PICKCOMPLETE", "2025-12-01T10:05:00-05:00"),
    ])
    out = cobot_line_features(s, rt, xw={})
    assert "cobot_secs_since_robot_prev_task" not in out.columns, (
        "the leaking interval feature is back; see D156 before re-adding it")
    assert "cobot_task_ordinal_in_ticket" not in out.columns, (
        "the completed-task ordinal is back; it leaks through its difference "
        "with ordinal_in_ticket — see D158 before re-adding it")


# --- D158: the leak lives in the PAIR, so the guard has to be the pair. Any
#     cobot-side position counter alongside the WMS-side one reconstructs the
#     difference. This pins the absence of a second ordinal generally, not
#     just the one name that was removed.
def test_no_second_position_counter_in_the_cobot_block():
    s, _ = build_spine(_picking(), _map(), _bridge())
    rt = pd.DataFrame([
        _rt_row("115400", "J1", "PICKCOMPLETE", "2025-12-01T10:00:00-05:00"),
        _rt_row("115401", "J2", "PICKCOMPLETE", "2025-12-01T10:05:00-05:00"),
    ])
    out = cobot_line_features(s, rt, xw={})
    counters = [c for c in out.columns
                if any(k in c.lower()
                       for k in ("ordinal", "position", "seq", "rank"))]
    assert not counters, (
        f"cobot block emits position counters {counters}; paired with "
        "rolling_context's ordinal_in_ticket their difference counts prior "
        "failures on the ticket (D158)")


# --- D154: has_told_location IS the population gate, so it is constant
#     wherever any model is trained or scored. A cobot feature set that
#     consists of the gate plus a label plus three worker attributes is not a
#     cobot feature set, and the naming defect at D152 hid that.
def test_cobot_block_emits_a_real_cobot_feature():
    s, _ = build_spine(_picking(), _map(), _bridge())
    rt = pd.DataFrame([
        _rt_row("115400", "J1", "PICKCOMPLETE", "2025-12-01T10:00:00-05:00"),
        _rt_row("115401", "J2", "PICKCOMPLETE", "2025-12-01T10:05:00-05:00"),
    ])
    out = cobot_line_features(s, rt, xw={})
    for c in ("cobot_robot_tasks_prev_15min", "cobot_robot_changed"):
        assert c in out.columns, f"{c} missing from cobot_line_features"
        assert pd.api.types.is_numeric_dtype(out[c]), (
            f"{c} is not numeric and will be dropped by select_dtypes (D150)")


# --- D168: the badge held in cobot_badge is a WORKER identifier, not a cobot
#     one (D152), and it reached the β-VAE as 126 one-hot columns while the
#     supervised model never saw it. It is now used only to join the worker's
#     attributes and is not emitted. The claim in the praxis that no model is
#     fitted on individual worker identity is unqualified only while this
#     holds, so it is pinned rather than left to inspection.
def test_worker_identity_is_not_emitted_as_a_feature():
    s, _ = build_spine(_picking(), _map(), _bridge())
    rt = pd.DataFrame([
        _rt_row("115400", "J1", "PICKCOMPLETE", "2025-12-01T10:00:00-05:00"),
        _rt_row("115401", "J2", "PICKCOMPLETE", "2025-12-01T10:05:00-05:00"),
    ])
    out = cobot_line_features(s, rt, xw={})
    assert "cobot_badge" not in out.columns, (
        "the worker badge is back in the feature matrix; see D168")
    ident = [c for c in out.columns
             if any(k in c.lower()
                    for k in ("badge", "employee", "execuser", "picker_id"))]
    assert not ident, f"identity-bearing columns emitted: {ident}"


def test_worker_attributes_still_join_without_the_badge():
    """Dropping the identifier must not drop what it was there to fetch."""
    s, _ = build_spine(_picking(), _map(), _bridge())
    rt = pd.DataFrame([
        _rt_row("115400", "J1", "PICKCOMPLETE", "2025-12-01T10:00:00-05:00",
                execuser="5341"),
        _rt_row("115401", "J2", "PICKCOMPLETE", "2025-12-01T10:05:00-05:00",
                execuser="5341"),
    ])
    wk = pd.DataFrame({"badge": ["5341"], "tenure_wks": [12.0],
                       "fte": [1.0], "picks_quartile": [3.0]})
    out = cobot_line_features(s, rt, xw={}, workers=wk)
    for c in ("worker_tenure_wks", "worker_fte", "worker_picks_quartile"):
        assert c in out.columns, f"{c} lost when the badge was dropped"
    assert out["worker_tenure_wks"].notna().any(), "join produced no values"
