"""Regression tests for wms_rpt.

Most of these are silent failures, not crashes.

"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wms_rpt import (  # noqa: E402
    Validator, clean_tables, coerce, read_rpt, sniff_layout, table_key,
)
from wms_rpt.cleaning import diagnose_json_failure, try_load_json  # noqa: E402

# ---------------------------------------------------------------- fixtures

HDR = ("pk                   eventtype        jsonmsg\n"
       + "-" * 20 + " " + "-" * 16 + " " + "-" * 20 + "\n")


def write_rpt(tmp_path: Path, name: str, body: str, footer_rows: int | None,
              bom: bool = True, trailer: bool = True) -> Path:
    txt = body
    if footer_rows is not None:
        txt += f"\n({footer_rows} rows affected)\n"
    if trailer:
        txt += "Completion time: 2026-08-04T00:00:00.0000000-04:00\n"
    p = tmp_path / name
    p.write_text(("\ufeff" if bom else "") + txt, encoding="utf-8")
    return p


@pytest.fixture
def simple_rpt(tmp_path):
    body = HDR + ('1                    CANCEL           {"a":1}\n'
                  '2                    NEW              {"a":2}\n')
    return write_rpt(tmp_path, "simple.rpt", body, 2)


# ---------------------------------------------------------------- table_key

@pytest.mark.parametrize("fname,expected", [
    ("2025-12_pick_tickets_wh1.rpt", "pick_tickets"),
    ("2025-12 cobot jsons wh1.rpt", "cobot_jsons"),      # space separators
    ("2025-12 slot history wh1.rpt", "slot_history"),
    ("2025-12_CKs_wh1.rpt", "cks"),
])
def test_table_key_handles_both_separators(fname, expected):
    assert table_key(Path(fname)) == expected


# ------------------------------------------------------------------ reader

def test_footer_and_trailer_are_stripped(simple_rpt):
    """Bug: 'Completion time:' trailers became a data row of NaNs."""
    df, lay = read_rpt(simple_rpt)
    assert len(df) == 2
    assert lay.declared_rows == 2


def test_checksum_captured(simple_rpt):
    df, lay = read_rpt(simple_rpt)
    assert lay.declared_rows == len(df)


def test_duplicate_header_names_deduped(tmp_path):
    """Bug: read_fwf raises on duplicate names (slots, early pick_tickets)."""
    body = ("a          a          b\n"
            "---------- ---------- ----------\n"
            "1          2          3\n")
    p = write_rpt(tmp_path, "dup.rpt", body, 1)
    df, lay = read_rpt(p)
    assert list(df.columns) == ["a", "a_1", "b"]


def test_last_column_open_ended(tmp_path):
    """Bug risk: a JSON blob longer than the ruler must not be parser-cut."""
    long_json = '{"k":"' + "x" * 40 + '"}'
    body = HDR + f"1                    NEW              {long_json}\n"
    p = write_rpt(tmp_path, "long.rpt", body, 1)
    df, _ = read_rpt(p)
    assert df["jsonmsg"].iloc[0] == long_json  # 48 chars > 20-char ruler


def test_wrapped_continuation_lines_are_stitched(tmp_path):
    """Bug: payloads wrapping onto the next physical line broke row parsing.

    Detection is structural now (D95): a real data row has something in the
    first field AND something past it, and leaves the ruler's inter-column gap
    blank. A wrapped fragment is continuous text that runs straight through
    that gap. This replaced majority-vote pattern learning, which worked here
    but ate 36 legitimate rows in the workers extract - see
    test_minority_first_field_not_stitched below, which is the other half of
    the same fix and must pass alongside this one.
    """
    rows = []
    for pk in range(1, 301):
        if pk % 50 == 0:
            j = ('{"EventType":"PICKCOMPLETE","EventInfo":"' + "y" * 250
                 + '","JobStatus":"OK"}')
            rows.append(f"{pk:<20} PICKCOMPLETE     {j[:180]}")
            rows.append(j[180:])  # continuation at column 0, no padding
        else:
            rows.append(f'{pk:<20} CANCEL           {{"EventType":"CANCEL"}}')
    p = write_rpt(tmp_path, "wrapped.rpt", HDR + "\n".join(rows), 300)
    df, lay = read_rpt(p)
    assert len(df) == 300
    assert lay.declared_rows == 300
    assert lay.stitched_lines == 6
    assert all(try_load_json(x) is not None for x in df["jsonmsg"])


def test_minority_first_field_not_stitched(tmp_path):
    """D95: a legitimate minority identifier in column 1 must not be read as
    a wrapped continuation. Value-pattern voting learned `\\d+` from numeric
    badges and consumed all 36 `Temp*` rows in the workers extract."""
    ruler = "-" * 8 + " " + "-" * 10
    lines = ["badge    hiredate  ", ruler]
    lines += [f"{i:<8} 2024-01-01" for i in range(1000, 1018)]  # 18 numeric
    lines += ["Temp1    NULL      ", "Temp2    NULL      "]      # 2 minority
    lines.append("(20 rows affected)")

    f = tmp_path / "2025-12 workers wh1.rpt"
    f.write_text("\n".join(lines), encoding="utf-8")

    df, lay = read_rpt(f)
    assert lay.stitched_lines == 0
    assert len(df) == 20
    assert set(df["badge"].str.strip()) >= {"Temp1", "Temp2"}


def test_stitching_inert_on_normal_files(simple_rpt):
    """Stitching must not fire when nothing is wrapped."""
    _, lay = read_rpt(simple_rpt)
    assert lay.stitched_lines == 0


def test_usecols_prunes_and_unknown_column_raises(tmp_path):
    body = ("a          b          c\n"
            "---------- ---------- ----------\n"
            "1          2          3\n")
    p = write_rpt(tmp_path, "cols.rpt", body, 1)
    df, _ = read_rpt(p, usecols=["a", "c"])
    assert list(df.columns) == ["a", "c"]
    with pytest.raises(KeyError):
        read_rpt(p, usecols=["nope"])


def test_utf16_detected(tmp_path):
    body = ("a          b\n"
            "---------- ----------\n"
            "1          x\n"
            "\n(1 rows affected)\n")
    p = tmp_path / "u16.rpt"
    p.write_text(body, encoding="utf-16")
    df, lay = read_rpt(p)
    assert lay.encoding == "utf-16"
    assert df["b"].iloc[0] == "x"


def test_reader_emits_no_futurewarnings(simple_rpt):
    """Bug: pandas >= 2.2 deprecation spam from DataFrame.replace downcasting.

    Treat any FutureWarning from the reader as a failure so the next pandas
    deprecation is caught by the suite, not by the user's console."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        read_rpt(simple_rpt)


# ------------------------------------------------------------------ coerce

def test_sparse_datetime_still_coerced(tmp_path):
    """Bug: picking.tsCKed (98.6 % null) stayed a string because the datetime
    threshold was judged against all rows instead of present values."""
    df = pd.DataFrame({"tscked": [None] * 98 + ["2025-12-01 08:00:00"] * 2})
    out, log = coerce(df)
    assert pd.api.types.is_datetime64_any_dtype(out["tscked"])


def test_zero_padded_ids_survive_via_category():
    df = pd.DataFrame({"slotid": pd.Series(["00042", "00043"], dtype="category")})
    out, _ = coerce(df)
    assert out["slotid"].dtype == "category"
    assert out["slotid"].iloc[0] == "00042"


def test_hash_not_numified():
    df = pd.DataFrame({"x": ["0001042BD1BC05AB", "23C94A5C710DD8AC"]})
    out, log = coerce(df)
    assert log.loc[log.column == "x", "decision"].iloc[0] in (
        "string", "category")


# --------------------------------------------------------------- validator

def _v(tables, layouts=None):
    v = Validator()
    v.structural(tables, layouts or {})
    v.domain(tables)
    return v


def test_ck_shortage_log_detected():
    """found is strictly < expected in every row -> ck_never_complete fires."""
    ck = pd.DataFrame({
        "ck_quantity": [1, 2, 5, 3],
        "quantityfoundbyck": [0, 1, 4, 0],
        "tscked": pd.NaT, "tspicked": pd.NaT,
    })
    v = Validator()
    v._cks_semantics({"cks": ck})
    checks = {f.check for f in v.findings}
    assert "ck_never_complete" in checks
    assert "ck_found_exceeds_expected" not in checks


def test_ck_found_exceeds_expected_is_error():
    ck = pd.DataFrame({"ck_quantity": [1], "quantityfoundbyck": [2]})
    v = Validator()
    v._cks_semantics({"cks": ck})
    assert any(f.check == "ck_found_exceeds_expected" and f.severity == "error"
               for f in v.findings)


def test_implausible_quantity_flagged():
    """12 audits recorded up to 93,302 units in slots holding <= 4 SKUs."""
    au = pd.DataFrame({"quantity": pd.array([3, 3, 93302], dtype="Int64")})
    v = Validator()
    v.domain({"slot_audit_history": au})
    assert any(f.check == "implausible_qty" for f in v.findings)


def test_case_variants_detected():
    """'Picking' vs 'PICKING' in cobot_jsons.messagetype."""
    cj = pd.DataFrame({"m": ["Picking", "PICKING"] * 30})
    v = Validator()
    v.structural({"t": cj}, {})
    assert any(f.check == "case_variants" for f in v.findings)


def test_duplicate_events_across_producers():
    """LocusWorker + LocusConsumer double-emit: dedup on pk misses these."""
    cj = pd.DataFrame({
        "pk": [1, 2], "pickticketid": ["A", "A"],
        "eventtype": ["CANCEL", "CANCEL"],
        "ts": ["2025-12-01", "2025-12-01"],
        "appname": ["LocusWorker", "LocusConsumer"],
    })
    v = Validator()
    v._cobot_events({"cobot_jsons": cj})
    assert any(f.check == "duplicate_events" for f in v.findings)


def test_parent_key_uniqueness_guard():
    """449 surplus pick_tickets rows (jobrobot variants) were an export error;
    the guard must flag any recurrence."""
    pt = pd.DataFrame({
        "pickticket_id": ["A", "A", "B"],
        "jobrobot": ["P3-1", "P3-2", "P3-1"],
    })
    v = Validator()
    v.referential({"pick_tickets": pt})
    assert any(f.check == "parent_key_not_unique" for f in v.findings)


# ---------------------------------------------------------------- cleaning

def test_colorresult_dropped_and_ck_features_derived():
    ck = pd.DataFrame({
        "ck_quantity": pd.array([2, 1], dtype="Int64"),
        "quantityfoundbyck": pd.array([1, 0], dtype="Int64"),
        "colorresult": ["GREEN", "ORANGE"],
    })
    clean, log = clean_tables({"cks": ck})
    d = clean["cks"]
    assert "colorresult" not in d.columns
    assert "ck_found_all" not in d.columns  # False by construction; never emit
    assert list(d["ck_shortfall_qty"]) == [1, 1]
    assert d["ck_fill_rate"].max() == 0.5  # ceiling (e-1)/e, never 1.0


def test_implausible_values_flagged_not_clipped():
    au = pd.DataFrame({"quantity": pd.array([3, 93302], dtype="Int64")})
    clean, _ = clean_tables({"slot_audit_history": au})
    d = clean["slot_audit_history"]
    assert d["quantity"].max() == 93302          # value preserved
    assert list(d["quantity_implausible"]) == [False, True]


def test_key_canonicalization():
    cj = pd.DataFrame({"pickticketid": ["A"], "eventtype": ["CANCEL"],
                       "ts": ["2025-12-01"], "appname": ["LocusWorker"]})
    clean, _ = clean_tables({"cobot_jsons": cj})
    assert "pickticket_id" in clean["cobot_jsons"].columns


def test_subset_loaded_tables_not_deduped():
    """picking dedup on the loaded subset overcounts 2,700 vs 926 true."""
    p = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
    clean, log = clean_tables({"picking": p}, subset_loaded={"picking": 3})
    assert len(clean["picking"]) == 2  # duplicates KEPT
    assert not (log["action"] == "drop_duplicates").any() if not log.empty else True


def test_implausible_scoped_by_size_class():
    """skuMaxCount counts SKUs, not units: a Pallet slot with one SKU holds
    thousands of units legitimately. The ceiling exempts bulk size classes but
    still catches the 40,078-unit Slot16 ledger error."""
    slots = pd.DataFrame({"slotid": ["P1", "S1"], "size": ["Pallet", "Slot16"]})
    sh = pd.DataFrame({
        "slotid": ["P1", "S1", "S1"],
        "physicalinventory": pd.array([2233, 40078, 3], dtype="Int64"),
        "virtualinventory": pd.array([2233, 40078, 3], dtype="Int64"),
    })
    tables = {"slots": slots, "slot_history": sh}

    v = Validator()
    v.domain(tables)
    iq = [f for f in v.findings if f.check == "implausible_qty"]
    assert iq and all("non-bulk" in f.detail for f in iq)
    assert all(f.n_affected == 1 for f in iq)          # only the Slot16 row

    clean, log = clean_tables(tables)
    d = clean["slot_history"]
    assert list(d["physicalinventory_implausible"]) == [False, True, False]
    assert d["physicalinventory"].max() == 40078       # value preserved


def test_implausible_unscoped_without_slots():
    """No slots table -> no exemption; behavior falls back to the ceiling."""
    sh = pd.DataFrame({"slotid": ["P1"],
                       "physicalinventory": pd.array([2233], dtype="Int64")})
    clean, _ = clean_tables({"slot_history": sh})
    assert bool(clean["slot_history"]["physicalinventory_implausible"].iloc[0])


# -------------------------------------------------------------------- JSON

def test_json_truncation_diagnosed():
    """Payloads cut mid-value have to be recognizable as truncated, not just
    unparseable - I need to tell "the export broke this" apart from "the robot
    sent something odd". The real cap is SSMS Results-to-File at 8,192 chars
    (see test_chunked_jsonmsg_reassembled); this fixture just cuts short."""
    cut = ('{"EventType":"NEW","JobId":"07be4a73","JobDate":"2025-12-01",'
           '"JobPriority":298,"SingleUnit":false,"JobTasks":{"' )
    assert try_load_json(cut) is None
    assert diagnose_json_failure(cut).startswith("truncated")


def test_json_double_encoded_and_trailing_garbage_recovered():
    assert try_load_json('"{\\"a\\":1}"') == {"a": 1}
    assert try_load_json('{"a":1}garbage') == {"a": 1}


def test_json_booleans_not_numified():
    """Bug: JSON `false` passed the numeric branch and became float 0.0."""
    cj = pd.DataFrame({
        "pickticketid": ["A"], "eventtype": ["NEW"], "ts": ["2025-12-01"],
        "appname": ["LocusWorker"],
        "jsonmsg": ['{"EventType":"NEW","SingleUnit":false}'],
    })
    clean, _ = clean_tables({"cobot_jsons": cj})
    d = clean["cobot_jsons"]
    assert d["json_singleunit"].dtype == "boolean"


def test_nested_jobtasks_list_survives_expansion():
    """Bug: full (untruncated) NEW payloads carry JobTasks as a LIST of task
    dicts. Lists are unhashable — .unique() raised TypeError and killed the
    expansion. They must survive as JSON strings, parquet-safe, content
    verbatim."""
    payload = ('{"EventType":"NEW","SingleUnit":false,"JobTasks":'
               '[{"TaskId":"t1","Qty":2},{"TaskId":"t2","Qty":1}]}')
    cj = pd.DataFrame({
        "pickticketid": ["A", "B"], "eventtype": ["NEW", "CANCEL"],
        "ts": ["2025-12-01"] * 2, "appname": ["LocusWorker"] * 2,
        "jsonmsg": [payload, '{"EventType":"CANCEL","SingleUnit":false}'],
    })
    clean, log = clean_tables({"cobot_jsons": cj})
    d = clean["cobot_jsons"]
    col = [c for c in d.columns if "jobtasks" in c]
    assert len(col) == 1
    tasks = json.loads(d[col[0]].dropna().iloc[0])
    assert tasks == [{"TaskId": "t1", "Qty": 2}, {"TaskId": "t2", "Qty": 1}]
    assert d["json_singleunit"].dtype == "boolean"     # typing still applied
    # parquet round-trip must survive (the whole point of serialising)
    import io as _io
    buf = _io.BytesIO()
    d.to_parquet(buf, index=False)
    back = pd.read_parquet(_io.BytesIO(buf.getvalue()))
    assert json.loads(back[col[0]].dropna().iloc[0]) == tasks


def test_chunked_jsonmsg_reassembled():
    """SSMS hard-caps text columns at 8192; the workaround exports jsonmsg as
    SUBSTRING chunks. Reassembly must restore payloads byte-exact (including
    ones spanning three chunks), stay NaN-safe, and then expand normally."""
    long_payload = ('{"EventType":"NEW","SingleUnit":false,"JobTasks":'
                    '{"OrderJobTask":[' +
                    ",".join(f'{{"TaskId":"t{i}","Qty":1}}' for i in range(700))
                    + "]}}")
    assert len(long_payload) > 16000     # forces a 3-chunk split
    short = '{"EventType":"CANCEL","SingleUnit":false}'

    def chunks(x, w=8000):
        out = [x[i:i + w] for i in range(0, len(x), w)]
        return out + [None] * (3 - len(out))

    lp, sp = chunks(long_payload), chunks(short)
    cj = pd.DataFrame({
        "pickticketid": ["A", "B", "C"], "eventtype": ["NEW", "CANCEL", "NEW"],
        "ts": ["2025-12-01"] * 3, "appname": ["LocusWorker"] * 3,
        "jsonmsg_1": [lp[0], sp[0], None],
        "jsonmsg_2": [lp[1], sp[1], None],
        "jsonmsg_3": [lp[2], sp[2], None],
    })
    clean, log = clean_tables({"cobot_jsons": cj})
    d = clean["cobot_jsons"]
    assert not any(c.startswith("jsonmsg_") for c in d.columns)
    assert (log["action"] == "reassemble_json").any()
    # long payload survived reassembly + expansion byte-exact
    col = [c for c in d.columns if "orderjobtask" in c][0]
    tasks = json.loads(d[col].dropna().iloc[0])
    assert len(tasks) == 700 and tasks[699]["TaskId"] == "t699"
    # all-null chunk row stays a clean NaN, not an empty-string payload
    assert d.loc[2, "json_truncated"] == False  # noqa: E712
    assert int(d["json_truncated"].sum()) == 0


def test_overflow_supplement_patches_truncated_payloads():
    """The full-chunked export would be ~68 GB of padding, so overflow rows
    ship in a small second file keyed by pk. The merge must replace exactly
    the truncated payloads, leave intact ones alone, then expand cleanly."""
    cut = ('{"EventType":"NEW","SingleUnit":false,"JobTasks":{"OrderJobTask":[{"Ta')
    full = ('{"EventType":"NEW","SingleUnit":false,"JobTasks":{"OrderJobTask":'
            '[{"TaskId":"t0","Qty":2},{"TaskId":"t1","Qty":1}]}}')
    ok = '{"EventType":"CANCEL","SingleUnit":false}'
    main = pd.DataFrame({
        "pk": pd.array([101, 102], dtype="Int64"),
        "pickticketid": ["A", "B"], "eventtype": ["NEW", "CANCEL"],
        "ts": ["2025-12-01"] * 2, "appname": ["LocusWorker"] * 2,
        "jsonmsg": [cut, ok],
    })
    overflow = pd.DataFrame({
        "pk": pd.array([101], dtype="Int64"),
        "jsonmsg_1": [full[:60]], "jsonmsg_2": [full[60:]], "jsonmsg_3": [None],
    })
    # Reproduce the field failure: sparse tail chunks pass through coerce()'s
    # <=50-distinct rule and arrive as CATEGORICAL; fillna("") raised on them.
    overflow["jsonmsg_2"] = overflow["jsonmsg_2"].astype("category")
    overflow["jsonmsg_3"] = overflow["jsonmsg_3"].astype("category")
    clean, log = clean_tables({"cobot_jsons": main,
                               "cobot_jsons_overflow": overflow})
    assert "cobot_jsons_overflow" not in clean
    d = clean["cobot_jsons"]
    assert int(d["json_truncated"].sum()) == 0
    col = [c for c in d.columns if "orderjobtask" in c][0]
    assert json.loads(d.loc[d.pk == 101, col].iloc[0])[1]["TaskId"] == "t1"
    # the intact CANCEL row was not touched
    assert d.loc[d.pk == 102, "json_eventtype"].iloc[0] == "CANCEL"
    assert (log["action"] == "overflow_patch").any()


def test_truncated_payloads_flagged_and_finding_filed():
    cut = ('{"EventType":"NEW","JobPriority":298,"SingleUnit":false,'
           '"JobTasks":{"' + "a" * 120)
    cj = pd.DataFrame({
        "pickticketid": ["A", "B"], "eventtype": ["NEW", "CANCEL"],
        "ts": ["2025-12-01"] * 2, "appname": ["LocusWorker"] * 2,
        "jsonmsg": [cut, '{"EventType":"CANCEL"}'],
    })
    v = Validator()
    clean, _ = clean_tables({"cobot_jsons": cj}, validator=v)
    d = clean["cobot_jsons"]
    assert list(d["json_truncated"]) == [True, False]
    assert any(f.check == "json_parse_failure" for f in v.findings)


# ------------------------------------------------------------------- tasks

from wms_rpt import explode_tasks, identify_fields, reconcile_with_picking  # noqa: E402
from wms_rpt.picks import wms_divergence


def _task_fixture():
    """Opaque SkuId/SlotId values that sit inside the reference domains.

    The point is that identify_fields must label them from overlap alone,
    without knowing the names - so the values are deliberately shapeless. They
    happen to be 64 chars because the July extracts were hashed; any opaque
    token would do, and the test would be no weaker with 6-digit ones."""
    sku_a, sku_b = "A" * 64, "B" * 64
    slot_x, slot_y = "C" * 64, "D" * 64
    tasks_1 = json.dumps([
        {"SkuId": sku_a, "SlotId": slot_x, "Quantity": 2, "Meta": {"z": 1}},
        {"SkuId": sku_b, "SlotId": slot_y, "Quantity": 1, "Meta": {"z": 2}},
    ])
    tasks_2 = json.dumps([{"SkuId": sku_a, "SlotId": slot_x, "Quantity": 3}])
    cj = pd.DataFrame({
        "pk": pd.array([1, 2], dtype="Int64"),
        "pickticket_id": ["T1", "T2"], "eventtype": ["NEW", "NEW"],
        "ts": ["2025-12-01"] * 2, "appname": ["LocusConsumer"] * 2,
        "json_jobtasks_orderjobtask": [tasks_1, tasks_2],
    })
    refs = {
        "slots": pd.DataFrame({"slotid": [slot_x, slot_y]}),
        "shippingintent_sku": pd.DataFrame({"sku": [sku_a, sku_b]}),
    }
    picking = pd.DataFrame({
        "pickticket_id": ["T1", "T1", "T2"],
        "quantity": pd.array([2, 1, 5], dtype="Int64"),
        "quantitypicked": pd.array([2, 1, 4], dtype="Int64"),
    })
    return cj, refs, picking


def test_explode_discovers_fields_and_keeps_nested():
    cj, _, _ = _task_fixture()
    td = explode_tasks(cj, "json_jobtasks_orderjobtask")
    assert len(td) == 3
    assert list(td["_task_seq"]) == [0, 1, 0]
    assert {"skuid", "slotid", "quantity"} <= set(td.columns)
    assert json.loads(td["meta"].dropna().iloc[0]) == {"z": 1}
    assert td["quantity"].sum() == 6


def test_field_identification_by_domain_overlap():
    # Labels are `table.column`, not a fixed vocabulary: domains are built
    # from whatever columns the caller supplies, so adding a reference domain
    # is a change to the caller rather than to identify_fields.
    cj, refs, _ = _task_fixture()
    td = explode_tasks(cj, "json_jobtasks_orderjobtask")
    ident = identify_fields(td, refs)
    got = dict(zip(ident.column, ident.identified_as))
    assert got["skuid"] == "shippingintent_sku.sku_ref"
    assert got["slotid"] == "slots.slotid_ref"
    assert got["quantity"] == "quantity"


def test_reconciliation_math():
    cj, refs, picking = _task_fixture()
    td = explode_tasks(cj, "json_jobtasks_orderjobtask")
    rec = reconcile_with_picking(td, picking, identify_fields(td, refs))
    r = rec.set_index("pickticket_id")
    # T1: robot 3 units vs wms ordered 3, picked 3 -> diffs 0
    assert r.loc["T1", "units_diff_vs_ordered"] == 0
    # T2: robot 3 vs ordered 5 (diff -2), picked 4 (diff -1)
    assert r.loc["T2", "units_diff_vs_ordered"] == -2
    assert r.loc["T2", "units_diff_vs_picked"] == -1


def test_reconciliation_declines_without_quantity_field():
    cj, refs, picking = _task_fixture()
    cj["json_jobtasks_orderjobtask"] = cj["json_jobtasks_orderjobtask"].str.replace(
        "Quantity", "Mystery")
    td = explode_tasks(cj, "json_jobtasks_orderjobtask")
    assert reconcile_with_picking(td, picking, identify_fields(td, refs)) is None


def test_reconcile_dedupes_reemissions_and_excludes_pack():
    """Jobs re-emit: each NEW event repeats the full task list, and PACK
    tasks ride along with qty 0-or-junk. Without dedupe+filter the first
    field run matched 0.0% of tickets."""
    task = {"JobTaskId": "j1", "TaskType": "PICK", "TaskQty": 2}
    pack = {"JobTaskId": "jp", "TaskType": "PACK", "TaskQty": 0}
    emission = json.dumps([task, pack])
    cj = pd.DataFrame({
        "pk": pd.array([1, 2], dtype="Int64"),
        "pickticket_id": ["T1", "T1"],          # SAME ticket, re-emitted
        "eventtype": ["NEW", "NEW"],
        "ts": ["2025-12-01", "2025-12-02"], "appname": ["LocusConsumer"] * 2,
        "json_jobtasks_orderjobtask": [emission, emission],
    })
    picking = pd.DataFrame({
        "pickticket_id": ["T1"],
        "quantity": pd.array([2], dtype="Int64"),
        "quantitypicked": pd.array([2], dtype="Int64"),
    })
    td = explode_tasks(cj, "json_jobtasks_orderjobtask")
    assert len(td) == 4                          # 2 emissions x 2 tasks
    rec = reconcile_with_picking(td, picking, identify_fields(td, {}))
    r = rec.set_index("pickticket_id")
    assert rec.attrs["duplicate_task_emissions_collapsed"] == 2
    assert rec.attrs["non_pick_tasks_excluded"] == 1
    assert r.loc["T1", "robot_units"] == 2       # not 4, not 8
    assert r.loc["T1", "units_diff_vs_ordered"] == 0


def test_reconcile_qty_override():
    """Result-side reconciliation uses execqty (did) not taskqty (told)."""
    t = json.dumps([{"JobTaskId": "j1", "TaskType": "PICK",
                     "TaskQty": 5, "ExecQty": 3}])
    cj = pd.DataFrame({
        "pk": pd.array([1], dtype="Int64"), "pickticket_id": ["T1"],
        "eventtype": ["PICKCOMPLETE"], "ts": ["2025-12-01"],
        "appname": ["LocusWorker"],
        "json_jobtasks_orderjobresulttask": [t],
    })
    picking = pd.DataFrame({
        "pickticket_id": ["T1"], "quantity": pd.array([5], dtype="Int64"),
        "quantitypicked": pd.array([3], dtype="Int64"),
    })
    td = explode_tasks(cj, "json_jobtasks_orderjobresulttask")
    rec = reconcile_with_picking(td, picking, identify_fields(td, {}),
                                 qty_col="execqty")
    assert rec.attrs["qty_field_used"] == "execqty"
    assert rec.set_index("pickticket_id").loc["T1", "units_diff_vs_picked"] == 0


def test_boolean_flag_never_a_quantity():
    """CountEligible is a boolean whose name contains 'count'; summing it as
    units produced 0.0%% reconciliation. Booleans are flags, full stop."""
    t = json.dumps([{"JobTaskId": "j1", "TaskType": "PICK", "TaskQty": 2,
                     "CountEligible": False}])
    cj = pd.DataFrame({
        "pk": pd.array([1], dtype="Int64"), "pickticket_id": ["T1"],
        "eventtype": ["NEW"], "ts": ["2025-12-01"], "appname": ["LC"],
        "json_jobtasks_orderjobtask": [t],
    })
    td = explode_tasks(cj, "json_jobtasks_orderjobtask")
    ident = identify_fields(td, {})
    got = dict(zip(ident.column, ident.identified_as))
    assert got["counteligible"] == "flag"
    assert got["taskqty"] == "quantity"
    picking = pd.DataFrame({"pickticket_id": ["T1"],
                            "quantity": pd.array([2], dtype="Int64"),
                            "quantitypicked": pd.array([2], dtype="Int64")})
    rec = reconcile_with_picking(td, picking, ident)
    assert rec.attrs["qty_field_used"] == "taskqty"


def test_negative_ledger_quantities_admitted_and_net_summed():
    """Order-side TaskQty spans negatives (adjustment emissions, field min
    -18). They are quantities; the reconciliation sums the NET instruction."""
    t = json.dumps([
        {"JobTaskId": "j1", "TaskType": "PICK", "TaskQty": 5},
        {"JobTaskId": "j2", "TaskType": "PICK", "TaskQty": -2},
    ])
    cj = pd.DataFrame({
        "pk": pd.array([1], dtype="Int64"), "pickticket_id": ["T1"],
        "eventtype": ["NEW"], "ts": ["2025-12-01"], "appname": ["LC"],
        "json_jobtasks_orderjobtask": [t],
    })
    td = explode_tasks(cj, "json_jobtasks_orderjobtask")
    ident = identify_fields(td, {})
    row = ident.set_index("column").loc["taskqty"]
    assert row.identified_as == "quantity"
    assert "negative" in row.evidence
    picking = pd.DataFrame({"pickticket_id": ["T1"],
                            "quantity": pd.array([3], dtype="Int64"),
                            "quantitypicked": pd.array([3], dtype="Int64")})
    rec = reconcile_with_picking(td, picking, ident)
    assert rec.set_index("pickticket_id").loc["T1", "robot_units"] == 3
    assert rec.set_index("pickticket_id").loc["T1", "units_diff_vs_ordered"] == 0

# --- defect: wms_divergence SUMS where line_grain_cobot ORs, so deduping on
#     jobtaskid is mandatory. Tasks re-emit about twice per slot, and summing
#     the duplicates doubled every robot quantity.
#
#     NOTE: wms_divergence emits identically zero in production now - it
#     needed the slot crosswalk, which D100 retired - and since 1.6.0 it is
#     out of LABEL_COLS as well as TARGETS (D138). Kept because the sum-vs-OR
#     distinction is a real trap and I would rather not rediscover it if the
#     label ever comes back.
def test_wms_divergence_dedupes_reemitted_tasks():
    spine = pd.DataFrame({"pickticket_id": ["PT1"], "slotid": ["H1"]})
    rt = pd.DataFrame({
        "pickticket_id": ["PT1"] * 2,
        "jobtaskid": ["J1", "J1"],          # same task, emitted twice
        "tasktype": ["PICK"] * 2,
        "_slot": ["S100"] * 2,
        "execqty": [3, 3],
    })
    picking = pd.DataFrame({"pickticket_id": ["PT1"], "slotid": ["H1"],
                            "quantitypicked": [3]})
    out = wms_divergence(spine, rt, picking, {"slot": {"100": "H1"}})
    assert out["wms_divergence"].tolist() == [0]   # 3 == 3, not 6 != 3


# --- Grain changes the answer. Summing across slots lets a +2 at one slot
#     cancel a -2 at another, so ticket grain reports agreement that slot grain
#     does not. The specific percentages I used to quote here came from the
#     crosswalk era and are withdrawn (D100), but the arithmetic is permanent
#     and this fixture demonstrates it in four rows.
def test_reconcile_slot_grain_differs_from_ticket_grain():
    ident = pd.DataFrame({"column": ["execqty"], "identified_as": ["quantity"]})
    rt = pd.DataFrame({
        "pickticket_id": ["PT1"] * 2, "jobtaskid": ["J1", "J2"],
        "tasktype": ["PICK"] * 2, "_slot": ["S100", "S200"],
        "execqty": [5, 1],                  # +2 at one slot, -2 at the other
    })
    picking = pd.DataFrame({
        "pickticket_id": ["PT1"] * 2, "slotid": ["H1", "H2"],
        "quantity": [3, 3], "quantitypicked": [3, 3]})
    xw = {"100": "H1", "200": "H2"}
    tkt = reconcile_with_picking(rt, picking, ident)
    slot = reconcile_with_picking(rt, picking, ident, slot_xw=xw)
    assert tkt["units_diff_vs_picked"].abs().sum() == 0    # cancels
    assert slot["units_diff_vs_picked"].abs().sum() == 4   # does not
    assert slot.attrs["grain"] == "(ticket, slot)"


def test_pick_tickets_is_ticket_grain_after_cleaning():
    """The export emits each ticket twice, jobrobot null then assigned. Exact-
    duplicate dropping misses it because the rows genuinely differ."""
    pt = pd.DataFrame({
        "pickticket_id": ["A", "A", "B", "B"],
        "jobrobot": [None, "P3-2836", None, "P3-1104"],
        "pickticketstyle": ["WavePickStyle"] * 4,
    })
    out = (pt.sort_values("jobrobot", na_position="last", kind="mergesort")
             .drop_duplicates("pickticket_id", keep="first")
             .reset_index(drop=True))
    assert out["pickticket_id"].is_unique
    assert out["jobrobot"].isna().sum() == 0
    assert set(out["jobrobot"]) == {"P3-2836", "P3-1104"}