"""Regression tests for the binomial window-label rule (wms_rpt 1.5.0).

Defect: ``rollup_labels`` labelled windows by anomalous-pick *share*, which
ignores how much evidence produced the share. A 7-pick window with one event
scored 14.3% and cleared any multiple of a ~3% base rate; a 400-pick window
needed 36 events for the same share. On the December wh1 test split, windows
labelled anomalous had a median of 7 picks against 341 for the rest, and
``-n_picks`` alone separated the classes at AUROC 0.963 — window size was a
better detector than the model it was meant to evaluate.

One test per historical defect, per the suite's convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wms_rpt.rollup import (DEFAULT_ALPHA, DEFAULT_MIN_PICKS, rollup_labels)

BASE = {"y": 0.03}


def _picks(n_by_window, rate_by_window, start="2025-12-01 08:00:00"):
    """Build a ts/labels pair placing ``n`` picks inside disjoint windows.

    Windows are spaced 60 minutes apart so the 15/5 fan-out cannot mix them.
    """
    t0 = pd.Timestamp(start)
    ts, y = [], []
    for i, (n, rate) in enumerate(zip(n_by_window, rate_by_window)):
        k = int(round(n * rate))
        # all inside a single 5-minute stride cell (ms spacing, not seconds:
        # 341 picks at 1s apart would straddle two cells and split the window)
        ts += [t0 + pd.Timedelta(hours=i) + pd.Timedelta(milliseconds=j)
               for j in range(n)]
        y += [1] * k + [0] * (n - k)
    ts = pd.Series(pd.to_datetime(ts))
    return ts, pd.DataFrame({"y": y}, index=ts.index)


def _labelled(ts, labels, **kw):
    out = rollup_labels(labels, ts, base_rates=BASE, **kw)
    return out[out["n_picks"] > 0]


def test_small_window_with_one_event_is_not_anomalous():
    """The core defect: n=7, k=1 is 14.3% share but p=0.19 — unremarkable."""
    ts, y = _picks([7], [1 / 7])
    out = _labelled(ts, y, min_picks=1)
    assert out["y_pval"].max() > DEFAULT_ALPHA
    assert (out["y"].dropna() == 0).all()


def test_proportion_rule_reproduces_the_defect():
    """Kept so the failure the pivot fixed stays demonstrable, not folklore."""
    ts, y = _picks([7], [1 / 7])
    out = rollup_labels(y, ts, method="proportion", base_rates=BASE,
                        rate_multiple=3.0)
    out = out[out["n_picks"] > 0]
    assert (out["y"] == 1).any(), "proportion rule should flag the 7-pick window"


def test_large_window_with_excess_events_is_anomalous():
    """n=341 at ~6% against a 3% base is p=0.004 — genuinely surprising."""
    ts, y = _picks([341], [20 / 341])
    out = _labelled(ts, y)
    assert (out["y"] == 1).all()
    assert out["y_pval"].max() < DEFAULT_ALPHA


def test_large_window_at_base_rate_is_normal():
    ts, y = _picks([400], [0.03])
    out = _labelled(ts, y)
    assert (out["y"] == 0).all()


def test_windows_below_min_picks_are_nan_not_zero():
    """Excluded for want of evidence — not asserted normal."""
    ts, y = _picks([5, 400], [0.0, 0.03])
    out = _labelled(ts, y, min_picks=DEFAULT_MIN_PICKS)
    small = out[out["n_picks"] < DEFAULT_MIN_PICKS]
    assert len(small) > 0
    assert small["y"].isna().all()


def test_zero_events_never_significant():
    ts, y = _picks([500], [0.0])
    out = _labelled(ts, y)
    assert (out["y"] == 0).all()
    assert np.isclose(out["y_pval"], 1.0).all()


def test_label_is_not_monotone_in_window_size():
    """The regression proper: size alone must not order the labels.

    Equal *shares* across very different sizes — the proportion rule calls
    all of them anomalous, the binomial rule calls only the large ones.
    """
    sizes = [10, 50, 200, 600]
    ts, y = _picks(sizes, [0.09] * len(sizes))
    out = _labelled(ts, y, min_picks=1).sort_values("n_picks")
    lab = out["y"].to_numpy()
    assert lab[0] == 0, "smallest window should not be flagged on share alone"
    assert lab[-1] == 1, "largest window with the same share should be flagged"


def test_base_rate_must_be_passed_for_a_reported_result():
    ts, y = _picks([400], [0.03])
    with pytest.warns(RuntimeWarning, match="fitted on training"):
        rollup_labels(y, ts, method="binomial")


def test_base_rate_out_of_range_raises():
    ts, y = _picks([400], [0.03])
    with pytest.raises(ValueError, match="must be in"):
        rollup_labels(y, ts, base_rates={"y": 0.0})


def test_unknown_method_raises():
    ts, y = _picks([400], [0.03])
    with pytest.raises(ValueError, match="method must be one of"):
        rollup_labels(y, ts, method="proportional")


def test_count_method_requires_min_count():
    ts, y = _picks([400], [0.03])
    with pytest.raises(ValueError, match="requires min_count"):
        rollup_labels(y, ts, method="count")


def test_counts_and_shares_always_returned():
    ts, y = _picks([400], [0.03])
    out = _labelled(ts, y)
    for col in ("n_picks", "y_n", "y_rate", "y_pval", "y_base_rate"):
        assert col in out.columns
