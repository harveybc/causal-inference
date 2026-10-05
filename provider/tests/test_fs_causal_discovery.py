"""PCMCI+ comparator stage: pinned library, planted lagged cause found, null not; NOT_RUN recorded when tigramite is absent."""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

from causal_inference_provider import fs_causal_batch as FB
from causal_inference_provider import fs_causal_discovery as D

HAS_TIGRAMITE = importlib.util.find_spec("tigramite") is not None


def test_pin_reports_library_version_and_license_or_error():
    pin = D.tigramite_pin()
    assert pin["library"] == "tigramite"
    if HAS_TIGRAMITE:
        assert pin["version"] and "GPL" in str(pin["license"]).upper() or pin["license"] is None
    else:
        assert "error" in pin


@pytest.mark.skipif(not HAS_TIGRAMITE, reason="tigramite not installed here; the stage records NOT_RUN (see next test)")
def test_pcmci_plus_finds_a_planted_lagged_cause_and_not_a_null():
    rng = np.random.default_rng(5)
    n = 3000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.6 * x[i - 1] + rng.normal()
    r = rng.normal(size=n)
    r[1:] += 0.5 * x[:-1]
    H = rng.normal(size=(n, 2))
    rec = D.pcmci_feature(x, r, H, tau_max=3, pc_alpha=0.05, alpha=0.01, var_names=["x", "r", "h1", "h2"])
    assert rec["state"] == "RUN" and any(l["lag"] == 1 and l["edge"] == "-->" for l in rec["links_x_to_r"]), rec
    z = rng.normal(size=n)
    rec0 = D.pcmci_feature(z, r, H, tau_max=3, pc_alpha=0.05, alpha=0.01, var_names=["z", "r", "h1", "h2"])
    assert rec0["verdict"] == "PCMCI_PLUS_NO_LINK_X_TO_R"


@pytest.mark.skipif(HAS_TIGRAMITE, reason="tigramite present: the NOT_RUN path is exercised only where it is absent")
def test_stage_records_not_run_without_tigramite(tmp_path):
    out = str(tmp_path / "out")
    os.makedirs(out)
    json.dump({"chunks": [], "inputs_root": str(tmp_path), "base": "batch_001", "candidates_total": 0}, open(os.path.join(out, "plan.json"), "w"))
    with pytest.raises(SystemExit, match="NOT_RUN"):
        D.run_pcmci(out)
    prog = json.load(open(os.path.join(out, "progress_discovery.json")))
    assert prog["state"] == "NOT_RUN" and prog["reason"]
