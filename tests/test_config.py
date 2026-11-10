import copy
import json

import pytest

from cv.config import CONFIG_PATH, load_thresholds, parse_thresholds

VALID = {
    "schema_version": 2,
    "squat": {
        "rep": {
            "side": {"down": 140, "up": 155, "min_drop": 25, "min_rep_s": 0.5},
            "front": {"down": 0.85, "up": 0.93, "min_drop": 0.1, "min_rep_s": 0.5},
        },
        "depth": {"max_thigh_deg": 28},
        "valgus": {"min_knee_hip_width_ratio": 0.85},
    },
    "view": {
        "side_max_shoulder_ratio": 0.3,
        "front_min_shoulder_ratio": 0.5,
        "classify_s": 2.0,
    },
    "tracking": {
        "visibility_min": 0.5,
        "max_gap_s": 0.3,
        "smooth_tau_s": 0.1,
        "standing_ref_s": 4.0,
        "low_confidence_below": 0.8,
    },
}


def _with(path: list[str], value):
    data = copy.deepcopy(VALID)
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return data


def test_repo_config_loads():
    th = load_thresholds(CONFIG_PATH)
    assert 0 < th.squat.rep_side.down < th.squat.rep_side.up <= 180
    assert 0 < th.squat.rep_front.down < th.squat.rep_front.up <= 1
    assert 0 <= th.tracking.visibility_min <= 1


def test_valid_example_parses():
    th = parse_thresholds(VALID)
    assert th.squat.depth.max_thigh_deg == 28.0
    assert th.view.classify_s == 2.0


def test_values_come_from_the_file(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(_with(["squat", "rep", "side", "down"], 111)))
    assert load_thresholds(path).squat.rep_side.down == 111.0


def test_missing_key_rejected():
    data = copy.deepcopy(VALID)
    del data["tracking"]["smooth_tau_s"]
    with pytest.raises(ValueError, match="missing keys"):
        parse_thresholds(data)


def test_typo_key_rejected():
    data = copy.deepcopy(VALID)
    data["squat"]["rep"]["side"]["dwon"] = data["squat"]["rep"]["side"].pop("down")
    with pytest.raises(ValueError, match="unknown keys"):
        parse_thresholds(data)


def test_unknown_top_level_section_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        parse_thresholds(_with(["bench"], {}))


@pytest.mark.parametrize("down", [155, 170])
def test_side_down_must_be_below_up(down):
    with pytest.raises(ValueError, match="down < up"):
        parse_thresholds(_with(["squat", "rep", "side", "down"], down))


def test_front_up_cannot_exceed_one():
    with pytest.raises(ValueError, match="up <= 1"):
        parse_thresholds(_with(["squat", "rep", "front", "up"], 1.2))


def test_view_ratios_must_be_ordered():
    with pytest.raises(ValueError, match="side_max_shoulder_ratio"):
        parse_thresholds(_with(["view", "side_max_shoulder_ratio"], 0.6))


@pytest.mark.parametrize("value", [True, "0.5", None, -1])
def test_bad_values_rejected(value):
    with pytest.raises(ValueError):
        parse_thresholds(_with(["tracking", "max_gap_s"], value))


def test_fraction_above_one_rejected():
    with pytest.raises(ValueError, match="low_confidence_below"):
        parse_thresholds(_with(["tracking", "low_confidence_below"], 1.5))


def test_wrong_version_rejected():
    with pytest.raises(ValueError, match="schema_version"):
        parse_thresholds(_with(["schema_version"], 1))
