import copy
import json

import pytest

from cv.config import CONFIG_PATH, load_thresholds, parse_thresholds

VALID = {
    "schema_version": 1,
    "squat": {
        "rep": {"down_deg": 130, "up_deg": 155, "min_rep_s": 0.5, "min_drop_deg": 25}
    },
    "tracking": {"visibility_min": 0.5, "max_gap_s": 0.3, "smooth_tau_s": 0.1},
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
    assert 0 < th.squat_rep.down_deg < th.squat_rep.up_deg <= 180
    assert 0 <= th.tracking.visibility_min <= 1


def test_values_come_from_the_file(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(_with(["squat", "rep", "down_deg"], 111)))
    assert load_thresholds(path).squat_rep.down_deg == 111.0


def test_missing_key_rejected():
    data = copy.deepcopy(VALID)
    del data["tracking"]["smooth_tau_s"]
    with pytest.raises(ValueError, match="missing keys"):
        parse_thresholds(data)


def test_typo_key_rejected():
    data = copy.deepcopy(VALID)
    data["squat"]["rep"]["dwon_deg"] = data["squat"]["rep"].pop("down_deg")
    with pytest.raises(ValueError, match="unknown keys"):
        parse_thresholds(data)


@pytest.mark.parametrize("down", [155, 170])
def test_down_must_be_below_up(down):
    with pytest.raises(ValueError, match="down_deg < up_deg"):
        parse_thresholds(_with(["squat", "rep", "down_deg"], down))


@pytest.mark.parametrize("value", [True, "0.5", None, -1])
def test_bad_values_rejected(value):
    with pytest.raises(ValueError):
        parse_thresholds(_with(["tracking", "max_gap_s"], value))


def test_visibility_above_one_rejected():
    with pytest.raises(ValueError, match="visibility_min"):
        parse_thresholds(_with(["tracking", "visibility_min"], 1.5))


def test_wrong_version_rejected():
    with pytest.raises(ValueError, match="schema_version"):
        parse_thresholds(_with(["schema_version"], 2))
