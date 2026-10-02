import pytest

from rtsp_warden.config import expand_env


def test_expand_env_replaces_braced_names():
    env = {"CAM_USER": "alice", "CAM_PASS": "s3cret"}
    data = {"cameras": [{"main_url": "rtsp://${CAM_USER}:${CAM_PASS}@10.0.0.5:554/x"}]}
    out = expand_env(data, env)
    assert out["cameras"][0]["main_url"] == "rtsp://alice:s3cret@10.0.0.5:554/x"


def test_expand_env_leaves_literal_dollars_alone():
    env = {"CAM_PASS": "pa$$word"}
    data = {"u": "rtsp://a:${CAM_PASS}@h/x", "v": "cost is $5 and $$"}
    out = expand_env(data, env)
    assert out["u"] == "rtsp://a:pa$$word@h/x"
    assert out["v"] == "cost is $5 and $$"


def test_expand_env_walks_lists_and_nested_dicts_and_keeps_non_strings():
    env = {"N": "front"}
    data = {"a": [{"name": "${N}", "port": 554, "flag": True}], "b": None}
    out = expand_env(data, env)
    assert out == {"a": [{"name": "front", "port": 554, "flag": True}], "b": None}


def test_expand_env_missing_variable_is_fatal():
    with pytest.raises(SystemExit) as ei:
        expand_env({"x": "${NOPE_MISSING}"}, {})
    assert "NOPE_MISSING" in str(ei.value)
