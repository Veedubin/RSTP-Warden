"""Tests for web/env_file.upsert_env_vars (the .env writer used for camera credentials)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from rtsp_warden.cli import _load_dotenv_file, _parse_dotenv_value
from rtsp_warden.web import env_file
from rtsp_warden.web.env_file import upsert_env_vars


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_creates_file_with_mode_0600(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    upsert_env_vars(env, {"CAM_FRONT_USER": "admin", "CAM_FRONT_PASS": "pw"})

    assert env.read_text() == 'CAM_FRONT_USER="admin"\nCAM_FRONT_PASS="pw"\n'
    assert _mode(env) == 0o600


def test_replaces_existing_keys_in_place_and_appends_new_ones(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# local settings\n"
        'WARDEN_DB_URL="sqlite:///x.db"\n'
        "\n"
        'CAM_FRONT_USER="old"\n'
        "CAM_FRONT_PASS=old # comment\n"
    )

    upsert_env_vars(env, {"CAM_FRONT_PASS": "new", "CAM_BACK_USER": "b"})

    assert env.read_text() == (
        "# local settings\n"
        'WARDEN_DB_URL="sqlite:///x.db"\n'
        "\n"
        'CAM_FRONT_USER="old"\n'
        'CAM_FRONT_PASS="new"\n'
        'CAM_BACK_USER="b"\n'
    )


def test_later_duplicates_of_a_replaced_key_are_dropped(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text('K="1"\nOTHER="x"\nK="2"\n')

    upsert_env_vars(env, {"K": "3"})

    assert env.read_text() == 'K="3"\nOTHER="x"\n'


def test_existing_world_readable_file_becomes_0600(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text('A="1"\n')
    env.chmod(0o644)

    upsert_env_vars(env, {"B": "2"})

    assert _mode(env) == 0o600
    assert env.read_text() == 'A="1"\nB="2"\n'


def test_quotes_and_backslashes_round_trip_through_the_cli_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    value = 'q"uo\\te # not a comment'
    upsert_env_vars(env, {"RW_T_ESC": value})

    line = env.read_text().splitlines()[0]
    assert line == 'RW_T_ESC="q\\"uo\\\\te # not a comment"'
    assert _parse_dotenv_value(line.partition("=")[2]) == value

    monkeypatch.setenv("RW_T_ESC", "placeholder")
    monkeypatch.delenv("RW_T_ESC")
    _load_dotenv_file(env)
    assert os.environ["RW_T_ESC"] == value


def test_percent_encoded_password_is_stored_verbatim(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    upsert_env_vars(env, {"CAM_FRONT_PASS": "s3cr%2Ft%23%3F%40%25"})

    line = env.read_text().splitlines()[0]
    assert _parse_dotenv_value(line.partition("=")[2]) == "s3cr%2Ft%23%3F%40%25"


@pytest.mark.parametrize(
    "values",
    [{"BAD-NAME": "x"}, {"1ABC": "x"}, {"OK": "two\nlines"}, {"OK": "cr\rhere"}],
)
def test_rejects_bad_keys_and_multiline_values_without_touching_the_file(
    tmp_path: Path, values: dict[str, str]
) -> None:
    env = tmp_path / ".env"
    env.write_text('KEEP="1"\n')

    with pytest.raises(ValueError):
        upsert_env_vars(env, values)

    assert env.read_text() == 'KEEP="1"\n'


def test_empty_mapping_is_a_no_op(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    upsert_env_vars(env, {})
    assert not env.exists()


def test_uses_a_lock_file_and_leaves_no_temp_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    upsert_env_vars(env, {"A": "1"})

    assert (tmp_path / ".env.lock").exists()
    assert not (tmp_path / ".env.tmp").exists()


def test_values_are_never_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[str] = []

    def capture(msg: str, *args: object) -> None:
        messages.append(msg % args)

    for level in ("debug", "info", "warning", "error"):
        monkeypatch.setattr(env_file.log, level, capture)

    upsert_env_vars(tmp_path / ".env", {"CAM_FRONT_PASS": "hunter2-secret"})

    assert messages, "expected one log line naming the keys"
    assert all("hunter2-secret" not in m for m in messages)
    assert any("CAM_FRONT_PASS" in m for m in messages)
