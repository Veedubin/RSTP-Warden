import os
from pathlib import Path

import pytest

from rtsp_warden.cli import _load_dotenv_file, _parse_dotenv_value


def _isolate(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Unset ``names`` and make monkeypatch restore (or delete) them after the test."""
    for name in names:
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


def test_dotenv_strips_trailing_comments_and_quotes(tmp_path: Path, monkeypatch):
    _isolate(monkeypatch, "T_A", "T_B", "T_C", "T_D")
    env = tmp_path / ".env"
    env.write_text(
        'T_A="127.0.0.1"   # 0.0.0.0 to expose beyond localhost\n'
        "T_B=plain # trailing comment\n"
        'T_C="has # hash inside"\n'
        "T_D='single'  # c\n"
    )
    _load_dotenv_file(env)

    assert os.environ["T_A"] == "127.0.0.1"
    assert os.environ["T_B"] == "plain"
    assert os.environ["T_C"] == "has # hash inside"
    assert os.environ["T_D"] == "single"


def test_double_quoted_value_unescapes_quote_and_backslash():
    assert _parse_dotenv_value(r'"a\"b"') == 'a"b'
    assert _parse_dotenv_value(r'"c\\d"') == "c\\d"


def test_escaped_quote_does_not_end_the_value():
    assert _parse_dotenv_value(r'"x\" # kept" # comment') == 'x" # kept'


def test_other_backslashes_are_kept():
    assert _parse_dotenv_value(r'"C:\path\n"') == r"C:\path\n"


def test_unterminated_double_quote_keeps_the_rest():
    assert _parse_dotenv_value('"open') == "open"


def test_single_quotes_stay_verbatim():
    assert _parse_dotenv_value(r"'a\"b'") == r"a\"b"
