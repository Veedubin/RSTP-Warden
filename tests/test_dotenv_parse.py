from pathlib import Path

from rtsp_warden.cli import _load_dotenv


def test_dotenv_strips_trailing_comments_and_quotes(tmp_path: Path, monkeypatch):
    for k in ("T_A", "T_B", "T_C", "T_D"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        'T_A="127.0.0.1"   # 0.0.0.0 to expose beyond localhost\n'
        "T_B=plain # trailing comment\n"
        'T_C="has # hash inside"\n'
        "T_D='single'  # c\n"
    )
    _load_dotenv(env)
    import os

    assert os.environ["T_A"] == "127.0.0.1"
    assert os.environ["T_B"] == "plain"
    assert os.environ["T_C"] == "has # hash inside"
    assert os.environ["T_D"] == "single"
