from pathlib import Path

import yaml

from rtsp_warden.web.routes.cameras import _persist_camera_field


def test_persist_only_touches_named_camera(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "cameras": [
                    {
                        "name": "a",
                        "main_url": "rtsp://x/a",
                        "sub_url": "rtsp://x/a2",
                        "sensitivity": 10,
                    },
                    {
                        "name": "b",
                        "main_url": "rtsp://x/b",
                        "sub_url": "rtsp://x/b2",
                        "sensitivity": 20,
                    },
                ]
            }
        )
    )
    _persist_camera_field(cfg, "b", "sensitivity", 75.0)
    data = yaml.safe_load(cfg.read_text())
    assert data["cameras"][0]["sensitivity"] == 10
    assert data["cameras"][1]["sensitivity"] == 75.0
