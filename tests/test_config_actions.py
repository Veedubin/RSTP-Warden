"""Config: top-level actions, rules[].actions checks, legacy alerts.notifiers mapping (R14)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from rtsp_warden import deprecations
from rtsp_warden.config import (
    AlertsConfig,
    AppConfig,
    AppriseActionSpec,
    NtfyActionSpec,
    WebhookActionSpec,
    load_config,
)


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture one-time deprecation warnings, starting from an empty once-only registry.

    The module logger is patched directly (not caplog): Alembic's fileConfig
    disables existing loggers once any DB test has run.
    """
    messages: list[str] = []
    monkeypatch.setattr(deprecations, "_WARNED", set())
    monkeypatch.setattr(deprecations.log, "warning", lambda msg, *args: messages.append(msg % args))
    return messages


def _cam(name: str = "yard", rules: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"name": name, "main_url": "rtsp://u:p@h/m", "rules": rules or []}


LEGACY = {
    "enabled": True,
    "notifiers": [
        {
            "name": "phone",
            "type": "ntfy",
            "url": "https://ntfy.example",
            "topic": "warden-home",
            "token": "tk_legacy_value",
            "priority": 4,
            "severities": ["warn", "error"],
        },
        {
            "name": "ha",
            "type": "webhook",
            "url": "http://ha.local/hook",
            "method": "PUT",
            "headers": {"X-Api-Key": "k_test"},
            "min_interval_seconds": 10,
        },
        {
            "name": "mail",
            "type": "apprise",
            "urls": ["mailto://u:pw_test@smtp.example.com"],
            "min_severity": "warn",
            "title_template": "[{severity}] {camera_name}",
        },
    ],
}


# --- actions -----------------------------------------------------------------


def test_actions_parse_all_three_types() -> None:
    cfg = AppConfig.model_validate(
        {
            "cameras": [],
            "actions": [
                {"name": "phone", "type": "ntfy", "url": " https://ntfy.sh ", "topic": "t"},
                {"name": "ha", "type": "webhook", "url": "http://ha.local/x"},
                {"name": "mail", "type": "apprise", "urls": [" mailto://u:p@h ", ""]},
            ],
        }
    )
    phone, ha, mail = cfg.actions
    assert isinstance(phone, NtfyActionSpec)
    assert phone.url == "https://ntfy.sh"
    assert (phone.token, phone.priority) == (None, None)
    assert isinstance(ha, WebhookActionSpec)
    assert (ha.method, ha.headers) == ("POST", {})
    assert isinstance(mail, AppriseActionSpec)
    assert mail.urls == ["mailto://u:p@h"]


def test_default_config_has_no_actions_and_round_trips_silently(warnings: list[str]) -> None:
    dumped = AppConfig(cameras=[]).model_dump(mode="json")
    assert dumped["actions"] == []
    assert dumped["alerts"] == {"enabled": True, "notifiers": []}
    again = AppConfig.model_validate(yaml.safe_load(yaml.safe_dump(dumped)))
    assert again.actions == []
    assert warnings == []


@pytest.mark.parametrize(
    ("action", "needle"),
    [
        ({"name": "p", "type": "ntfy", "url": "https://n", "tpoic": "t"}, "tpoic"),
        ({"name": "p", "type": "ntfy", "url": "  "}, "url must be non-empty"),
        ({"name": "p", "type": "ntfy", "url": "https://n", "priority": 9}, "between 1 and 5"),
        ({"name": "p", "type": "apprise", "urls": [" "]}, "at least one non-empty"),
        ({"name": "p", "type": "mqtt", "url": "x"}, "does not match any of the expected tags"),
        ({"name": "a/b", "type": "webhook", "url": "http://h"}, "must not contain '/'"),
        ({"name": "x" * 65, "type": "webhook", "url": "http://h"}, "at most 64"),
        ({"name": " ", "type": "webhook", "url": "http://h"}, "must be non-empty"),
    ],
)
def test_invalid_actions_are_rejected(action: dict[str, Any], needle: str) -> None:
    with pytest.raises(ValidationError, match=needle):
        AppConfig.model_validate({"cameras": [], "actions": [action]})


def test_webhook_header_must_be_ascii_and_value_is_not_echoed() -> None:
    action = {
        "name": "ha",
        "type": "webhook",
        "url": "http://h",
        "headers": {"Authorization": "Bearer sécret_value"},
    }
    with pytest.raises(ValidationError) as excinfo:
        AppConfig.model_validate({"cameras": [], "actions": [action]})
    text = str(excinfo.value)
    assert "header 'Authorization' must be ASCII" in text
    assert "sécret_value" not in text


def test_action_errors_do_not_echo_tokens() -> None:
    action = {"name": "p", "type": "ntfy", "url": "https://n", "token": "tk_x_value", "bad": 1}
    with pytest.raises(ValidationError) as excinfo:
        AppConfig.model_validate({"cameras": [], "actions": [action]})
    assert "tk_x_value" not in str(excinfo.value)


def test_duplicate_action_names_are_rejected() -> None:
    actions = [
        {"name": "phone", "type": "ntfy", "url": "https://n", "topic": "a"},
        {"name": "phone", "type": "webhook", "url": "http://h"},
    ]
    with pytest.raises(ValidationError, match="'phone' is defined more than once in actions"):
        AppConfig.model_validate({"cameras": [], "actions": actions})


# --- rules[].actions ---------------------------------------------------------


def test_rule_naming_a_defined_action_loads() -> None:
    cfg = AppConfig.model_validate(
        {
            "cameras": [_cam(rules=[{"name": "people", "actions": ["phone"]}])],
            "actions": [{"name": "phone", "type": "ntfy", "url": "https://n", "topic": "t"}],
        }
    )
    assert cfg.cameras[0].rules[0].actions == ["phone"]


def test_rule_naming_an_unknown_action_fails_and_says_where() -> None:
    raw = {
        "cameras": [_cam(rules=[{"name": "people", "actions": ["phone", "pager"]}])],
        "actions": [{"name": "phone", "type": "ntfy", "url": "https://n", "topic": "t"}],
    }
    with pytest.raises(ValidationError) as excinfo:
        AppConfig.model_validate(raw)
    text = str(excinfo.value)
    assert "camera 'yard' rule 'people' names unknown action 'pager'" in text
    assert "defined actions: phone" in text


def test_rule_naming_a_disabled_legacy_notifier_gets_a_hint(warnings: list[str]) -> None:
    raw = {
        "cameras": [_cam(rules=[{"name": "people", "actions": ["old"]}])],
        "alerts": {
            "notifiers": [{"name": "old", "type": "webhook", "url": "http://h", "enabled": False}]
        },
    }
    with pytest.raises(ValidationError) as excinfo:
        AppConfig.model_validate(raw)
    text = str(excinfo.value)
    assert "unknown action 'old' (defined actions: none)" in text
    assert "'old' is under alerts.notifiers with enabled: false" in text


def test_rule_may_name_a_migrated_legacy_notifier(warnings: list[str]) -> None:
    raw = {"cameras": [_cam(rules=[{"name": "people", "actions": ["phone"]}])], "alerts": LEGACY}
    cfg = AppConfig.model_validate(raw)
    assert [a.name for a in cfg.actions] == ["phone", "ha", "mail"]


# --- legacy alerts.notifiers -------------------------------------------------


def test_legacy_notifiers_move_to_actions_with_one_warning(warnings: list[str]) -> None:
    cfg = AppConfig.model_validate({"cameras": [], "alerts": LEGACY})

    phone, ha, mail = cfg.actions
    assert phone == NtfyActionSpec(
        name="phone",
        type="ntfy",
        url="https://ntfy.example",
        topic="warden-home",
        token="tk_legacy_value",
        priority=4,
    )
    assert ha == WebhookActionSpec(
        name="ha",
        type="webhook",
        url="http://ha.local/hook",
        method="PUT",
        headers={"X-Api-Key": "k_test"},
    )
    assert mail == AppriseActionSpec(
        name="mail", type="apprise", urls=["mailto://u:pw_test@smtp.example.com"]
    )
    assert cfg.alerts.notifiers == []

    assert len(warnings) == 1
    msg = warnings[0]
    assert "alerts.notifiers is deprecated" in msg
    assert "moved to actions: phone, ha, mail" in msg
    for key in (
        "phone.severities",
        "ha.min_interval_seconds",
        "mail.min_severity",
        "mail.title_template",
    ):
        assert key in msg
    assert "tk_legacy_value" not in msg
    assert "left in alerts.notifiers" not in msg


def test_legacy_warning_is_logged_once_per_process(warnings: list[str]) -> None:
    AppConfig.model_validate({"cameras": [], "alerts": LEGACY})
    AppConfig.model_validate({"cameras": [], "alerts": LEGACY})
    assert len(warnings) == 1


def test_disabled_legacy_entries_stay_where_they_are(warnings: list[str]) -> None:
    alerts = {
        "notifiers": [
            {"name": "phone", "type": "ntfy", "url": "https://n", "topic": "t"},
            {"name": "old", "type": "webhook", "url": "http://h", "enabled": False},
        ]
    }
    cfg = AppConfig.model_validate({"cameras": [], "alerts": alerts})
    assert [a.name for a in cfg.actions] == ["phone"]
    assert [n.name for n in cfg.alerts.notifiers] == ["old"]
    assert len(warnings) == 1
    assert "left in alerts.notifiers because enabled is false: old" in warnings[0]


def test_alerts_enabled_false_is_ignored(warnings: list[str]) -> None:
    alerts = {
        "enabled": False,
        "notifiers": [{"name": "phone", "type": "ntfy", "url": "https://n", "topic": "t"}],
    }
    cfg = AppConfig.model_validate({"cameras": [], "alerts": alerts})
    assert [a.name for a in cfg.actions] == ["phone"]
    assert "alerts.enabled: false is ignored" in warnings[0]


def test_legacy_name_colliding_with_an_action_is_an_error(warnings: list[str]) -> None:
    raw = {
        "cameras": [],
        "actions": [{"name": "phone", "type": "webhook", "url": "http://h"}],
        "alerts": LEGACY,
    }
    with pytest.raises(ValidationError, match="'phone' is defined more than once across actions"):
        AppConfig.model_validate(raw)
    assert warnings == []


def test_disabled_legacy_name_colliding_with_an_action_is_an_error(warnings: list[str]) -> None:
    raw = {
        "cameras": [],
        "actions": [{"name": "old", "type": "webhook", "url": "http://h"}],
        "alerts": {
            "notifiers": [{"name": "old", "type": "webhook", "url": "http://x", "enabled": False}]
        },
    }
    with pytest.raises(ValidationError, match="'old' is defined more than once"):
        AppConfig.model_validate(raw)


def test_legacy_entry_that_cannot_become_an_action_names_the_entry(warnings: list[str]) -> None:
    alerts = {"notifiers": [{"name": "a/b", "type": "webhook", "url": "http://h"}]}
    with pytest.raises(ValidationError) as excinfo:
        AppConfig.model_validate({"cameras": [], "alerts": alerts})
    text = str(excinfo.value)
    assert "alerts.notifiers entry 'a/b' cannot become an action" in text
    assert "must not contain '/'" in text


def test_legacy_round_trip_does_not_duplicate_or_warn_again(warnings: list[str]) -> None:
    cfg = AppConfig.model_validate({"cameras": [], "alerts": LEGACY})
    assert len(warnings) == 1
    dumped = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    again = AppConfig.model_validate(yaml.safe_load(dumped))
    assert [a.name for a in again.actions] == ["phone", "ha", "mail"]
    assert again.actions == cfg.actions
    assert len(warnings) == 1


def test_constructor_path_does_not_mutate_the_callers_alerts(warnings: list[str]) -> None:
    alerts = AlertsConfig.model_validate(LEGACY)
    cfg = AppConfig(cameras=[], alerts=alerts)
    assert [a.name for a in cfg.actions] == ["phone", "ha", "mail"]
    assert [n.name for n in alerts.notifiers] == ["phone", "ha", "mail"]
    assert cfg.alerts is not alerts


def test_load_config_maps_legacy_yaml_with_env_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    monkeypatch.setenv("NTFY_TOKEN", "tk_env_value")
    path = tmp_path / "config.yaml"
    path.write_text(
        "cameras:\n"
        "  - name: yard\n"
        "    main_url: rtsp://u:p@h/m\n"
        "alerts:\n"
        "  enabled: true\n"
        "  notifiers:\n"
        "    - name: phone\n"
        "      type: ntfy\n"
        "      url: https://ntfy.sh\n"
        "      topic: warden-home\n"
        "      token: ${NTFY_TOKEN}\n"
        "      severities: [warn, error]\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert [(a.name, a.token) for a in cfg.actions] == [("phone", "tk_env_value")]
    assert len(warnings) == 1


def test_load_config_collision_message_hides_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    monkeypatch.setenv("NTFY_TOKEN", "tk_env_value")
    path = tmp_path / "config.yaml"
    path.write_text(
        "cameras: []\n"
        "actions:\n"
        "  - name: phone\n"
        "    type: ntfy\n"
        "    url: https://ntfy.sh\n"
        "    topic: warden-home\n"
        "    token: ${NTFY_TOKEN}\n"
        "alerts:\n"
        "  notifiers:\n"
        "    - name: phone\n"
        "      type: ntfy\n"
        "      url: https://ntfy.sh\n"
        "      token: ${NTFY_TOKEN}\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as excinfo:
        load_config(path)
    text = str(excinfo.value)
    assert "'phone' is defined more than once" in text
    assert "tk_env_value" not in text
    assert "warden-home" not in text
