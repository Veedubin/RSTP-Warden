from rtsp_warden.db.bootstrap import ensure_admin_user
from rtsp_warden.db.schema import list_users


def test_creates_admin_from_env_when_no_users(clean_db):
    created = ensure_admin_user(
        {"WARDEN_ADMIN_USERNAME": "ops", "WARDEN_ADMIN_PASSWORD": "pw12345678"}
    )
    assert created == ("ops", "pw12345678")
    users = list_users()
    assert [u.username for u in users] == ["ops"]
    assert users[0].role == "admin"


def test_generates_password_when_env_absent(clean_db):
    created = ensure_admin_user({})
    assert created is not None
    username, password = created
    assert username == "admin"
    assert len(password) >= 12


def test_noop_when_users_exist(db_with_user):
    assert ensure_admin_user({}) is None


def test_bootstrap_database_can_skip_admin_creation(clean_db):
    from rtsp_warden.db.bootstrap import bootstrap_database

    assert bootstrap_database(create_admin=False) is None
    assert list_users() == []
