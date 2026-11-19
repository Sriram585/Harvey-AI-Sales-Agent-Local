"""Shared test setup."""

import os

os.environ.pop("SELLBIRD_HOME", None)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _private_dashboard_token(tmp_path_factory):
    """Never create or read the checkout's real data/dashboard-token."""
    import sellbird.dashboard as dash

    dash.token_store.configure(tmp_path_factory.mktemp("token") / "dashboard-token")
    yield
    dash.token_store.configure(dash.TOKEN_FILE)
