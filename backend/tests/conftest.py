"""Shared test setup."""
import pytest


@pytest.fixture(autouse=True, scope="session")
def _tables():
    """Creates any missing tables before starting.

    Without this, a table just added to the model doesn't exist until the app starts, and the tests
    that use it fail with an OperationalError that has nothing to do with what they test."""
    from app.db import init_db
    init_db()
