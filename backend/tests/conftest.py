"""Preparación común de las pruebas."""
import pytest


@pytest.fixture(autouse=True, scope="session")
def _tablas():
    """Crea las tablas que falten antes de empezar.

    Sin esto, una tabla recién añadida al modelo no existe hasta que se arranca la app, y las pruebas
    que la usan fallan con un OperationalError que no tiene nada que ver con lo que prueban."""
    from app.db import init_db
    init_db()
