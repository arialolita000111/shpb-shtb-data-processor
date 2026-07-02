import pytest

from shpb_processor import i18n


@pytest.fixture(autouse=True)
def default_test_language():
    i18n.set_language("en_US", save=False)
    yield
    i18n.set_language("en_US", save=False)
