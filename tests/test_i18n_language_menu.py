import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox

from shpb_processor import i18n as i18n_module
from shpb_processor.i18n import DEFAULT_LANGUAGE, LanguageManager
from shpb_processor.ui import main_window as main_window_module
from shpb_processor.ui.main_window import MainWindow


@pytest.fixture(autouse=True)
def reset_language():
    i18n_module.set_language("en_US", save=False)
    yield
    i18n_module.set_language("en_US", save=False)


def _manager_with_settings_path(settings_path):
    manager = LanguageManager()
    manager.settings_path = settings_path
    manager._language = manager._load_configured_language()
    return manager


def test_default_language_is_english_without_saved_config(tmp_path):
    manager = _manager_with_settings_path(tmp_path / "settings.json")

    assert DEFAULT_LANGUAGE == "en_US"
    assert manager.language == "en_US"
    assert manager.translate("app.title") == "SHPB/SHTB Data Processor"


def test_invalid_saved_language_falls_back_to_english(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"language": "de_DE"}), encoding="utf-8")

    manager = _manager_with_settings_path(settings_path)

    assert manager.language == "en_US"


def test_language_menu_uses_cross_language_title_and_bilingual_items(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        main_window_module,
        "set_language",
        lambda language: i18n_module.set_language(language, save=False),
    )

    window = MainWindow()
    try:
        assert window._language_menu.title() == "语言"
        assert _language_action_texts(window) == ["中文(Chinese)", "English(英文)"]
        assert window._language_actions["en_US"].isChecked()

        window._change_language("zh_CN")
        assert window._language_menu.title() == "Language"
        assert _language_action_texts(window) == ["中文(Chinese)", "English(英文)"]
        assert window._language_actions["zh_CN"].isChecked()

        window._change_language("en_US")
        assert window._language_menu.title() == "语言"
        assert _language_action_texts(window) == ["中文(Chinese)", "English(英文)"]
        assert window._language_actions["en_US"].isChecked()
    finally:
        window.close()
        app.processEvents()


def _language_action_texts(window):
    return [window._language_actions[code].text() for code in ("zh_CN", "en_US")]
