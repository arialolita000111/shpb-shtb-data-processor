from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_LANGUAGE = "zh_CN"
SUPPORTED_LANGUAGES = {
    "zh_CN": "中文",
    "en_US": "English",
}


class LanguageManager:
    """Load UI/log translations and persist the selected desktop language."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.locales_dir = self.project_root / "locales"
        self.settings_path = self.project_root / "config" / "settings.json"
        self._language = self._load_configured_language()
        self._catalog_cache: dict[str, dict[str, Any]] = {}

    @property
    def language(self) -> str:
        return self._language

    def set_language(self, language: str, *, save: bool = True) -> None:
        if language not in SUPPORTED_LANGUAGES:
            language = DEFAULT_LANGUAGE
        self._language = language
        self._load_catalog(language)
        if save:
            self._save_configured_language(language)

    def translate(self, key: str, **kwargs: object) -> str:
        value = self._lookup(key, self._language)
        if value is None and self._language != DEFAULT_LANGUAGE:
            value = self._lookup(key, DEFAULT_LANGUAGE)
        if value is None:
            value = key
        if kwargs:
            try:
                return str(value).format(**kwargs)
            except Exception:
                return str(value)
        return str(value)

    def translate_message(self, message: str) -> str:
        catalog = self._load_catalog(self._language)
        messages = catalog.get("_messages", {})
        if isinstance(messages, dict) and message in messages:
            return str(messages[message])
        if self._language != DEFAULT_LANGUAGE:
            fallback = self._load_catalog(DEFAULT_LANGUAGE).get("_messages", {})
            if isinstance(fallback, dict) and message in fallback:
                return str(fallback[message])
        return message

    def _lookup(self, key: str, language: str) -> Any:
        catalog = self._load_catalog(language)
        return catalog.get(key)

    def _load_catalog(self, language: str) -> dict[str, Any]:
        if language in self._catalog_cache:
            return self._catalog_cache[language]
        path = self.locales_dir / f"{language}.json"
        if not path.exists():
            self._catalog_cache[language] = {}
            return self._catalog_cache[language]
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            data = {}
        self._catalog_cache[language] = data
        return data

    def _load_configured_language(self) -> str:
        if not self.settings_path.exists():
            return DEFAULT_LANGUAGE
        try:
            with self.settings_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return DEFAULT_LANGUAGE
        language = data.get("language") if isinstance(data, dict) else None
        return language if language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE

    def _save_configured_language(self, language: str) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, object] = {}
        if self.settings_path.exists():
            try:
                with self.settings_path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    data.update(loaded)
            except Exception:
                data = {}
        data["language"] = language
        temp_path = self.settings_path.with_suffix(self.settings_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temp_path.replace(self.settings_path)


language_manager = LanguageManager()


def tr(key: str, **kwargs: object) -> str:
    return language_manager.translate(key, **kwargs)


def tr_message(message: str) -> str:
    return language_manager.translate_message(message)


def get_language() -> str:
    return language_manager.language


def set_language(language: str, *, save: bool = True) -> None:
    language_manager.set_language(language, save=save)
