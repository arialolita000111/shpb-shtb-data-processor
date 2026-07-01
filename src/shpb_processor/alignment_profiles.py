from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shpb_processor.i18n import tr


PROFILE_SCHEMA_VERSION = 1


def load_alignment_profiles(path: Path) -> dict[str, dict[str, Any]]:
    """Load saved three-wave alignment profiles from a JSON file."""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    profiles = data.get("profiles", data) if isinstance(data, dict) else {}
    if not isinstance(profiles, dict):
        return {}

    return {
        str(name): profile
        for name, profile in profiles.items()
        if isinstance(name, str) and isinstance(profile, dict)
    }


def write_alignment_profiles(path: Path, profiles: dict[str, dict[str, Any]]) -> None:
    """Write alignment profiles atomically for normal desktop use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PROFILE_SCHEMA_VERSION,
        "profiles": dict(sorted(profiles.items(), key=lambda item: item[0].lower())),
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temp_path.replace(path)


def save_alignment_profile(path: Path, name: str, profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError(tr("profiles.error.empty_name"))
    profiles = load_alignment_profiles(path)
    profiles[clean_name] = dict(profile)
    write_alignment_profiles(path, profiles)
    return profiles


def delete_alignment_profile(path: Path, name: str) -> dict[str, dict[str, Any]]:
    profiles = load_alignment_profiles(path)
    profiles.pop(name, None)
    write_alignment_profiles(path, profiles)
    return profiles
