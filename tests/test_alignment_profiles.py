from shpb_processor.alignment_profiles import (
    delete_alignment_profile,
    load_alignment_profiles,
    save_alignment_profile,
)


def test_alignment_profiles_round_trip(tmp_path):
    path = tmp_path / "alignment_profiles.json"
    profile = {
        "windows_us": {
            "incident_start_us": 10.0,
            "incident_end_us": 20.0,
            "reflected_start_us": 30.0,
            "reflected_end_us": 40.0,
            "transmitted_start_us": 50.0,
            "transmitted_end_us": 60.0,
        },
        "alignment": {"auto_micro_adjust": True, "alignment_objective": "force_balance"},
    }

    save_alignment_profile(path, "steel setup", profile)
    profiles = load_alignment_profiles(path)

    assert profiles["steel setup"]["windows_us"]["reflected_start_us"] == 30.0
    assert profiles["steel setup"]["alignment"]["alignment_objective"] == "force_balance"


def test_alignment_profiles_delete(tmp_path):
    path = tmp_path / "alignment_profiles.json"
    save_alignment_profile(path, "setup A", {"windows_us": {}})
    save_alignment_profile(path, "setup B", {"windows_us": {}})

    profiles = delete_alignment_profile(path, "setup A")

    assert "setup A" not in profiles
    assert "setup B" in load_alignment_profiles(path)
