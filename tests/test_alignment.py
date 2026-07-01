import numpy as np

from shpb_processor.models import BarParameters, PulseWindow, WaveSegments
from shpb_processor.wave import AlignmentSettings, align_waves


def test_auto_alignment_corrects_known_distance_timing_offsets():
    dt = 0.25e-6
    time = np.arange(0.0, 140e-6, dt)
    c_bar = 5000.0
    incident_distance = 0.050
    transmitted_distance = 0.025
    interface_center = 70e-6
    ref_extra = 5 * dt
    tra_extra = -3 * dt

    incident = _gaussian(time, interface_center - incident_distance / c_bar)
    reflected = -0.4 * _gaussian(time, interface_center + incident_distance / c_bar + ref_extra)
    transmitted = 0.6 * _gaussian(time, interface_center + transmitted_distance / c_bar + tra_extra)
    segments = _segments(time, incident, reflected, transmitted)
    bar = _bar(c_bar, incident_distance, transmitted_distance)

    aligned = align_waves(segments, bar, AlignmentSettings(min_overlap_points=12))

    assert aligned.metadata["auto_alignment_status"] == "passed"
    assert aligned.metadata["auto_alignment_final_error"] < 0.05
    assert abs(aligned.metadata["auto_alignment_reflected_delta_s"] + ref_extra) <= 2 * dt
    assert abs(aligned.metadata["auto_alignment_transmitted_delta_s"] - abs(tra_extra)) <= 2 * dt


def test_force_balance_objective_reports_balance_metrics():
    dt = 0.25e-6
    time = np.arange(0.0, 140e-6, dt)
    c_bar = 5000.0
    incident_distance = 0.050
    transmitted_distance = 0.025
    interface_center = 70e-6
    ref_extra = 4 * dt
    tra_extra = -2 * dt

    incident = _gaussian(time, interface_center - incident_distance / c_bar)
    reflected = -0.4 * _gaussian(time, interface_center + incident_distance / c_bar + ref_extra)
    transmitted = 0.6 * _gaussian(time, interface_center + transmitted_distance / c_bar + tra_extra)
    segments = _segments(time, incident, reflected, transmitted)
    bar = _bar(c_bar, incident_distance, transmitted_distance)

    aligned = align_waves(
        segments,
        bar,
        AlignmentSettings(alignment_objective="force_balance", min_overlap_points=12),
    )

    assert aligned.metadata["auto_alignment_objective"] == "force_balance"
    assert aligned.metadata["auto_alignment_final_force_balance_error"] < 0.05
    assert aligned.metadata["auto_alignment_force_balance_improvement"] > 0.0


def test_auto_alignment_searches_when_gauge_distances_are_unknown():
    dt = 0.5e-6
    time = np.arange(0.0, 160e-6, dt)
    incident_center = 50e-6
    reflected_center = 78e-6
    transmitted_center = 66e-6

    incident = _gaussian(time, incident_center)
    reflected = -0.4 * _gaussian(time, reflected_center)
    transmitted = 0.6 * _gaussian(time, transmitted_center)
    segments = _segments(time, incident, reflected, transmitted)
    bar = _bar(5000.0, 0.0, 0.0)

    aligned = align_waves(segments, bar, AlignmentSettings(min_overlap_points=12))

    assert aligned.metadata["auto_alignment_unknown_distances"] is True
    assert aligned.metadata["auto_alignment_status"] == "passed"
    assert aligned.metadata["auto_alignment_final_error"] < 0.05
    assert abs(aligned.metadata["auto_alignment_reflected_delta_s"] - (incident_center - reflected_center)) <= 2 * dt
    assert abs(aligned.metadata["auto_alignment_transmitted_delta_s"] - (incident_center - transmitted_center)) <= 2 * dt


def test_auto_alignment_marks_unreliable_when_transmitted_wave_is_too_weak():
    dt = 0.25e-6
    time = np.arange(0.0, 120e-6, dt)
    center = 50e-6
    incident = _gaussian(time, center)
    reflected = -0.4 * _gaussian(time, center)
    transmitted = np.zeros_like(time)
    segments = _segments(time, incident, reflected, transmitted)
    bar = _bar(5000.0, 0.0, 0.0)

    aligned = align_waves(segments, bar, AlignmentSettings(min_overlap_points=12))

    assert aligned.metadata["auto_alignment_status"] == "failed"
    assert aligned.metadata["auto_alignment_passed"] is False
    assert any("unreliable" in warning for warning in aligned.metadata["warnings"])


def _segments(time, incident, reflected, transmitted):
    return WaveSegments(
        time_s=time,
        incident=incident,
        reflected=reflected,
        transmitted=transmitted,
        incident_window=PulseWindow(start_s=float(time[0]), end_s=float(time[-1]), label="incident", confidence=1.0),
        reflected_window=PulseWindow(start_s=float(time[0]), end_s=float(time[-1]), label="reflected", confidence=1.0),
        transmitted_window=PulseWindow(start_s=float(time[0]), end_s=float(time[-1]), label="transmitted", confidence=1.0),
    )


def _bar(c_bar, incident_distance, transmitted_distance):
    return BarParameters(
        incident_diameter_m=0.0145,
        transmitted_diameter_m=0.0145,
        elastic_modulus_pa=200e9,
        density_kg_m3=7800.0,
        wave_speed_m_s=c_bar,
        incident_gauge_distance_m=incident_distance,
        transmitted_gauge_distance_m=transmitted_distance,
    )


def _gaussian(time, center, amplitude=800e-6, sigma=5e-6):
    return amplitude * np.exp(-0.5 * ((time - center) / sigma) ** 2)
