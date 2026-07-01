from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shpb_processor.config import WorkspaceConfig, material_bar_parameters, save_workspace_config
from shpb_processor.dispersion import DispersionSettings
from shpb_processor.dispersion.correction import _cached_frequency_response, _resolved_phase_strength
from shpb_processor.models import BarParameters, PulseWindow
from shpb_processor.wave import FixedPulseWindows, PulseDetectionSettings


@dataclass(frozen=True)
class DispersionExample:
    name: str
    description: str
    incident_distance_m: float
    transmitted_distance_m: float
    bar_diameter_m: float
    phase_strength: float
    amplitude_strength: float
    rise_s: float
    plateau_s: float
    noise_std_strain: float
    random_seed: int


EXAMPLES = [
    DispersionExample(
        name="synthetic_shpb_dispersion_long_clean",
        description="Long gauge distances with a sharp clean pulse; dispersion correction should sharpen the wavefronts.",
        incident_distance_m=1.20,
        transmitted_distance_m=0.90,
        bar_diameter_m=0.0145,
        phase_strength=0.20,
        amplitude_strength=0.04,
        rise_s=8e-6,
        plateau_s=45e-6,
        noise_std_strain=0.0,
        random_seed=120,
    ),
    DispersionExample(
        name="synthetic_shpb_dispersion_long_noisy",
        description="Long gauge distances with mild gauge noise; useful for checking whether correction helps without amplifying noise excessively.",
        incident_distance_m=1.00,
        transmitted_distance_m=0.75,
        bar_diameter_m=0.0145,
        phase_strength=0.16,
        amplitude_strength=0.04,
        rise_s=8e-6,
        plateau_s=45e-6,
        noise_std_strain=8e-6,
        random_seed=121,
    ),
    DispersionExample(
        name="synthetic_shpb_dispersion_large_bar",
        description="Large-diameter bar with a steep pulse; radial dispersion is intentionally stronger.",
        incident_distance_m=0.80,
        transmitted_distance_m=0.60,
        bar_diameter_m=0.0250,
        phase_strength=0.16,
        amplitude_strength=0.04,
        rise_s=8e-6,
        plateau_s=45e-6,
        noise_std_strain=5e-6,
        random_seed=122,
    ),
]


def main() -> None:
    output_dir = ROOT / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    for example in EXAMPLES:
        dataframe, metadata, config = build_example(example)
        csv_path = output_dir / f"{example.name}.csv"
        metadata_path = output_dir / f"{example.name}_metadata.json"
        config_path = output_dir / f"{example.name}_config.json"

        dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")
        pd.Series(metadata).to_json(metadata_path, force_ascii=False, indent=2)
        save_workspace_config(config, config_path)
        print(csv_path)
        print(metadata_path)
        print(config_path)

        if importlib.util.find_spec("openpyxl") or importlib.util.find_spec("xlsxwriter"):
            xlsx_path = output_dir / f"{example.name}.xlsx"
            dataframe.to_excel(xlsx_path, index=False)
            print(xlsx_path)


def build_example(example: DispersionExample) -> tuple[pd.DataFrame, dict[str, object], WorkspaceConfig]:
    sampling_frequency_hz = 5_000_000.0
    duration_s = 1.10e-3
    interface_start_s = 4.20e-4
    fall_s = example.rise_s
    amplitude = 800e-6
    reflected_scale = -0.40
    transmitted_scale = 0.60

    bar = _bar(example)
    dispersion = DispersionSettings(
        enabled=True,
        poisson_ratio=0.30,
        phase_strength=example.phase_strength,
        amplitude_correction=True,
        amplitude_strength=example.amplitude_strength,
        taper_fraction=0.05,
    )

    time_s = np.arange(0.0, duration_s, 1.0 / sampling_frequency_hz)
    c_bar = bar.resolved_wave_speed_m_s
    incident_start_s = interface_start_s - example.incident_distance_m / c_bar
    reflected_start_s = interface_start_s + example.incident_distance_m / c_bar
    transmitted_start_s = interface_start_s + example.transmitted_distance_m / c_bar

    incident_ideal = _smooth_trapezoid(time_s, incident_start_s, example.rise_s, example.plateau_s, fall_s, amplitude)
    reflected_ideal = reflected_scale * _smooth_trapezoid(time_s, reflected_start_s, example.rise_s, example.plateau_s, fall_s, amplitude)
    transmitted_ideal = transmitted_scale * _smooth_trapezoid(
        time_s,
        transmitted_start_s,
        example.rise_s,
        example.plateau_s,
        fall_s,
        amplitude,
    )

    radius_m = 0.5 * example.bar_diameter_m
    incident_gauge = _inverse_dispersion_response(
        incident_ideal,
        1.0 / sampling_frequency_hz,
        example.incident_distance_m,
        radius_m,
        bar,
        dispersion,
    )
    reflected_gauge = _inverse_dispersion_response(
        reflected_ideal,
        1.0 / sampling_frequency_hz,
        -example.incident_distance_m,
        radius_m,
        bar,
        dispersion,
    )
    transmitted_gauge = _inverse_dispersion_response(
        transmitted_ideal,
        1.0 / sampling_frequency_hz,
        -example.transmitted_distance_m,
        radius_m,
        bar,
        dispersion,
    )

    rng = np.random.default_rng(example.random_seed)
    incident_signal = incident_gauge + reflected_gauge + rng.normal(0.0, example.noise_std_strain, len(time_s))
    transmitted_signal = transmitted_gauge + rng.normal(0.0, example.noise_std_strain, len(time_s))
    dataframe = pd.DataFrame(
        {
            "Time/us": time_s * 1e6,
            "Incident_strain(με)": incident_signal * 1e6,
            "Transmitted_strain(με)": transmitted_signal * 1e6,
        }
    )

    windows = FixedPulseWindows(
        incident=_window(incident_start_s, example.rise_s, example.plateau_s, fall_s, "incident"),
        reflected=_window(reflected_start_s, example.rise_s, example.plateau_s, fall_s, "reflected"),
        transmitted=_window(transmitted_start_s, example.rise_s, example.plateau_s, fall_s, "transmitted"),
    )
    config = WorkspaceConfig()
    config.bar = bar
    config.dispersion = dispersion
    config.pulse_detection = PulseDetectionSettings(fixed_windows=windows)
    config.output.report_title = f"Dispersion correction example - {example.name}"

    metadata = {
        "case": example.name,
        "description": example.description,
        "requires_dispersion_correction": True,
        "synthetic_generation_note": "The gauge signals were generated by inverse-applying the same approximate dispersion response used by the software, then adding optional noise.",
        "recommended_use": "Compare processing with dispersion enabled and disabled using the companion config file.",
        "sampling_frequency_hz": sampling_frequency_hz,
        "time_unit": "us",
        "strain_unit": "με",
        "incident_gauge_distance_m": example.incident_distance_m,
        "transmitted_gauge_distance_m": example.transmitted_distance_m,
        "bar_diameter_m": example.bar_diameter_m,
        "bar_wave_speed_m_s": c_bar,
        "dispersion_phase_strength": example.phase_strength,
        "dispersion_amplitude_strength": example.amplitude_strength,
        "noise_std_strain": example.noise_std_strain,
        "interface_start_s": interface_start_s,
        "incident_window_s": [windows.incident.start_s, windows.incident.end_s],
        "reflected_window_s": [windows.reflected.start_s, windows.reflected.end_s],
        "transmitted_window_s": [windows.transmitted.start_s, windows.transmitted.end_s],
    }
    return dataframe, metadata, config


def _bar(example: DispersionExample) -> BarParameters:
    base = material_bar_parameters("steel")
    return BarParameters(
        incident_diameter_m=example.bar_diameter_m,
        transmitted_diameter_m=example.bar_diameter_m,
        elastic_modulus_pa=base.elastic_modulus_pa,
        density_kg_m3=base.density_kg_m3,
        material_name="steel",
        poisson_ratio=0.30,
        wave_speed_m_s=base.resolved_wave_speed_m_s,
        incident_gauge_distance_m=example.incident_distance_m,
        transmitted_gauge_distance_m=example.transmitted_distance_m,
    )


def _inverse_dispersion_response(
    signal: np.ndarray,
    dt: float,
    distance_m: float,
    radius_m: float,
    bar: BarParameters,
    dispersion: DispersionSettings,
) -> np.ndarray:
    centered = signal - np.mean(signal)
    _, response = _cached_frequency_response(
        len(centered),
        dt,
        float(distance_m),
        float(radius_m),
        float(bar.resolved_wave_speed_m_s),
        float(dispersion.poisson_ratio or bar.poisson_ratio or 0.30),
        _resolved_phase_strength(bar, dispersion),
        bool(dispersion.amplitude_correction),
        float(dispersion.amplitude_strength),
        float(dispersion.taper_fraction),
        None,
    )
    spectrum = np.fft.rfft(centered)
    return np.fft.irfft(spectrum / response, n=len(centered)) + np.mean(signal)


def _smooth_trapezoid(
    time_s: np.ndarray,
    start_s: float,
    rise_s: float,
    plateau_s: float,
    fall_s: float,
    amplitude: float,
) -> np.ndarray:
    values = np.zeros_like(time_s, dtype=float)
    rise = (time_s >= start_s) & (time_s < start_s + rise_s)
    values[rise] = 0.5 * (1.0 - np.cos(np.pi * (time_s[rise] - start_s) / rise_s))
    plateau = (time_s >= start_s + rise_s) & (time_s < start_s + rise_s + plateau_s)
    values[plateau] = 1.0
    fall = (time_s >= start_s + rise_s + plateau_s) & (time_s < start_s + rise_s + plateau_s + fall_s)
    values[fall] = 0.5 * (1.0 + np.cos(np.pi * (time_s[fall] - (start_s + rise_s + plateau_s)) / fall_s))
    return amplitude * values


def _window(start_s: float, rise_s: float, plateau_s: float, fall_s: float, label: str) -> PulseWindow:
    padding_s = 80e-6
    return PulseWindow(
        start_s=max(0.0, start_s - padding_s),
        end_s=start_s + rise_s + plateau_s + fall_s + padding_s,
        label=label,
        confidence=1.0,
    )


if __name__ == "__main__":
    main()
