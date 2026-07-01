from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from shpb_processor.config import material_bar_parameters
from shpb_processor.models import ExperimentType, SpecimenParameters, SpecimenShape


def generate_synthetic_shpb_case(
    case: str = "ideal",
    sampling_frequency_hz: float = 5_000_000.0,
    duration_s: float = 4.5e-4,
    noise_std: float = 0.0,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    rng = np.random.default_rng(random_seed)
    bar = material_bar_parameters("steel")
    specimen = SpecimenParameters(
        shape=SpecimenShape.CYLINDER,
        diameter_m=0.010,
        length_m=0.008,
        experiment_type=ExperimentType.COMPRESSION,
        specimen_id=f"synthetic_{case}",
        material_name="synthetic_material",
    )

    dt = 1.0 / sampling_frequency_hz
    time_s = np.arange(0.0, duration_s, dt)
    c_bar = bar.resolved_wave_speed_m_s
    interface_start_s = 1.60e-4
    pulse_width_s = 6.0e-5
    amplitude = 800e-6

    shape = _smooth_pulse
    interface_time = time_s
    interface_incident = amplitude * shape(interface_time, interface_start_s, pulse_width_s)
    interface_reflected = -0.40 * amplitude * shape(interface_time, interface_start_s, pulse_width_s)
    transmitted_scale = 0.60
    if case == "unbalanced":
        transmitted_scale = 0.42
    interface_transmitted = transmitted_scale * amplitude * shape(interface_time, interface_start_s, pulse_width_s)

    if case == "overlap":
        incident_distance = 0.18
        reflected_distance = 0.18
    else:
        incident_distance = bar.incident_gauge_distance_m
        reflected_distance = bar.incident_gauge_distance_m
    transmitted_distance = bar.transmitted_gauge_distance_m

    incident_gauge = amplitude * shape(time_s, interface_start_s - incident_distance / c_bar, pulse_width_s)
    reflected_gauge = -0.40 * amplitude * shape(time_s, interface_start_s + reflected_distance / c_bar, pulse_width_s)
    transmitted_gauge = transmitted_scale * amplitude * shape(time_s, interface_start_s + transmitted_distance / c_bar, pulse_width_s)

    if case == "noisy":
        noise_std = max(noise_std, 20e-6)
    incident_signal = incident_gauge + reflected_gauge
    transmitted_signal = transmitted_gauge
    if noise_std > 0:
        incident_signal = incident_signal + rng.normal(0.0, noise_std, size=len(time_s))
        transmitted_signal = transmitted_signal + rng.normal(0.0, noise_std, size=len(time_s))

    dataframe = pd.DataFrame(
        {
            "Time/us": time_s * 1e6,
            "Incident_strain(με)": incident_signal * 1e6,
            "Transmitted_strain(με)": transmitted_signal * 1e6,
        }
    )
    metadata = {
        "case": case,
        "sampling_frequency_hz": sampling_frequency_hz,
        "time_unit": "us",
        "strain_unit": "με",
        "incident_diameter_m": bar.incident_diameter_m,
        "transmitted_diameter_m": bar.transmitted_diameter_m,
        "elastic_modulus_pa": bar.elastic_modulus_pa,
        "density_kg_m3": bar.density_kg_m3,
        "wave_speed_m_s": c_bar,
        "incident_gauge_distance_m": incident_distance,
        "transmitted_gauge_distance_m": transmitted_distance,
        "specimen_diameter_m": specimen.diameter_m or 0.0,
        "specimen_length_m": specimen.length_m,
        "expected_interface_start_s": interface_start_s,
        "expected_pulse_width_s": pulse_width_s,
        "expected_peak_incident_strain": float(np.nanmax(interface_incident)),
        "expected_peak_reflected_strain": float(np.nanmin(interface_reflected)),
        "expected_peak_transmitted_strain": float(np.nanmax(interface_transmitted)),
    }
    return dataframe, metadata


def write_sample_files(output: str | Path) -> list[Path]:
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    cases = ["ideal", "noisy", "unbalanced", "overlap"]
    for case in cases:
        dataframe, metadata = generate_synthetic_shpb_case(case)
        csv_path = output_dir / f"synthetic_shpb_{case}.csv"
        dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")
        written.append(csv_path)

        metadata_path = output_dir / f"synthetic_shpb_{case}_metadata.json"
        pd.Series(metadata).to_json(metadata_path, force_ascii=False, indent=2)
        written.append(metadata_path)

        if importlib.util.find_spec("openpyxl") or importlib.util.find_spec("xlsxwriter"):
            xlsx_path = output_dir / f"synthetic_shpb_{case}.xlsx"
            dataframe.to_excel(xlsx_path, index=False)
            written.append(xlsx_path)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SHPB validation datasets.")
    parser.add_argument("--output", default="examples", help="Output directory for generated sample files.")
    args = parser.parse_args(argv)
    written = write_sample_files(args.output)
    for path in written:
        print(path)


def _smooth_pulse(time_s: np.ndarray, start_s: float, width_s: float) -> np.ndarray:
    phase = (time_s - start_s) / width_s
    pulse = np.zeros_like(time_s, dtype=float)
    active = (phase >= 0.0) & (phase <= 1.0)
    pulse[active] = np.sin(np.pi * phase[active]) ** 2
    return pulse


if __name__ == "__main__":
    main()
