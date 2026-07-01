from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SHPB/SHTB Data Processor")
    parser.add_argument("--generate-samples", metavar="DIR", help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command")

    sample_parser = subparsers.add_parser("generate-samples", help="Generate synthetic validation datasets.")
    sample_parser.add_argument("output", help="Output directory for generated sample files.")

    batch_parser = subparsers.add_parser("batch", help="Run reproducible batch processing.")
    batch_parser.add_argument("--config", required=True, help="Workspace configuration JSON/YAML file.")
    batch_parser.add_argument("--input", required=True, help="Input directory containing CSV/XLSX data files.")
    batch_parser.add_argument("--output", required=True, help="Output directory for reports and results.")
    args = parser.parse_args(argv)

    if args.generate_samples or args.command == "generate-samples":
        from shpb_processor.sample_data import write_sample_files

        output = args.generate_samples or args.output
        for path in write_sample_files(output):
            print(path)
        return 0

    if args.command == "batch":
        from shpb_processor.batch import run_batch

        output = run_batch(args.config, args.input, args.output)
        print(output)
        return 0

    try:
        from shpb_processor.ui import run_app
    except ModuleNotFoundError as exc:
        missing = exc.name
        print(
            f"Missing GUI dependency: {missing}. Install dependencies with: python -m pip install -e .",
            file=sys.stderr,
        )
        return 2
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
