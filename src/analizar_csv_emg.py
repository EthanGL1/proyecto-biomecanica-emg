"""Compatibility entry point for descriptive EMG processing.

Use this name for older CSV-only commands; the shared CLI now accepts one
formal CSV or one ZIP containing CSV recordings. No predictive model is loaded.
"""

from pathlib import Path

from analizar_zip_emg import (
    ArchiveSafetyError,
    ZipRecording,
    default_output_dir,
    discover_single_csv,
    main,
    process_single_csv,
)

InputValidationError = ArchiveSafetyError
CsvRecording = ZipRecording

if __name__ == "__main__":
    raise SystemExit(main())