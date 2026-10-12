"""Build a source-only ZIP from an explicit allowlist, never clinical/run files."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", "DATA_FORMAT.md", "run_tft.py", "requirements.txt",
         "requirements-training.txt", "requirements-optional.txt", ".gitignore",
         "config/tft_candidates.json", "docs/paper_protocol.md",
         "docs/implementation_notes.md", "docs/data_availability.md",
         "scripts/run_paper_darts.py", "scripts/tune_subjectwise_fivefold.py",
         "scripts/check_saved_model_causality.py", "scripts/audit_completed_fivefold.py",
         "scripts/run_tft_controls.py", "scripts/fit_dej_rule.py",
         "scripts/extract_features.py", "scripts/build_clean_source.py"]


def source_files():
    paths = [ROOT / name for name in FILES]
    paths.extend(sorted((ROOT / "src/biopac_tft_oct").glob("*.py")))
    paths.extend(sorted((ROOT / "tests").glob("test_*.py")))
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe source file: {path}")
        path.resolve().relative_to(ROOT.resolve())
    return paths


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Keep earlier source packages; use a new filename.")
    paths = source_files()
    # Preserve the checked-in license without capturing unrelated local license edits.
    if (ROOT / ".git").exists():
        license_content = subprocess.check_output(["git", "show", "HEAD:LICENSE"], cwd=ROOT)
    else:
        license_content = (ROOT / "LICENSE").read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        archive.writestr("LICENSE", license_content)
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Source archive integrity check failed.")
        names = archive.namelist()
        forbidden = {".csv", ".xlsx", ".npy", ".npz", ".pkl", ".ckpt", ".pt", ".pyc"}
        if any(Path(name).suffix.lower() in forbidden for name in names):
            raise ValueError("A non-source file entered the clean package.")
    print(f"Source-only ZIP: {output}; {len(paths)+1} files; no clinical data or generated predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    build(parser.parse_args().output)
