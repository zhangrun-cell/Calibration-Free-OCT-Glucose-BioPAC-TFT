"""Freeze the site-specific DEJ mapping using only a documented discovery table."""
import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biopac_tft_oct.pipeline import save_rule, validate_cohorts
from biopac_tft_oct.rule_discovery import fit_dej_guided_dynamic_signal_extraction_rule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-table", required=True, type=Path)
    parser.add_argument("--cohort-manifest", required=True, type=Path)
    parser.add_argument("--output-rule", required=True, type=Path)
    parser.add_argument("--require-paper-counts", action="store_true")
    args = parser.parse_args()
    manifest = pd.read_csv(args.cohort_manifest, dtype=str)
    validate_cohorts(manifest, require_paper_counts=args.require_paper_counts)
    table = pd.read_csv(args.discovery_table)
    expected = manifest.loc[manifest.cohort.eq("discovery"), ["subject", "session"]]
    if not {"subject", "session"}.issubset(table.columns) or table.session.duplicated().any():
        raise ValueError("Discovery table requires unique session IDs and participant IDs.")
    actual_ids = set(zip(table.subject.astype(str), table.session.astype(str)))
    if actual_ids != set(zip(expected.subject, expected.session)):
        raise ValueError("Discovery table membership does not match the locked cohort manifest.")
    rule = fit_dej_guided_dynamic_signal_extraction_rule(table, dej_depth_col="thickness_scale")
    save_rule(rule, args.output_rule, discovery_subjects=expected.subject, discovery_sessions=expected.session)
    print("Discovery-only DEJ mapping fitted and saved. Prediction labels were not used.")


if __name__ == "__main__":
    main()
