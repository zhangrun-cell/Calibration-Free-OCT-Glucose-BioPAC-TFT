"""Extract five causal OCT signals from surface-aligned, laterally averaged profiles."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biopac_tft_oct.pipeline import load_rule, sequential_signal_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True, type=Path, help="NumPy (time,depth) intensity array.")
    parser.add_argument("--rule", required=True, type=Path)
    parser.add_argument("--site", choices=["arm", "finger"], required=True)
    parser.add_argument("--epidermal-range", type=int, nargs=2, required=True)
    parser.add_argument("--dej-range", type=int, nargs=2, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-segments", type=int, default=10)
    parser.add_argument("--max-shift", type=int, default=30)
    parser.add_argument("--epidermis-depth", type=int, default=10)
    args = parser.parse_args()
    rule, provenance = load_rule(args.rule)
    if args.subject in provenance["discovery_subjects"] or args.session in provenance["discovery_sessions"]:
        raise ValueError("Prediction extraction cannot use a discovery participant/session.")
    if args.output.exists():
        raise FileExistsError(args.output)
    features, peaks, centres = sequential_signal_features(
        np.load(args.profiles, allow_pickle=False), rule=rule, site=args.site,
        epidermal_range=args.epidermal_range, dej_range=args.dej_range,
        n_segments=args.n_segments, max_shift=args.max_shift, epidermis_depth=args.epidermis_depth)
    frame = pd.DataFrame(features, columns=[f"slopmean{i}" for i in range(1, 6)])
    frame[["epidermal_peak", "dej_peak", "thickness_scale"]] = peaks
    frame[[f"window_center_{i}" for i in range(1, 6)]] = centres
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print("Sequential OCT features written; reference blood glucose was not an extraction input.")


if __name__ == "__main__":
    main()
