"""Run directLFQ in a child process and persist the result for the parent process."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        raise SystemExit("Usage: python -m mstocirc2.differential_expression.directlfq_bridge <input.pkl> <output.pkl>")

    input_path = Path(args[0])
    output_path = Path(args[1])
    sys.stderr.write(f"[directlfq_bridge] input={input_path} output={output_path}\n")
    sys.stderr.flush()

    import directlfq.protein_intensity_estimation as pie
    sys.stderr.write("[directlfq_bridge] imported directlfq\n")
    sys.stderr.flush()

    wide = pd.read_pickle(input_path)
    sys.stderr.write(f"[directlfq_bridge] loaded wide rows={len(wide)} cols={len(wide.columns)}\n")
    sys.stderr.flush()
    protein_df, _ = pie.estimate_protein_intensities(
        wide,
        min_nonan=1,
        num_samples_quadratic=10,
        num_cores=1,
    )
    sys.stderr.write(f"[directlfq_bridge] estimated protein rows={len(protein_df)}\n")
    sys.stderr.flush()
    protein_df.to_pickle(output_path)
    sys.stderr.write("[directlfq_bridge] wrote output\n")
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
