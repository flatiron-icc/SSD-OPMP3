"""Extract the final Molden block from a Q-Chem output file.

Q-Chem appends the Molden text to the end of the log when
``MOLDEN_FORMAT = TRUE``.  This helper lifts the final appended Molden block
into a standalone ``.molden`` file.
"""

from __future__ import annotations

from pathlib import Path


START_MARKER = "======= MOLDEN-FORMATTED INPUT FILE FOLLOWS ======="
END_MARKER   = "======= END OF MOLDEN-FORMATTED INPUT FILE ======="


def qchem_out_to_molden(
    out_file:    str | Path,
    molden_file: str | Path | None = None,
) -> Path:
    """Write the last Molden block of a Q-Chem output to ``molden_file``
    (default: the output's path with suffix .molden) and return its path."""
    out_path    = Path(out_file)
    molden_path = out_path.with_suffix(".molden") if molden_file is None else Path(molden_file)

    last_block:    list[str] | None = None
    current_block: list[str] | None = None

    with open(out_path, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if START_MARKER in line:
                current_block = []
                continue
            if END_MARKER in line and current_block is not None:
                last_block    = current_block
                current_block = None
                continue
            if current_block is not None:
                current_block.append(line.rstrip())

    if last_block is None:
        raise ValueError(f"No Molden block found in {out_path}")

    molden_path.write_text("\n".join(last_block).rstrip() + "\n", encoding="utf-8")
    return molden_path
