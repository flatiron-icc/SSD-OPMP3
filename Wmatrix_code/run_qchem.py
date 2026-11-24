"""--source qchem: write the Q-Chem SCF input for an XYZ file, run it, and
extract the Molden orbitals that the PySCF SCF restarts from.  The job is
unrestricted, restricted open-shell (``roscf true``, which still prints both
spin blocks) or, with closed_shell, restricted closed-shell.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

from electric_field import ElectricField, qchem_multipole_field_block
from qchem_out_to_molden import qchem_out_to_molden
from workflow_common import WorkflowPaths, derive_output_paths, read_xyz


MEM_STATIC = 1500
SEPARATOR  = "@@@@@"

# Nominal share of mem_total handed to the coupled-cluster code, and the
# margin kept free on top of mem_static.  See _cc_memory.
CC_MEMORY_FRACTION = 0.8
CC_MEMORY_MARGIN   = 100

_THANK_PATTERN = re.compile(r"Thank", flags=re.IGNORECASE)


def _render_rem_block(lines: list[str]) -> str:
    """Return one ``$rem`` block from a list of keyword lines, skipping empty ones."""
    body = "\n".join(line for line in lines if line)
    return f"$rem\n{body}\n$end\n"


def _count_matches(text: str, pattern: re.Pattern[str]) -> int:
    """Return the number of non-overlapping matches of ``pattern`` in ``text``."""
    return sum(1 for _ in pattern.finditer(text))


def _cc_memory(memory: int) -> int:
    """``cc_memory`` for the --qc-mp3 job: CC_MEMORY_FRACTION of ``memory``,
    capped so that Q-Chem's mem_total > cc_memory + mem_static holds."""
    headroom  = memory - MEM_STATIC - CC_MEMORY_MARGIN
    cc_memory = int(min(CC_MEMORY_FRACTION * memory, headroom))
    if cc_memory < 1:
        raise ValueError(
            f"--memory {memory} MB is too small for the Q-Chem MP3 job: it must "
            f"exceed mem_static ({MEM_STATIC} MB) by more than "
            f"{CC_MEMORY_MARGIN} MB."
        )
    return cc_memory


def make_qchem_input(
    charge:         int,
    multiplicity:   int,
    geometry:       str,
    functional:     str,
    basis:          str,
    memory:         int,
    frozen_core:    str | int,
    restricted:     bool,
    do_mp3:         bool,
    electric_field: ElectricField | None = None,
    closed_shell:   bool = False,
) -> str:
    """Return the complete Q-Chem input file as a text string.

    When ``do_mp3`` is True, a second job section separated by ``@@@@@``
    is appended.  The second job reads the DFT orbitals and runs MP3 with
    the specified frozen-core setting.
    """
    if closed_shell and multiplicity != 1:
        raise ValueError(
            "closed_shell=True requires multiplicity 1, "
            f"but the input has multiplicity {multiplicity}."
        )
    # A closed-shell job is a restricted one.
    restricted = restricted or closed_shell
    field_block = qchem_multipole_field_block(electric_field)

    first_rem_lines = [
        f"method {functional}",
        "print_orbitals 1000000000",
        "molden_format true",
        "symmetry false",
        "sym_ignore true",
        f"basis {basis}",
        "scf_algorithm diis_gdm",
        "thresh_diis_switch 6",
        "internal_stability_iter 15",
        "internal_stability_davidson_iter 1000",
        "internal_stability_roots 1",
        "internal_stability_conv 5",
        "max_scf_cycles 1000",
        "scf_convergence 9",
        f"mem_total {int(memory)}",
        f"mem_static {MEM_STATIC}",
        "cc_symmetry false",
    ]
    if closed_shell:
        first_rem_lines.append("unrestricted false")
    else:
        first_rem_lines.append("unrestricted true")
        first_rem_lines.append("roscf true" if restricted else "")

    first_job = (
        f"$molecule\n{charge} {multiplicity}\n{geometry}\n$end\n\n"
        f"{_render_rem_block(first_rem_lines)}"
    )
    if field_block:
        first_job += f"\n{field_block}"

    if not do_mp3:
        return first_job

    # MP3 on a restricted open-shell reference: unrestricted false with gen_scfman false.
    unrestricted_line = "unrestricted false" if restricted else "unrestricted true"
    second_rem_lines = [
        "method mp3",
        "scf_guess read",
        "max_scf_cycles 0",
        "gen_scfman false",
        unrestricted_line,
        "symmetry false",
        "sym_ignore true",
        f"basis {basis}",
        "scf_convergence 7",
        f"n_frozen_core {frozen_core}",
        f"mem_total {int(memory)}",
        f"cc_memory {_cc_memory(int(memory))}",
        f"mem_static {MEM_STATIC}",
        "cc_symmetry false",
    ]
    second_job = (
        f"{SEPARATOR}\n"
        "$molecule\nread\n$end\n\n"
        f"{_render_rem_block(second_rem_lines)}"
    )
    if field_block:
        second_job += f"\n{field_block}"

    return first_job + second_job


def _inspect_qchem_output(output_file: Path, do_mp3: bool) -> None:
    """Raise RuntimeError if the Q-Chem run did not complete successfully."""
    if not output_file.exists():
        raise RuntimeError(
            f"Q-Chem did not produce the expected output file: {output_file}"
        )
    text = output_file.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise RuntimeError(f"Q-Chem output file is empty: {output_file}")

    required = 2 if do_mp3 else 1
    if _count_matches(text, _THANK_PATTERN) < required:
        raise RuntimeError(
            f"Q-Chem did not complete successfully. See {output_file}. "
            "Inspect the Q-Chem output before using the result.\n"
        )


def run_qchem(
    geometry:       str | Path,
    threads:        int,
    memory:         int,
    functional:     str,
    basis:          str,
    frozen_core:    str | int,
    restricted:     bool,
    do_mp3:         bool,
    closed_shell:   bool = False,
    electric_field: ElectricField | None = None,
    output_paths:   WorkflowPaths | None = None,
) -> Path:
    """Write <stem>.in, run Q-Chem, check that it finished, and return the
    path of the extracted <stem>.molden."""
    xyz_path = Path(geometry).resolve()
    paths    = derive_output_paths(xyz_path) if output_paths is None else output_paths

    charge, multiplicity, geometry_block = read_xyz(xyz_path)
    qchem_input = make_qchem_input(
        charge=charge,
        multiplicity=multiplicity,
        geometry=geometry_block,
        functional=functional,
        basis=basis,
        memory=memory,
        frozen_core=frozen_core,
        restricted=restricted,
        do_mp3=do_mp3,
        electric_field=electric_field,
        closed_shell=closed_shell
    )
    paths.qchem_input.write_text(qchem_input, encoding="utf-8")

    proc = subprocess.run(
        ["qchem", "-nt", str(int(threads)), str(paths.qchem_input), str(paths.qchem_output)],
        check=False,
    )

    if proc.returncode != 0 and not paths.qchem_output.exists():
        raise RuntimeError(
            "Q-Chem exited with a non-zero status and did not produce the expected "
            f"output file. Inspect: {paths.qchem_input}, {paths.qchem_output}"
        )

    _inspect_qchem_output(paths.qchem_output, do_mp3=do_mp3)
    return qchem_out_to_molden(paths.qchem_output, molden_file=paths.qchem_molden)
