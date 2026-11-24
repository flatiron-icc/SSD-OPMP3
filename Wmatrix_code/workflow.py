"""High-level orchestration for the W-matrix generation workflow."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys

from electric_field import electric_field_to_string
from pyscf_orbitals import extract_spin_orbitals_and_occupations
from semicanonicalize import semicanonicalize
from pyscf_w_matrices import run_pyscf_w_matrices
from pyscf_dft import dft_from_molden, dft_from_xyz, reference_from_molden
from run_qchem import run_qchem
from workflow_common import (
    SCF_SOURCES,
    SUPPORTED_SOURCES,
    read_xyz,
    WorkflowConfig,
    WorkflowResult,
    build_result_files,
    check_batch_oversubscription,
    correlation_energy_map,
    derive_output_paths,
    derive_batch_paths,
    emit_result,
    list_inputs,
    resolve_frozen_core,
    suggested_default_workers,
    w_matrix_file_map,
)


def validate_config(config: WorkflowConfig) -> None:
    """Raise a descriptive error if the configuration is inconsistent."""
    if config.source not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported source mode: {config.source}")
    if config.threads < 1:
        raise ValueError("--threads must be at least 1")
    if config.memory < 1:
        raise ValueError("--memory must be positive")
    if config.workers is not None and config.workers < 1:
        raise ValueError("--workers must be at least 1")
    if config.source in SCF_SOURCES and not config.functional:
        raise ValueError(
            f"--functional is required for --source {config.source} (an SCF is run)"
        )
    if config.source == "molden-direct" and config.functional:
        raise ValueError(
            "--functional has no effect with --source molden-direct: no SCF is run "
            "and the orbitals in the Molden file are used as written. Remove the flag."
        )
    if config.source in {"qchem", "pyscf", "chk"} and not config.basis:
        raise ValueError(
            "--basis is required for every source mode except 'molden' and "
            "'molden-direct', which read the AO basis from the Molden file"
        )
    if config.source != "qchem" and config.qc_mp3:
        raise ValueError("--qc-mp3 is only meaningful in --source qchem mode")
    if config.df and not config.auxbasis:
        raise ValueError("--df requires --auxbasis to be specified")
    if not config.input_path.exists():
        raise ValueError(f"Input path does not exist: {config.input_path}")
    if config.input_path.is_file():
        suffix = config.input_path.suffix.lower()
        if config.source in {"qchem", "pyscf"} and suffix != ".xyz":
            raise ValueError("The XYZ-based source modes require a single .xyz file as input")
        if config.source in {"molden", "molden-direct"} and suffix != ".molden":
            raise ValueError(
                f"The {config.source} source mode requires a single .molden file as input"
            )
        if config.source == "chk" and suffix != ".chk":
            raise ValueError("The chk source mode requires a single .chk file as input")
        if config.closed_shell:
            _require_singlet_input(config.input_path, suffix)


def _require_singlet_input(input_path: Path, suffix: str) -> None:
    """Reject --closed-shell for a non-singlet .xyz or .chk input (Molden
    inputs are checked when loaded)."""
    if suffix == ".xyz":
        _, multiplicity, _ = read_xyz(input_path)
    elif suffix == ".chk":
        from pyscf.scf.chkfile import load_mol
        multiplicity = int(load_mol(str(input_path)).spin) + 1
    else:
        return
    if multiplicity != 1:
        raise ValueError(
            "--closed-shell requires a spin-singlet input, but "
            f"{input_path.name} has multiplicity {multiplicity}. "
            "Remove --closed-shell (use --restricted for restricted open-shell orbitals)."
        )


def _validate_frozen_core_count(ncore: int, mf_ref) -> None:
    """Check that the requested frozen-core count fits inside the occupied space."""
    res = extract_spin_orbitals_and_occupations(mf_ref)
    if len(res) == 2:
        # Closed shell: one spatial orbital set whose occupations are 0 or 2,
        # so the number of occupied orbitals is half the electron count.
        occ = res[1]
        nocc = int(round(0.5 * float(occ.sum())))
        if ncore > nocc:
            raise ValueError(
                "Requested frozen-core count exceeds the occupied-space size "
                f"(ncore={ncore}, nocc={nocc})."
            )
    else:
        assert len(res) == 4
        occ_a, occ_b = res[2:]
        nocc_a = int(round(float(occ_a.sum())))
        nocc_b = int(round(float(occ_b.sum())))
        if ncore > nocc_a or ncore > nocc_b:
            raise ValueError(
                "Requested frozen-core count exceeds the occupied-space size in at least "
                f"one spin channel (ncore={ncore}, nocc_alpha={nocc_a}, nocc_beta={nocc_b})."
            )


def run_one_workflow(config: WorkflowConfig) -> WorkflowResult:
    """Reference determinant -> semicanonicalization -> W matrices -> summary."""
    validate_config(config)
    input_path = config.input_path.resolve()
    paths      = derive_output_paths(input_path)

    if config.source == "qchem":
        molden_file = run_qchem(
            geometry=input_path,
            threads=config.threads,
            memory=config.memory,
            functional=config.functional,
            basis=str(config.basis),
            frozen_core=config.frozen_core,
            restricted=config.restricted,
            do_mp3=config.qc_mp3,
            electric_field=config.electric_field,
            output_paths=paths,
            closed_shell=config.closed_shell,
        )
        mf_dft = dft_from_molden(
            molden_file=molden_file,
            functional=config.functional,
            threads=config.threads,
            memory=config.memory,
            restricted=config.restricted,
            electric_field=config.electric_field,
            output_paths=paths,
            basis=str(config.basis),
            closed_shell=config.closed_shell,
        )
    elif config.source == "molden":
        mf_dft = dft_from_molden(
            molden_file=input_path,
            functional=config.functional,
            threads=config.threads,
            memory=config.memory,
            restricted=config.restricted,
            electric_field=config.electric_field,
            output_paths=paths,
            basis=str(config.basis),
            closed_shell=config.closed_shell,
        )
    elif config.source == "pyscf":
        mf_dft = dft_from_xyz(
            geometry=input_path,
            functional=config.functional,
            basis=str(config.basis),
            threads=config.threads,
            memory=config.memory,
            restricted=config.restricted,
            electric_field=config.electric_field,
            output_paths=paths,
            closed_shell=config.closed_shell,
        )
    elif config.source == "chk":
        mf_dft = dft_from_xyz(
            geometry=input_path,
            functional=config.functional,
            basis=str(config.basis),
            threads=config.threads,
            memory=config.memory,
            restricted=config.restricted,
            electric_field=config.electric_field,
            output_paths=paths,
            use_chkfile=True,
            closed_shell=config.closed_shell,
        )
    elif config.source == "molden-direct":
        # No SCF: the Molden orbitals are used exactly as written.
        mf_dft = reference_from_molden(
            molden_file=input_path,
            threads=config.threads,
            memory=config.memory,
            closed_shell=config.closed_shell,
            restricted=config.restricted and not config.closed_shell,
            basis=config.basis,
        )

    direct = config.source == "molden-direct"
    ncore  = resolve_frozen_core(config.frozen_core, mf_dft.mol)
    _validate_frozen_core_count(ncore, mf_dft)

    sc = semicanonicalize(
        mf_dft=mf_dft,
        input_path=input_path,
        ncore=ncore,
        electric_field=config.electric_field,
        output_paths=paths,
    )
    w_results = run_pyscf_w_matrices(
        mol=mf_dft.mol,
        semicanonical=sc,
        input_path=input_path,
        ncore=ncore,
        df=config.df,
        auxbasis=config.auxbasis,
        memory=config.memory,
        output_paths=paths,
    )

    result = WorkflowResult(
        input_path=input_path,
        source=config.source,
        electric_field=config.electric_field,
        ncore=ncore,
        files=build_result_files(
            paths, sc,
            include_pyscf=not direct,
            include_qchem=config.source == "qchem",
        ),
        w_matrix_files=w_matrix_file_map(w_results),
        energies=correlation_energy_map(
            sc, w_results,
            "Reference energy (HF on Molden density)" if direct
            else "Reference energy (HF on KS density)",
        ),
    )

    # Written here too, so that callers other than cli.py get the summary.
    emit_result(
        result,
        df_auxbasis=config.auxbasis if config.df else None,
        print_stdout=False,
    )
    return result


def _build_child_command(config: WorkflowConfig, input_file: Path) -> list[str]:
    """The cli.py command of one batch-mode child, with every setting of ``config``."""
    cli_path = Path(__file__).resolve().with_name("cli.py")
    cmd = [
        sys.executable, str(cli_path), str(input_file),
        "--source",      config.source,
        "--threads",     str(config.threads),
        "--memory",      str(config.memory),
        "--frozen-core", str(config.frozen_core),
    ]
    if config.functional is not None:
        cmd.extend(["--functional", config.functional])
    if config.basis is not None:
        cmd.extend(["--basis", config.basis])
    if config.df:
        cmd.append("--df")
    if config.auxbasis is not None:
        cmd.extend(["--auxbasis", config.auxbasis])
    if config.restricted:
        cmd.append("--restricted")
    if config.closed_shell:
        cmd.append("--closed-shell")
    if config.qc_mp3:
        cmd.append("--qc-mp3")
    if config.electric_field is not None:
        cmd.extend(["--electric-field", *(str(v) for v in config.electric_field)])
    return cmd


def _run_child_job(config: WorkflowConfig, input_file: Path) -> dict:
    """Run one batch child process and capture its output to files."""
    batch = derive_batch_paths(input_file)
    cmd   = _build_child_command(config, input_file)
    with open(batch.stdout, "w", encoding="utf-8") as out_fh, \
         open(batch.stderr, "w", encoding="utf-8") as err_fh:
        proc = subprocess.run(cmd, stdout=out_fh, stderr=err_fh, text=True)
    return {
        "input":       input_file,
        "returncode":  proc.returncode,
        "stdout_file": batch.stdout,
        "stderr_file": batch.stderr,
    }


def run_many_workflows(config: WorkflowConfig) -> int:
    """Batch mode: one cli.py process per input file in the directory."""
    validate_config(config)
    inputs  = list_inputs(config.input_path, config.source)
    workers = config.workers
    if workers is None:
        workers = suggested_default_workers(config.threads, len(inputs))

    print(f"Input directory   = {config.input_path.resolve()}")
    print(f"Source mode       = {config.source}")
    print(f"Electric field    = {electric_field_to_string(config.electric_field)}")
    print(f"Discovered inputs = {len(inputs)}")
    print(f"Batch workers     = {workers}")

    for warning in check_batch_oversubscription(workers, config.threads):
        print(f"WARNING: {warning}")

    failures: list[tuple[Path, int, Path, Path]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_run_child_job, config, f): f for f in inputs
        }
        for future in as_completed(future_map):
            res = future.result()
            if res["returncode"] == 0:
                print(f"OK   {res['input'].name}")
            else:
                print(f"FAIL {res['input'].name} (exit {res['returncode']})")
                failures.append((
                    res["input"], res["returncode"],
                    res["stdout_file"], res["stderr_file"],
                ))

    print()
    print(f"Completed = {len(inputs) - len(failures)}")
    print(f"Failed    = {len(failures)}")

    if failures:
        print()
        print("Failures:")
        for input_file, returncode, stdout_file, stderr_file in failures:
            print(f"  {input_file.name}")
            print(f"    exit code : {returncode}")
            print(f"    stdout    : {stdout_file}")
            print(f"    stderr    : {stderr_file}")
        return 1
    return 0
