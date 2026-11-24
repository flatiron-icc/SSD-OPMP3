"""Command-line entry point for the W-matrix generation workflow.

Run ``python cli.py --help`` for the options, or see README.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from electric_field import parse_electric_field
from workflow import run_many_workflows, run_one_workflow, validate_config
from workflow_common import SUPPORTED_SOURCES, WorkflowConfig, emit_result


_DESCRIPTION = """\
Generate the W2 and W3 matrices, the MP2/MP3 energies given by their traces
and the OP-MP3, SOP-MP3 and SSD-OPMP3 energies for one reference Slater
determinant.  SSD-OPMP3 is indicative: its x_ss is the paper's value for
--functional (hf 0, spw92 0.75, pbe 0.7, r2scan 0.6, pbe0/wb97x-v/wb97m-v 0.5),
otherwise 0.5, and is stated in the summary.

The determinant can come from a DFT or HF calculation that this workflow runs
for you (--source qchem / pyscf / molden / chk), or from a Molden file whose
orbitals are used exactly as written, with no SCF of any kind
(--source molden-direct).  The second route is how orbitals from a method the
workflow cannot run itself, such as kappa-OOMP2, are fed in.

Outputs are written next to the input file: <stem>.final.out and
<stem>.final.yaml (summary), eight <stem>.W{2,3}{alpha,beta}_{ss,os}.npy
matrices, and the semicanonical orbitals.  See README.md section 6.
"""

_EPILOG = """\
examples:
  # PySCF PBE0 reference, no external program needed
  python cli.py h2o.xyz --source pyscf --functional pbe0 --basis cc-pvdz

  # Same, but the DFT is run by Q-Chem (needs 'qchem' on PATH)
  python cli.py h2o.xyz --source qchem --functional pbe0 --basis cc-pvdz

  # Orbitals from an external program used as-is, no SCF (e.g. kappa-OOMP2)
  python cli.py koomp2.molden --source molden-direct

  # Frozen core, density fitting, and a static field
  python cli.py h2o.xyz --source pyscf --functional pbe0 --basis cc-pvdz \\
      --frozen-core fc --df --auxbasis cc-pvdz-ri --electric-field 0 0 0.001

  # Every .xyz in a directory, three at a time
  python cli.py ./inputs/ --source pyscf --functional pbe0 --basis cc-pvdz --workers 3
"""


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        help=(
            "One input file (.xyz for qchem/pyscf, .molden for molden/molden-direct, "
            ".chk for chk) or a directory of such files (batch mode)."
        ),
    )

    ref = parser.add_argument_group("reference determinant")
    ref.add_argument(
        "--source",
        choices=SUPPORTED_SOURCES,
        default="qchem",
        help=(
            "Where the reference determinant comes from (default: qchem). "
            "qchem: run Q-Chem DFT on an .xyz file, then restart PySCF from its Molden. "
            "pyscf: run PySCF DFT on an .xyz file. "
            "molden: restart PySCF DFT from a Molden file and re-converge. "
            "chk: restart PySCF DFT from a PySCF checkpoint. "
            "molden-direct: use the orbitals in a Molden file exactly as written, "
            "with NO SCF (for orbitals from methods this workflow cannot run, "
            "e.g. kappa-OOMP2)."
        ),
    )
    ref.add_argument(
        "--functional",
        default=None,
        help=(
            "Exchange-correlation functional, e.g. pbe0 or hf. Required for every "
            "source except molden-direct, where it must be omitted (no SCF is run)."
        ),
    )
    ref.add_argument(
        "--basis",
        default=None,
        help=(
            "AO basis set, e.g. cc-pvdz. Required for qchem, pyscf and chk. "
            "molden and molden-direct read the AO basis from the Molden file "
            "(for those, --basis only selects the ECP family for elements beyond Kr; "
            "without it, the cc-* family. A Molden file carries no ECP, so give "
            "--basis for a file in a def2-* basis)."
        ),
    )
    ref.add_argument(
        "--restricted", action="store_true",
        help=(
            "Use restricted open-shell (ROKS/ROHF) orbitals instead of unrestricted. "
            "With molden-direct: check that the file is a restricted open-shell "
            "determinant (reject it if spin polarized) and carry it as ROHF."
        ),
    )
    ref.add_argument(
        "--closed-shell", action="store_true",
        help=(
            "Use a restricted closed-shell (RKS/RHF) reference and the closed-shell "
            "correlation code. Spin-singlet inputs only (a non-singlet input is "
            "rejected); overrides --restricted."
        ),
    )

    corr = parser.add_argument_group("correlation treatment")
    corr.add_argument(
        "--frozen-core", default="0", dest="frozen_core",
        help=(
            "Number of frozen core orbitals per spin, or 'fc' to let PySCF infer the "
            "standard chemical core (default: 0, i.e. all electrons correlated)."
        ),
    )
    corr.add_argument(
        "--df", action="store_true",
        help="Use density fitting for the W-matrix integrals. Requires --auxbasis.",
    )
    corr.add_argument(
        "--auxbasis", default=None,
        help="Auxiliary basis for --df, e.g. cc-pvdz-ri; per-element substitution as for --basis.",
    )
    corr.add_argument(
        "--electric-field", nargs=3, type=float, metavar=("EX", "EY", "EZ"), default=None,
        help=(
            "Uniform static electric field in atomic units, applied in the SCF and "
            "in the Fock matrix used for semicanonicalization."
        ),
    )

    res = parser.add_argument_group("resources")
    res.add_argument(
        "--threads", type=int, default=8,
        help="Thread count for Q-Chem and PySCF (default: 8)",
    )
    res.add_argument(
        "--memory", type=int, default=16000,
        help="Memory in MB for Q-Chem and PySCF (default: 16000)",
    )
    res.add_argument(
        "--workers", type=int, default=None,
        help=(
            "Batch mode only: maximum number of input files processed at the same "
            "time, each as its own process with --threads threads. Default: chosen "
            "from the CPU count and --threads."
        ),
    )

    qc = parser.add_argument_group("Q-Chem only")
    qc.add_argument(
        "--qc-mp3", action="store_true", dest="qc_mp3",
        help=(
            "Also run an MP3 job in Q-Chem after the DFT job, for cross-checking by "
            "hand. Its energy is left in the Q-Chem .out file and not used otherwise."
        ),
    )
    return parser


def build_config(args: argparse.Namespace) -> WorkflowConfig:
    """Translate parsed CLI arguments into a typed configuration object."""
    return WorkflowConfig(
        input_path=Path(args.input_path).resolve(),
        source=args.source,
        functional=args.functional,
        basis=args.basis,
        threads=args.threads,
        memory=args.memory,
        frozen_core=args.frozen_core,
        df=args.df,
        auxbasis=args.auxbasis,
        restricted=args.restricted,
        qc_mp3=args.qc_mp3,
        workers=args.workers,
        electric_field=parse_electric_field(args.electric_field),
        closed_shell=args.closed_shell,
    )


def main() -> int:
    """Program entry point."""
    parser = build_parser()
    args   = parser.parse_args()
    config = build_config(args)

    # Inconsistent options are reported as a usage error.
    try:
        validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))

    if config.input_path.is_dir():
        return run_many_workflows(config)

    result      = run_one_workflow(config)
    df_auxbasis = config.auxbasis if config.df else None
    emit_result(result, df_auxbasis=df_auxbasis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
