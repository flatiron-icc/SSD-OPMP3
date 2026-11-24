"""Argument and logging plumbing shared by the entry points."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from data import BlockSource, require_present, sources


def add_verbosity_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Debug-level logging.")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Warnings and errors only.")


def add_source_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--xyz", nargs="+", type=Path, default=[], metavar="DIR",
                   help="W-matrix trees laid out as <DIR>/<xc>/<name>.xyz/<name>.*; "
                        "searched after every --molden tree.")
    p.add_argument("--molden", nargs="+", type=Path, default=[],
                   metavar="DIR",
                   help="W-matrix trees laid out as <DIR>/<xc>/<name>.molden/<name>.*; "
                        "searched first.")
    p.add_argument("--flat", nargs="+", type=Path, default=[], metavar="DIR",
                   help="W-matrix trees laid out as <DIR>/<xc>/<name>.*; "
                        "searched last.")
    p.add_argument("--xc", required=True,
                   help="Reference determinant: the <xc> directory name "
                        "inside each tree, e.g. KOOMP2 or PBE0.")


def configure_logging(args: argparse.Namespace) -> None:
    """Bare messages on stderr; -q wins over -v."""
    level = (logging.WARNING if args.quiet else
             logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr,
                        force=True)


def resolve_sources(args: argparse.Namespace) -> list[BlockSource]:
    """The --molden / --xyz / --flat trees, checked to exist."""
    srcs = sources(args.xyz, args.molden, args.flat)
    if not srcs:
        raise SystemExit("error: give at least one --molden, --xyz or --flat tree")
    require_present("W-matrix trees not found",
                    [str(s.base_dir) for s in srcs if not s.base_dir.is_dir()])
    return srcs
