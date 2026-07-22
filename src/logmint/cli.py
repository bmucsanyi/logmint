"""The command line interface (spec section 8)."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import polars as pl

from logmint import _blobs, _read, _run, _schema, _verify

_EXIT_PROBLEMS = 1


def _ls(args: argparse.Namespace) -> int:
    """List the runs of a corpus with their status.

    Args:
        args: Parsed arguments.

    Returns:
        The exit status.

    """
    root = Path(args.root)
    rows = []
    for run in _read.scan(root):
        if args.status and args.status != run.status:
            continue
        config = json.loads(
            (root / "runs" / run.rid / "run.json").read_text(encoding="utf-8")
        )
        rows.append({"run": run.rid, "status": run.status, **config})
    if rows:
        with pl.Config(tbl_rows=-1, tbl_cols=-1):
            print(pl.DataFrame(rows))
    print(json.dumps(_read.census(root)))
    return 0


def _show(args: argparse.Namespace) -> int:
    """Print one run: its config, its attempts, and its terminal metrics.

    Args:
        args: Parsed arguments.

    Returns:
        The exit status.

    """
    root = Path(args.root)
    directory = _run.run_dir(root, args.run)
    if not directory.is_dir():
        print(f"no such run: {args.run}", file=sys.stderr)
        return _EXIT_PROBLEMS
    print(f"run     {args.run}")
    print(f"config  {(directory / 'run.json').read_text(encoding='utf-8')}")
    for path in _run.attempt_paths(directory):
        records = _run.records(path)
        start = next((r for r in records if r.get("kind") == "start"), {})
        end = next((r for r in records if r.get("kind") == "status"), {})
        print(
            f"attempt {_run.attempt_index(path)}  "
            f"git={start.get('git')} host={start.get('host')} "
            f"resumed_from={start.get('resumed_from')} "
            f"status={end.get('status', 'incomplete')} wall_s={end.get('wall_s')}"
        )
    return 0


def _query(args: argparse.Namespace) -> int:
    """Run SQL against the corpus.

    Args:
        args: Parsed arguments.

    Returns:
        The exit status.

    """
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(_read.query(args.root, args.sql))
    return 0


def _verify_cmd(args: argparse.Namespace) -> int:
    """Check every invariant, exiting nonzero on any error.

    Args:
        args: Parsed arguments.

    Returns:
        The exit status.

    """
    problems = _verify.verify(args.root, blobs=args.blobs)
    for problem in problems:
        print(problem)
    errors = sum(p.level == _verify.ERROR for p in problems)
    print(f"{len(problems)} problem(s), {errors} error(s)")
    return _EXIT_PROBLEMS if errors else 0


def _gc(args: argparse.Namespace) -> int:
    """Delete unreachable blobs.

    Args:
        args: Parsed arguments.

    Returns:
        The exit status.

    """
    root = Path(args.root)
    abandoned = len(_blobs.staged(root))
    removed = _blobs.collect(root, dry_run=args.dry_run, grace_s=args.grace)
    for ref in removed:
        print(("would remove " if args.dry_run else "removed ") + ref)
    verb = "would remove" if args.dry_run else "removed"
    print(
        f"{len(removed)} unreachable blob(s), {verb} "
        f"{abandoned} abandoned staging file(s)"
    )
    return 0


def _schema_cmd(args: argparse.Namespace) -> int:
    """Regenerate SCHEMA.md.

    Args:
        args: Parsed arguments.

    Returns:
        The exit status.

    """
    print(_schema.write(args.root))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser.

    """
    parser = argparse.ArgumentParser(prog="logmint", description=__doc__)
    parser.add_argument("--version", action="version", version=version("logmint"))
    parser.add_argument("--root", default=".", help="the corpus root")
    sub = parser.add_subparsers(dest="command", required=True)

    ls = sub.add_parser("ls", help="list runs")
    ls.add_argument("--status", choices=["finished", "failed", "incomplete"])
    ls.set_defaults(fn=_ls)

    show = sub.add_parser("show", help="show one run")
    show.add_argument("run")
    show.set_defaults(fn=_show)

    query = sub.add_parser("query", help="run SQL against the corpus")
    query.add_argument("sql")
    query.set_defaults(fn=_query)

    verify = sub.add_parser("verify", help="check every invariant")
    verify.add_argument("--blobs", action="store_true", help="re-hash blob contents")
    verify.set_defaults(fn=_verify_cmd)

    gc = sub.add_parser("gc", help="delete unreachable blobs")
    gc.add_argument("--dry-run", action="store_true")
    gc.add_argument(
        "--grace",
        type=float,
        default=_blobs.GRACE_S,
        metavar="SECONDS",
        help="leave blobs written more recently than this alone (default: one hour), "
        "because a blob is written before the record that references it",
    )
    gc.set_defaults(fn=_gc)

    schema = sub.add_parser("schema", help="regenerate SCHEMA.md")
    schema.set_defaults(fn=_schema_cmd)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Arguments, defaulting to the process arguments.

    Returns:
        The exit status.

    """
    args = build_parser().parse_args(argv)
    status: int = args.fn(args)
    return status


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover -- the module-as-script entry
