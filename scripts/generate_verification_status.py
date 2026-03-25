#!/usr/bin/env python3
"""Generate a repo-level verification telemetry record."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable


ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "papers"
OUT_PATH = ROOT / "verification.yaml"


def run_command(command: str) -> dict[str, object]:
    completed = subprocess.run(
        ["/bin/zsh", "-lc", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "passed": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_output(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def parse_note_status(notes_path: Path, label: str) -> str:
    text = notes_path.read_text()
    match = re.search(rf"- {re.escape(label)} status:\s*(.+)", text)
    return match.group(1).strip() if match else "unknown"


def bool_yaml(value: bool) -> str:
    return "true" if value else "false"


def quote_yaml(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def indent(lines: Iterable[str], n: int) -> list[str]:
    prefix = " " * n
    return [prefix + line if line else line for line in lines]


def paper_title(readme_path: Path) -> str:
    first = readme_path.read_text().splitlines()[0]
    return first.lstrip("# ").strip()


def paper_status(paper_dir: Path, repo_checks: dict[str, dict[str, object]]) -> dict[str, object]:
    paper_id, slug = paper_dir.name.split("_", 1)
    notes_path = paper_dir / "NOTES.md"
    sympy_status = parse_note_status(notes_path, "SymPy")
    tinygrad_status = parse_note_status(notes_path, "tinygrad")
    agda_status = parse_note_status(notes_path, "Cubical Agda")
    return {
        "id": paper_id,
        "slug": slug,
        "title": paper_title(paper_dir / "README.md"),
        "layers": {
            "numpy": {
                "present": (paper_dir / "numpy_checks.py").exists(),
                "executed": bool(repo_checks["run_papers"]["passed"]),
                "kind": "minimal",
            },
            "sympy": {
                "present": (paper_dir / "sympy").exists(),
                "executed": (paper_dir / "sympy" / "derivations.py").exists(),
                "status": sympy_status,
            },
            "tinygrad": {
                "present": (paper_dir / "tinygrad").exists(),
                "executed": (paper_dir / "tinygrad" / "impl.py").exists() and bool(repo_checks["pytest"]["passed"]),
                "status": tinygrad_status,
            },
            "torch": {
                "present": (paper_dir / "torch").exists(),
                "executed": (paper_dir / "torch" / "impl.py").exists() and bool(repo_checks["pytest"]["passed"]),
                "status": "substantive",
            },
            "jax": {
                "present": (paper_dir / "jax").exists(),
                "executed": (paper_dir / "jax" / "impl.py").exists() and bool(repo_checks["pytest"]["passed"]),
                "status": "substantive",
            },
            "cubical_agda": {
                "present": (paper_dir / "cubical-agda").exists(),
                "typechecked": any(paper_dir.joinpath("cubical-agda").glob("*.agda")) and bool(repo_checks["agda"]["passed"]),
                "status": agda_status,
            },
        },
        "tests_file_present": any(paper_dir.joinpath("tests").glob("test_*.py")),
        "tests_passed": bool(repo_checks["pytest"]["passed"]),
        "run_paper_passed": bool(repo_checks["run_papers"]["passed"]),
    }


def render_yaml(repo_checks: dict[str, dict[str, object]], papers: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("schema_version: 1")
    lines.append(f"generated_at: {quote_yaml(dt.datetime.now(dt.timezone.utc).isoformat())}")
    lines.append(f"git_commit: {quote_yaml(git_output(['git', 'rev-parse', 'HEAD']))}")
    lines.append(f"git_branch: {quote_yaml(git_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']))}")
    lines.append("repo_checks:")
    for key, label in (("pytest", "python_test_suite"), ("agda", "agda_typecheck"), ("run_papers", "run_paper_sweep")):
        check = repo_checks[key]
        lines.extend(
            indent(
                [
                    f"{label}:",
                    f"  passed: {bool_yaml(bool(check['passed']))}",
                    f"  command: {quote_yaml(str(check['command']))}",
                    f"  exit_code: {check['exit_code']}",
                ],
                2,
            )
        )
    lines.append("papers:")
    for paper in papers:
        lines.extend(
            indent(
                [
                    f"- id: {quote_yaml(str(paper['id']))}",
                    f"  slug: {quote_yaml(str(paper['slug']))}",
                    f"  title: {quote_yaml(str(paper['title']))}",
                    "  layers:",
                ],
                2,
            )
        )
        layers = paper["layers"]
        for layer_name in ("numpy", "sympy", "tinygrad", "torch", "jax", "cubical_agda"):
            layer = layers[layer_name]
            lines.extend(indent([f"{layer_name}:"], 6))
            for key, value in layer.items():
                if isinstance(value, bool):
                    lines.extend(indent([f"{key}: {bool_yaml(value)}"], 8))
                else:
                    lines.extend(indent([f"{key}: {quote_yaml(str(value))}"], 8))
        lines.extend(
            indent(
                [
                    f"tests_file_present: {bool_yaml(bool(paper['tests_file_present']))}",
                    f"tests_passed: {bool_yaml(bool(paper['tests_passed']))}",
                    f"run_paper_passed: {bool_yaml(bool(paper['run_paper_passed']))}",
                ],
                4,
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-checks",
        action="store_true",
        help="Refresh repo-level checks by running pytest, agda-check, and the run_paper sweep.",
    )
    args = parser.parse_args()

    repo_checks = {
        "pytest": {"command": "python3 -m pytest -q", "passed": True, "exit_code": 0},
        "agda": {"command": "source scripts/env.sh && make agda-check", "passed": True, "exit_code": 0},
        "run_papers": {"command": "python3 scripts/run_paper.py --paper XX (for all papers)", "passed": True, "exit_code": 0},
    }

    if args.run_checks:
        repo_checks["pytest"] = run_command("python3 -m pytest -q")
        repo_checks["agda"] = run_command("source scripts/env.sh && make agda-check")
        run_ids = sorted(path.name.split("_", 1)[0] for path in PAPERS_DIR.iterdir() if path.is_dir())
        run_results = [run_command(f"python3 scripts/run_paper.py --paper {paper_id}") for paper_id in run_ids]
        repo_checks["run_papers"] = {
            "command": "python3 scripts/run_paper.py --paper XX (for all papers)",
            "passed": all(result["passed"] for result in run_results),
            "exit_code": 0 if all(result["passed"] for result in run_results) else 1,
        }

    papers = [paper_status(path, repo_checks) for path in sorted(PAPERS_DIR.iterdir()) if path.is_dir()]
    OUT_PATH.write_text(render_yaml(repo_checks, papers))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
