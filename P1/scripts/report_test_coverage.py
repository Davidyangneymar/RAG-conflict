from __future__ import annotations

import argparse
import json
import sys
import trace
import unittest
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "p1"
TEST_ROOT = REPO_ROOT / "tests"


@dataclass
class FileCoverage:
    path: Path
    executable_lines: int
    covered_lines: int

    @property
    def missing_lines(self) -> int:
        return max(0, self.executable_lines - self.covered_lines)

    @property
    def percent(self) -> float:
        if self.executable_lines == 0:
            return 100.0
        return round((self.covered_lines / self.executable_lines) * 100, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run P1 unittest suite and report stdlib trace coverage.")
    parser.add_argument("--pattern", default="test_*.py", help="unittest discovery pattern")
    parser.add_argument("--write-report", type=Path, help="optional markdown report path")
    parser.add_argument("--write-json", type=Path, help="optional JSON summary path")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    ignoredirs = [
        sys.base_prefix,
        sys.exec_prefix,
        str(REPO_ROOT / ".venv312"),
    ]
    tracer = trace.Trace(count=True, trace=False, ignoredirs=ignoredirs)
    result = tracer.runfunc(run_tests, args.pattern)
    coverage_rows = collect_coverage(tracer.results().counts)
    totals = summarize(coverage_rows)

    print_summary(coverage_rows, totals)
    if args.write_report:
        write_report(args.write_report, coverage_rows, totals, result)
    if args.write_json:
        write_json(args.write_json, coverage_rows, totals, result)
    return 0 if result.wasSuccessful() else 1


def run_tests(pattern: str) -> unittest.result.TestResult:
    suite = unittest.defaultTestLoader.discover(str(TEST_ROOT), pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=1)
    return runner.run(suite)


def collect_coverage(counts: dict[tuple[str, int], int]) -> list[FileCoverage]:
    executed_by_file: dict[Path, set[int]] = {}
    for filename, line_number in counts:
        path = Path(filename).resolve()
        if is_target_source_file(path):
            executed_by_file.setdefault(path, set()).add(line_number)

    rows: list[FileCoverage] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        executable = set(trace._find_executable_linenos(str(path)))  # stdlib private helper used by trace.py itself.
        covered = executed_by_file.get(path.resolve(), set()) & executable
        rows.append(
            FileCoverage(
                path=path,
                executable_lines=len(executable),
                covered_lines=len(covered),
            )
        )
    return rows


def is_target_source_file(path: Path) -> bool:
    try:
        path.relative_to(SRC_ROOT)
    except ValueError:
        return False
    return path.suffix == ".py" and "__pycache__" not in path.parts


def summarize(rows: list[FileCoverage]) -> dict[str, float | int]:
    executable = sum(row.executable_lines for row in rows)
    covered = sum(row.covered_lines for row in rows)
    percent = round((covered / executable) * 100, 2) if executable else 100.0
    return {
        "file_count": len(rows),
        "executable_lines": executable,
        "covered_lines": covered,
        "missing_lines": max(0, executable - covered),
        "coverage_percent": percent,
    }


def print_summary(rows: list[FileCoverage], totals: dict[str, float | int]) -> None:
    print("\nP1 stdlib trace coverage")
    print(f"files={totals['file_count']} covered={totals['covered_lines']}/{totals['executable_lines']} "
          f"coverage={totals['coverage_percent']}%")
    for row in rows:
        relative = row.path.relative_to(REPO_ROOT)
        print(f"{relative}: {row.covered_lines}/{row.executable_lines} ({row.percent}%)")


def write_report(
    path: Path,
    rows: list[FileCoverage],
    totals: dict[str, float | int],
    result: unittest.result.TestResult,
) -> None:
    lines = [
        "# P1 Test Coverage Report",
        "",
        "Generated with `scripts/report_test_coverage.py` using Python stdlib `trace` over test discovery and execution.",
        "",
        "## Summary",
        "",
        f"- Tests run: `{result.testsRun}`",
        f"- Failures: `{len(result.failures)}`",
        f"- Errors: `{len(result.errors)}`",
        f"- Source files counted: `{totals['file_count']}`",
        f"- Covered executable lines: `{totals['covered_lines']} / {totals['executable_lines']}`",
        f"- Line coverage: `{totals['coverage_percent']}%`",
        "",
        "## Per-File Coverage",
        "",
        "| File | Covered | Executable | Coverage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        relative = row.path.relative_to(REPO_ROOT)
        lines.append(f"| `{relative}` | {row.covered_lines} | {row.executable_lines} | {row.percent}% |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a lightweight stdlib trace report, not a replacement for `coverage.py` branch coverage.",
            "- The result is enough for a course-project audit claim that testing is now measured, not only counted.",
            "- Remaining high-value testing work is branch coverage and a broader historical failure-bucket regression set.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(
    path: Path,
    rows: list[FileCoverage],
    totals: dict[str, float | int],
    result: unittest.result.TestResult,
) -> None:
    payload = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "totals": totals,
        "files": [
            {
                "path": str(row.path.relative_to(REPO_ROOT)),
                "covered_lines": row.covered_lines,
                "executable_lines": row.executable_lines,
                "missing_lines": row.missing_lines,
                "coverage_percent": row.percent,
            }
            for row in rows
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
