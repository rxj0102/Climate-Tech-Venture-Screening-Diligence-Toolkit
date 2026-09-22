"""Smoke tests for every module's standalone ``main.py`` entry point, plus
the top-level orchestrator. These exist so the "run end-to-end and print
your own results" scripts are covered by the test suite too, not just the
underlying library functions."""

from __future__ import annotations

import pytest

import diligence.main
import fund_returns.main
import main as top_level_main
import screening.main
import unit_economics.main


def test_screening_main_runs_and_prints_shortlist(capsys: pytest.CaptureFixture[str]) -> None:
    screening.main.run(n_deals=20, seed=1, shortlist_size=5)
    captured = capsys.readouterr()
    assert "MODULE 1" in captured.out
    assert "Shortlist" in captured.out


def test_unit_economics_main_runs_and_prints_percentiles(
    capsys: pytest.CaptureFixture[str],
) -> None:
    unit_economics.main.run()
    captured = capsys.readouterr()
    assert "MODULE 2" in captured.out
    assert "Percentiles" in captured.out
    assert "Tornado" in captured.out


def test_fund_returns_main_runs_and_prints_moic(capsys: pytest.CaptureFixture[str]) -> None:
    fund_returns.main.run()
    captured = capsys.readouterr()
    assert "MODULE 3" in captured.out
    assert "MOIC" in captured.out
    assert "Power-law" in captured.out


def test_diligence_main_runs_and_prints_override(capsys: pytest.CaptureFixture[str]) -> None:
    diligence.main.run()
    captured = capsys.readouterr()
    assert "MODULE 4" in captured.out
    assert "OVERRIDE" in captured.out


def test_top_level_main_runs_full_pipeline(capsys: pytest.CaptureFixture[str]) -> None:
    top_level_main.run()
    captured = capsys.readouterr()
    assert "FULL END-TO-END RUN" in captured.out
    assert "RECOMMENDATION SUMMARY ACROSS SHORTLIST" in captured.out
    assert "MODULE 1" in captured.out
    assert "MODULE 2" in captured.out
    assert "MODULE 3" in captured.out
    assert "MODULE 4" in captured.out
