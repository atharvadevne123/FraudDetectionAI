"""Tests for utils.logging."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class TestConfigureLogging:
    def test_configure_stdout_only(self, tmp_path):
        from utils.logging import configure_logging
        configure_logging(level="DEBUG")

    def test_configure_with_log_file(self, tmp_path):
        from utils.logging import configure_logging
        log_file = tmp_path / "test.log"
        configure_logging(level="INFO", log_file=log_file)

    def test_configure_with_path_object(self, tmp_path):
        from utils.logging import configure_logging
        configure_logging(level="WARNING", log_file=tmp_path / "warn.log")

    def test_configure_without_file_does_not_create_file(self, tmp_path):
        from utils.logging import configure_logging
        configure_logging(level="INFO")
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 0


class TestGetLogger:
    def test_returns_logger(self):
        from utils.logging import get_logger
        log = get_logger("test_module")
        assert log is not None

    def test_logger_is_callable(self):
        from utils.logging import get_logger
        log = get_logger("test_module")
        assert callable(log.info)
        assert callable(log.warning)
        assert callable(log.error)

    def test_different_names_return_bound_loggers(self):
        from utils.logging import get_logger
        log_a = get_logger("module_a")
        log_b = get_logger("module_b")
        assert log_a is not log_b
