from __future__ import annotations

import logging

from mana_agent.commands import cli_internal


def test_quiet_filter_drops_normal_logs_keeps_warnings(monkeypatch) -> None:
    monkeypatch.setattr(cli_internal, "CLI_VERBOSE_MODE", False)
    f = cli_internal._QuietChatConsoleFilter()

    def _rec(name: str, level: int) -> logging.LogRecord:
        return logging.LogRecord(name, level, __file__, 1, "msg", None, None)

    # Normal chat mode drops all sub-warning records from the visible console.
    assert f.filter(_rec("mana_agent.parsers.python_parser", logging.DEBUG)) is False
    assert f.filter(_rec("mana_agent.analysis.chunker", logging.INFO)) is False
    assert f.filter(_rec("mana_agent.services.index_service", logging.DEBUG)) is False
    assert f.filter(_rec("mana_agent.vector_store.faiss_store", logging.INFO)) is False
    assert f.filter(_rec("mana_agent.commands.cli_internal", logging.INFO)) is False
    assert f.filter(_rec("mana_agent.multi_agent.runtime.tool_worker_process", logging.INFO)) is False

    # Warnings/errors always pass, even from noisy loggers.
    assert f.filter(_rec("mana_agent.services.index_service", logging.WARNING)) is True
    assert f.filter(_rec("mana_agent.commands.cli_internal", logging.ERROR)) is True


def test_quiet_filter_verbose_keeps_non_noisy_info(monkeypatch) -> None:
    monkeypatch.setattr(cli_internal, "CLI_VERBOSE_MODE", True)
    f = cli_internal._QuietChatConsoleFilter()

    def _rec(name: str, level: int) -> logging.LogRecord:
        return logging.LogRecord(name, level, __file__, 1, "msg", None, None)

    assert f.filter(_rec("mana_agent.commands.cli_internal", logging.INFO)) is True
    assert f.filter(_rec("mana_agent.multi_agent.runtime.tool_worker_process", logging.INFO)) is True
    assert f.filter(_rec("mana_agent.parsers.python_parser", logging.INFO)) is False


def test_install_quiet_console_targets_only_stream_handlers(tmp_path) -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(tmp_path / "x.log")
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    try:
        cli_internal._install_quiet_chat_console_logging()
        assert any(isinstance(f, cli_internal._QuietChatConsoleFilter) for f in stream_handler.filters)
        # FileHandler must NOT be filtered (file log keeps everything).
        assert not any(isinstance(f, cli_internal._QuietChatConsoleFilter) for f in file_handler.filters)

        # Idempotent: calling again does not add a second filter.
        cli_internal._install_quiet_chat_console_logging()
        assert sum(isinstance(f, cli_internal._QuietChatConsoleFilter) for f in stream_handler.filters) == 1
    finally:
        root.removeHandler(stream_handler)
        root.removeHandler(file_handler)
        file_handler.close()
        root.handlers = original_handlers
