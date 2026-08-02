from __future__ import annotations

from pathlib import Path

from mana_agent.documents.detector import detect_document_type
from mana_agent.documents.service import DocumentService
from mana_agent.documents.types import DocumentFileType
from mana_agent.multi_agent.core.ids import new_task_id
from mana_agent.multi_agent.core.types import QueueJob, QueueJobStatus, QueueJobType
from mana_agent.multi_agent.tools.tool_manager import ToolsManager
from mana_agent.tools.contracts import coding_tool_contracts_payload
from mana_agent.transactional_actions.runtime import approve_action


def test_document_detection_supports_requested_types() -> None:
    assert detect_document_type("report.docx").file_type == DocumentFileType.DOCX
    assert detect_document_type("report.pdf").file_type == DocumentFileType.PDF
    assert detect_document_type("budget.xlsx").file_type == DocumentFileType.XLSX
    assert detect_document_type("budget.xlsm").file_type == DocumentFileType.XLSM
    assert detect_document_type("table.csv").file_type == DocumentFileType.CSV
    assert not detect_document_type("image.png").supported


def test_docx_create_read_update_and_query(tmp_path: Path) -> None:
    service = DocumentService(tmp_path)

    created = service.create(
        "docs/report.docx",
        content={
            "title": "Quarterly Report",
            "paragraphs": ["Payment terms are Net 30.", "Revenue improved."],
            "tables": [[["Metric", "Value"], ["Revenue", "1200"]]],
        },
    )
    assert created["ok"] is True

    read = service.read("docs/report.docx")
    assert read["ok"] is True
    assert read["file_type"] == "docx"
    assert any("Payment terms" in chunk["content"] for chunk in read["chunks"])
    assert read["analysis"]["table_count"] == 1

    cached = service.read("docs/report.docx")
    assert cached["cache_hit"] is True

    query = service.query("Net 30", paths=["docs/report.docx"])
    assert query["results"][0]["citation"]["paragraph"] >= 1

    updated = service.update(
        "docs/report.docx",
        operation="replace_text",
        payload={"old_text": "Net 30", "new_text": "Net 45"},
    )
    assert updated["ok"] is True
    assert updated["backup_path"]
    reread = service.read("docs/report.docx")
    assert reread["cache_hit"] is False
    assert any("Net 45" in chunk["content"] for chunk in reread["chunks"])


def test_xlsx_create_read_formula_safety_and_delete(tmp_path: Path) -> None:
    service = DocumentService(tmp_path)
    created = service.create(
        "budget.xlsx",
        content={"rows": [["Month", "Amount", "Formula"], ["March", 1000, "=B2*2"]]},
    )
    assert created["ok"] is True

    read = service.read("budget.xlsx")
    assert read["ok"] is True
    assert read["analysis"]["sheet_count"] == 1
    assert read["analysis"]["formula_count"] == 1
    assert read["analysis"]["sheets"][0]["headers"] == ["Month", "Amount", "Formula"]

    blocked = service.update(
        "budget.xlsx",
        operation="update_cell",
        payload={"sheet": "Sheet1", "cell": "C2", "value": "manual"},
    )
    assert blocked["ok"] is False
    assert blocked["error"] == "formula_replacement_requires_explicit_flag"

    updated = service.update(
        "budget.xlsx",
        operation="update_cell",
        payload={"sheet": "Sheet1", "cell": "B2", "value": 1200},
    )
    assert updated["ok"] is True

    delete_blocked = service.delete("budget.xlsx")
    assert delete_blocked["error"] == "explicit_delete_required"
    pending_delete = service.delete("budget.xlsx", explicit=True)
    assert pending_delete["error_code"] == "approval_required"
    approval_id = approve_action(
        tmp_path,
        pending_delete["action_id"],
        approved_by="pytest-local-user",
    )
    deleted = service.delete(
        "budget.xlsx",
        explicit=True,
        action_approval_id=approval_id,
    )
    assert deleted["ok"] is True
    assert not (tmp_path / "budget.xlsx").exists()


def test_xlsx_create_writes_explicit_cells_and_sum_formula(tmp_path: Path) -> None:
    service = DocumentService(tmp_path)
    created = service.create(
        "cell_sum.xlsx",
        content={
            "sheets": {
                "Sheet1": {
                    "cells": [
                        {"cell": "A1", "value": 100},
                        {"cell": "A2", "value": 200},
                        {"cell": "A3", "value": 300},
                        {"cell": "A4", "formula": "=SUM(A1:A3)"},
                    ]
                }
            }
        },
    )

    assert created["ok"] is True
    assert created["verification"]["non_empty_cells"] == 4
    assert created["verification"]["formula_count"] == 1

    read = service.read("cell_sum.xlsx")
    assert read["ok"] is True
    assert read["analysis"]["formula_count"] == 1
    assert any("=SUM(A1:A3)" in chunk["content"] for chunk in read["chunks"])


def test_xlsx_create_rejects_blank_or_malformed_content(tmp_path: Path) -> None:
    service = DocumentService(tmp_path)

    blank = service.create(
        "blank.xlsx",
        content={"description": "Excel workbook with numbers 100, 200, 300 and their sum"},
    )
    assert blank["ok"] is False
    assert blank["error"] == "invalid_excel_schema"
    assert not (tmp_path / "blank.xlsx").exists()

    list_sheets = service.create(
        "list_sheets.xlsx",
        content={"sheets": [{"name": "Sheet1", "cells": [{"cell": "A1", "value": 100}]}]},
    )
    assert list_sheets["ok"] is False
    assert list_sheets["error"] == "invalid_excel_schema"
    assert not (tmp_path / "list_sheets.xlsx").exists()


def test_pdf_create_read_and_corrupt_failure(tmp_path: Path) -> None:
    service = DocumentService(tmp_path)
    created = service.create(
        "summary.pdf",
        content={
            "title": "Payment Overview",
            "subtitle": "Verified terms",
            "sections": [
                {
                    "heading": "Terms",
                    "paragraphs": ["Payment terms Net 30"],
                    "bullets": [f"Evidence item {index}: retained in full." for index in range(180)],
                }
            ],
        },
    )
    assert created["ok"] is True
    assert created["verification"]["layout"] == "styled_report"
    assert created["verification"]["page_count"] > 1

    read = service.read("summary.pdf")
    assert read["ok"] is True
    assert read["analysis"]["page_count"] > 1
    assert any("Payment terms" in chunk["content"] for chunk in read["chunks"])
    assert any("Evidence item 179" in chunk["content"] for chunk in read["chunks"])

    (tmp_path / "bad.pdf").write_text("not a pdf", encoding="utf-8")
    corrupt = service.read("bad.pdf")
    assert corrupt["ok"] is False
    assert corrupt["detection"]["file_type"] == "pdf"


def test_pdf_create_rejects_empty_content(tmp_path: Path) -> None:
    result = DocumentService(tmp_path).create(
        "empty.pdf",
        content={"title": "Title only"},
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_pdf_content"
    assert not (tmp_path / "empty.pdf").exists()


def test_csv_fixture_query_and_discovery(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/documents/sample.csv")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "sample.csv").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    service = DocumentService(tmp_path)

    discovered = service.discover()
    assert discovered["files"][0]["path"] == "docs/sample.csv"
    result = service.query("Net 30", file_types=["csv"])
    assert result["results"][0]["row"] == 2


def test_document_tool_contracts_and_queue_execution(tmp_path: Path) -> None:
    names = {item["name"] for item in coding_tool_contracts_payload()["tools"]}
    assert {
        "document_detect",
        "document_read",
        "document_analyze",
        "document_query",
        "document_create",
        "document_update",
        "document_delete",
    }.issubset(names)

    manager = ToolsManager(tmp_path)
    task_id = new_task_id()
    job = QueueJob(
        job_id="job_document_create",
        task_id=task_id,
        requested_by_agent_id="agent_tool",
        job_type=QueueJobType.DOCUMENT,
        payload={
            "tool": "document_create",
            "args": {"path": "docs/report.docx", "content": {"paragraphs": ["Hello document"]}},
        },
        status=QueueJobStatus.QUEUED,
    )
    result = manager.execute_job(job)
    assert result.ok is True
    assert (tmp_path / "docs" / "report.docx").exists()
