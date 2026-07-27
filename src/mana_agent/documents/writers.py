from __future__ import annotations

import csv
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .detector import detect_document_type
from .types import DocumentFileType


def _require(module_name: str, package: str) -> Any:
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as exc:
        raise RuntimeError(f"{package} is required for this document operation") from exc


def _backup(path: Path) -> str:
    backup_path = path.with_suffix(path.suffix + ".bak")
    counter = 1

    while backup_path.exists():
        backup_path = path.with_suffix(path.suffix + f".bak{counter}")
        counter += 1

    shutil.copy2(path, backup_path)
    return str(backup_path)


def _validate_workbook_create_content(path: Path, content: Any) -> dict[str, Any] | None:
    if not isinstance(content, dict):
        return None

    if "sheet1" in content or "Sheet1" in content:
        return {
            "ok": False,
            "error": "invalid_excel_schema",
            "message": (
                "Do not pass sheet data directly as content['sheet1'] or content['Sheet1']. "
                "Use content['sheets'] = {'Sheet1': {'cells': [...]}}."
            ),
            "path": str(path),
            "expected_schema": {
                "sheets": {
                    "Sheet1": {
                        "cells": [
                            {"cell": "A200", "value": 200},
                            {"cell": "A300", "value": 300},
                            {"cell": "A400", "value": 400},
                            {"cell": "B200", "formula": "=SUM(A200,A300,A400)"},
                        ]
                    }
                }
            },
        }

    sheets = content.get("sheets")
    if sheets is not None and not isinstance(sheets, dict):
        return {
            "ok": False,
            "error": "invalid_excel_schema",
            "message": "content['sheets'] must be a dict keyed by sheet name, not a list.",
            "path": str(path),
        }

    workbook_payload = _normalize_workbook_payload(content)
    if not _workbook_payload_has_writable_content(workbook_payload):
        return {
            "ok": False,
            "error": "invalid_excel_schema",
            "message": (
                "Excel content must include at least one writable row, table, cell, or formula. "
                "Use content['sheets'] = {'Sheet1': {'cells': [...]}} for explicit cells."
            ),
            "path": str(path),
            "expected_schema": {
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
        }

    return None


def _atomic_save(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=str(path.parent),
        suffix=path.suffix,
    ) as tmp:
        temp_path = Path(tmp.name)

    try:
        writer(temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def create_document(
    path: Path,
    *,
    content: Any,
    file_type: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if path.exists() and not overwrite:
        return {
            "ok": False,
            "error": "target_exists",
            "path": str(path),
        }

    kind = _resolve_document_type(path, file_type)

    if kind == DocumentFileType.DOCX:
        return _create_docx(path, content=content, kind=kind)

    if kind in {DocumentFileType.XLSX, DocumentFileType.XLSM}:
        validation_error = _validate_workbook_create_content(path, content)
        if validation_error:
            return validation_error

        return _create_workbook(path, content=content, kind=kind)

    if kind == DocumentFileType.CSV:
        return _create_csv(path, content=content, kind=kind)

    if kind == DocumentFileType.PDF:
        return _create_pdf(path, content=content, kind=kind)

    return {
        "ok": False,
        "error": "unsupported_file_type",
        "path": str(path),
        "file_type": getattr(kind, "value", str(kind)),
    }


def update_document(
    path: Path,
    *,
    operation: str,
    payload: dict[str, Any],
    backup: bool = True,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "ok": False,
            "error": "file_not_found",
            "path": str(path),
        }

    kind = detect_document_type(path).file_type
    backup_path = _backup(path) if backup else ""

    if kind == DocumentFileType.DOCX:
        return _update_docx(
            path,
            operation=operation,
            payload=payload,
            backup_path=backup_path,
        )

    if kind in {DocumentFileType.XLSX, DocumentFileType.XLSM}:
        return _update_workbook(
            path,
            operation=operation,
            payload=payload,
            backup_path=backup_path,
            keep_vba=kind == DocumentFileType.XLSM,
        )

    if kind == DocumentFileType.PDF and operation == "metadata":
        return _update_pdf_metadata(
            path,
            payload=payload,
            backup_path=backup_path,
        )

    return {
        "ok": False,
        "error": "unsupported_update_operation",
        "path": str(path),
        "operation": operation,
        "backup_path": backup_path,
    }


def delete_document(
    path: Path,
    *,
    explicit: bool = False,
    backup: bool = True,
) -> dict[str, Any]:
    if not explicit:
        return {
            "ok": False,
            "error": "explicit_delete_required",
            "path": str(path),
        }

    if not path.exists() or not path.is_file():
        return {
            "ok": False,
            "error": "file_not_found",
            "path": str(path),
        }

    backup_path = _backup(path) if backup else ""
    path.unlink()

    return {
        "ok": True,
        "path": str(path),
        "deleted": True,
        "backup_path": backup_path,
        "files_changed": [str(path)],
    }


def _resolve_document_type(
    path: Path,
    file_type: str | None,
) -> DocumentFileType:
    if file_type:
        return DocumentFileType(str(file_type).lower())

    detected = detect_document_type(path)
    if detected.file_type:
        return detected.file_type

    suffix = path.suffix.lower()
    if suffix == ".docx":
        return DocumentFileType.DOCX
    if suffix == ".pdf":
        return DocumentFileType.PDF
    if suffix == ".xlsx":
        return DocumentFileType.XLSX
    if suffix == ".xlsm":
        return DocumentFileType.XLSM
    if suffix == ".csv":
        return DocumentFileType.CSV

    raise ValueError(f"unsupported document type for path: {path}")


def _create_docx(
    path: Path,
    *,
    content: Any,
    kind: DocumentFileType,
) -> dict[str, Any]:
    docx = _require("docx", "python-docx")
    doc = docx.Document()

    payload = (
        content
        if isinstance(content, dict)
        else {"paragraphs": str(content).splitlines()}
    )

    title = str(payload.get("title") or "").strip()
    if title:
        doc.add_heading(title, level=1)

    for paragraph in payload.get("paragraphs") or []:
        text = str(paragraph).strip()
        if text:
            doc.add_paragraph(text)

    for table_payload in payload.get("tables") or []:
        rows = list(table_payload or [])
        if not rows:
            continue

        table = doc.add_table(
            rows=len(rows),
            cols=max(len(row) for row in rows),
        )

        for row_index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                table.cell(row_index, col_index).text = str(value)

    _atomic_save(path, lambda target: doc.save(str(target)))

    return {
        "ok": True,
        "path": str(path),
        "file_type": kind.value,
        "created": True,
        "files_changed": [str(path)],
    }


def _create_workbook(
    path: Path,
    *,
    content: Any,
    kind: DocumentFileType,
) -> dict[str, Any]:
    openpyxl = _require("openpyxl", "openpyxl")
    workbook = openpyxl.Workbook()

    workbook_payload = _normalize_workbook_payload(content)

    sheets = workbook_payload.get("sheets") or {}

    if sheets:
        default_sheet = workbook.active
        workbook.remove(default_sheet)

        for sheet_name, sheet_spec in sheets.items():
            worksheet = workbook.create_sheet(title=str(sheet_name))
            _write_sheet(worksheet, sheet_spec)
    else:
        worksheet = workbook.active
        worksheet.title = "Sheet1"
        _write_sheet(worksheet, workbook_payload)

    _atomic_save(path, lambda target: workbook.save(str(target)))

    verification = _verify_workbook_created(path, keep_vba=kind == DocumentFileType.XLSM)

    if not verification.get("ok"):
        return {
            "ok": False,
            "error": "workbook_verification_failed",
            "path": str(path),
            "file_type": kind.value,
            "verification": verification,
            "files_changed": [str(path)],
        }
    return {
        "ok": True,
        "path": str(path),
        "file_type": kind.value,
        "created": True,
        "files_changed": [str(path)],
        "verification": verification,
        "warning": (
            "Created an .xlsm workbook without embedded VBA macros. "
            "Existing macros can only be preserved during update operations."
            if kind == DocumentFileType.XLSM
            else ""
        ),
    }


def _create_csv(
    path: Path,
    *,
    content: Any,
    kind: DocumentFileType,
) -> dict[str, Any]:
    rows = content.get("rows", content) if isinstance(content, dict) else content

    def write_csv(target: Path) -> None:
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)

            if rows and isinstance(rows, list) and isinstance(rows[0], dict):
                headers = list(rows[0].keys())
                writer.writerow(headers)

                for row in rows:
                    writer.writerow([row.get(header) for header in headers])
            else:
                writer.writerows(rows or [])

    _atomic_save(path, write_csv)

    return {
        "ok": True,
        "path": str(path),
        "file_type": kind.value,
        "created": True,
        "files_changed": [str(path)],
    }


def _normalize_workbook_payload(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        if isinstance(content.get("sheets"), dict):
            return {
                **content,
                "sheets": {
                    str(sheet_name): _normalize_sheet_spec(sheet_spec)
                    for sheet_name, sheet_spec in content["sheets"].items()
                },
            }

        return {
            "sheets": {
                "Sheet1": _normalize_sheet_spec(content),
            }
        }

    return {
        "sheets": {
            "Sheet1": {
                "rows": content or [],
            }
        }
    }


def _normalize_sheet_spec(sheet_spec: Any) -> dict[str, Any]:
    if isinstance(sheet_spec, list):
        return _normalize_sheet_blocks(sheet_spec)

    if not isinstance(sheet_spec, dict):
        return {"rows": sheet_spec or []}

    normalized = dict(sheet_spec)

    cells = list(normalized.get("cells") or [])
    tables = list(normalized.get("tables") or [])

    if _looks_like_table_block(normalized):
        table = _normalize_table_block(normalized)
        if table:
            tables.append(table)

    if _looks_like_cell_block(normalized):
        cell = _normalize_cell_block(normalized)
        if cell:
            cells.append(cell)

    formulas = normalized.get("formulas") or []
    if isinstance(formulas, list):
        for formula_block in formulas:
            if isinstance(formula_block, dict):
                cell = _normalize_formula_block(formula_block)
                if cell:
                    cells.append(cell)

    if cells:
        normalized["cells"] = cells
    if tables:
        normalized["tables"] = tables

    return normalized


def _normalize_sheet_blocks(blocks: list[Any]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    rows: list[Any] = []
    unknown_blocks: list[Any] = []

    for block in blocks:
        if not isinstance(block, dict):
            rows.append(block)
            continue

        block_type = str(block.get("type", "")).lower().strip()

        if block_type == "table" or _looks_like_table_block(block):
            table = _normalize_table_block(block)
            if table:
                tables.append(table)
            continue

        if block_type == "formula":
            cell = _normalize_formula_block(block)
            if cell:
                cells.append(cell)
            continue

        if block_type == "cell" or _looks_like_cell_block(block):
            cell = _normalize_cell_block(block)
            if cell:
                cells.append(cell)
            continue

        if block_type == "cells" and isinstance(block.get("cells"), list):
            for item in block["cells"]:
                if isinstance(item, dict):
                    cell = _normalize_cell_block(item)
                    if cell:
                        cells.append(cell)
            continue

        if block_type == "tables" and isinstance(block.get("tables"), list):
            for item in block["tables"]:
                if isinstance(item, dict):
                    table = _normalize_table_block(item)
                    if table:
                        tables.append(table)
            continue

        unknown_blocks.append(block)

    normalized: dict[str, Any] = {}

    if rows:
        normalized["rows"] = rows
    if cells:
        normalized["cells"] = cells
    if tables:
        normalized["tables"] = tables
    if unknown_blocks:
        normalized["unknown_blocks"] = unknown_blocks

    return normalized


def _workbook_payload_has_writable_content(workbook_payload: dict[str, Any]) -> bool:
    sheets = workbook_payload.get("sheets")
    if not isinstance(sheets, dict) or not sheets:
        return False

    return any(_sheet_spec_has_writable_content(sheet_spec) for sheet_spec in sheets.values())


def _sheet_spec_has_writable_content(sheet_spec: Any) -> bool:
    spec = _normalize_sheet_spec(sheet_spec)

    rows = spec.get("rows")
    if isinstance(rows, list) and bool(rows):
        return True

    tables = spec.get("tables")
    if isinstance(tables, list):
        for table in tables:
            if not isinstance(table, dict):
                continue
            columns = table.get("columns") or table.get("headers") or []
            rows = table.get("rows") or []
            if columns or rows:
                return True

    cells = spec.get("cells")
    if isinstance(cells, list):
        for cell in cells:
            if isinstance(cell, dict) and str(cell.get("cell") or "").strip():
                return True

    formulas = spec.get("formulas")
    if isinstance(formulas, list):
        for formula in formulas:
            if isinstance(formula, dict) and _normalize_formula_block(formula):
                return True

    return False


def _write_sheet(worksheet: Any, sheet_spec: Any) -> None:
    spec = _normalize_sheet_spec(sheet_spec)

    rows = spec.get("rows")
    if rows:
        _write_rows(worksheet, rows)

    for table in spec.get("tables") or []:
        _write_table(worksheet, table)

    for cell in spec.get("cells") or []:
        _write_cell(worksheet, cell)

    for formula in spec.get("formulas") or []:
        normalized = _normalize_formula_block(formula)
        if normalized:
            _write_cell(worksheet, normalized)


def _write_rows(worksheet: Any, rows: Any) -> None:
    if not isinstance(rows, list):
        rows = [rows]

    if rows and isinstance(rows[0], dict):
        headers = list(rows[0].keys())
        worksheet.append(headers)

        for row in rows:
            worksheet.append([row.get(header) for header in headers])
        return

    for row in rows:
        if isinstance(row, (list, tuple)):
            worksheet.append(list(row))
        else:
            worksheet.append([row])


def _write_table(worksheet: Any, table: dict[str, Any]) -> None:
    openpyxl = _require("openpyxl", "openpyxl")
    coordinate_to_tuple = openpyxl.utils.cell.coordinate_to_tuple

    start_cell = str(table.get("start_cell") or "A1")
    start_row, start_col = coordinate_to_tuple(start_cell)

    columns = table.get("columns") or table.get("headers") or []
    rows = table.get("rows") or []

    current_row = start_row

    if columns:
        for col_offset, column_name in enumerate(columns):
            worksheet.cell(
                row=current_row,
                column=start_col + col_offset,
                value=column_name,
            )
        current_row += 1

    for row_offset, row_values in enumerate(rows):
        values = list(row_values) if isinstance(row_values, (list, tuple)) else [row_values]

        for col_offset, value in enumerate(values):
            worksheet.cell(
                row=current_row + row_offset,
                column=start_col + col_offset,
                value=value,
            )


def _write_cell(worksheet: Any, cell: dict[str, Any]) -> None:
    coordinate = str(cell.get("cell") or "").strip()
    if not coordinate:
        return

    if "formula" in cell:
        formula = str(cell["formula"])
        worksheet[coordinate] = formula if formula.startswith("=") else f"={formula}"
        return

    worksheet[coordinate] = cell.get("value")


def _looks_like_table_block(block: dict[str, Any]) -> bool:
    return (
        "rows" in block
        and isinstance(block.get("rows"), list)
        and (
            "columns" in block
            or "headers" in block
            or "start_cell" in block
        )
    )


def _looks_like_cell_block(block: dict[str, Any]) -> bool:
    return "cell" in block and (
        "value" in block
        or "formula" in block
    )


def _normalize_table_block(
    block: dict[str, Any],
) -> dict[str, Any] | None:
    rows = block.get("rows")
    if not isinstance(rows, list):
        return None

    table: dict[str, Any] = {
        "start_cell": str(block.get("start_cell") or "A1"),
        "rows": rows,
    }

    columns = block.get("columns", block.get("headers"))
    if isinstance(columns, list):
        table["columns"] = columns

    if "name" in block:
        table["name"] = block["name"]

    if "style" in block:
        table["style"] = block["style"]

    return table


def _normalize_formula_block(
    block: dict[str, Any],
) -> dict[str, Any] | None:
    cell = block.get("cell")
    formula = block.get("formula")

    if not cell or formula is None:
        return None

    return {
        "cell": str(cell),
        "formula": str(formula),
    }


def _normalize_cell_block(
    block: dict[str, Any],
) -> dict[str, Any] | None:
    cell = block.get("cell")
    if not cell:
        return None

    normalized: dict[str, Any] = {"cell": str(cell)}

    if "formula" in block:
        normalized["formula"] = str(block["formula"])
    elif "value" in block:
        normalized["value"] = block["value"]
    else:
        normalized["value"] = None

    return normalized


def _verify_workbook_created(path: Path, *, keep_vba: bool) -> dict[str, Any]:
    try:
        openpyxl = _require("openpyxl", "openpyxl")
        workbook = openpyxl.load_workbook(
            str(path),
            data_only=False,
            keep_vba=keep_vba,
        )

        non_empty_cells = 0
        formulas = 0

        for worksheet in workbook.worksheets:
            for row in worksheet.iter_rows():
                for cell in row:
                    if cell.value is not None:
                        non_empty_cells += 1
                        if isinstance(cell.value, str) and cell.value.startswith("="):
                            formulas += 1

        return {
            "ok": non_empty_cells > 0,
            "sheet_count": len(workbook.sheetnames),
            "sheets": list(workbook.sheetnames),
            "non_empty_cells": non_empty_cells,
            "formula_count": formulas,
            "error": "" if non_empty_cells > 0 else "workbook_has_no_cell_content",
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }


def _update_docx(
    path: Path,
    *,
    operation: str,
    payload: dict[str, Any],
    backup_path: str,
) -> dict[str, Any]:
    docx = _require("docx", "python-docx")
    doc = docx.Document(str(path))

    if operation == "append_section":
        title = str(payload.get("title") or "").strip()
        if title:
            doc.add_heading(title, level=int(payload.get("level") or 1))

        for paragraph in payload.get("paragraphs") or [payload.get("text", "")]:
            text = str(paragraph).strip()
            if text:
                doc.add_paragraph(text)

    elif operation == "replace_text":
        old = str(payload.get("old_text") or "")
        new = str(payload.get("new_text") or "")

        if not old:
            return {
                "ok": False,
                "error": "old_text_required",
                "path": str(path),
                "backup_path": backup_path,
            }

        replaced = 0
        for paragraph in doc.paragraphs:
            if old in paragraph.text:
                paragraph.text = paragraph.text.replace(old, new)
                replaced += 1

        if replaced == 0:
            return {
                "ok": False,
                "error": "text_not_found",
                "path": str(path),
                "backup_path": backup_path,
            }

    elif operation == "add_table":
        rows = list(payload.get("rows") or [])
        if not rows:
            return {
                "ok": False,
                "error": "rows_required",
                "path": str(path),
                "backup_path": backup_path,
            }

        table = doc.add_table(
            rows=len(rows),
            cols=max(len(row) for row in rows),
        )

        for row_index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                table.cell(row_index, col_index).text = str(value)

    elif operation == "metadata":
        props = doc.core_properties
        for key, value in payload.items():
            if hasattr(props, key):
                setattr(props, key, str(value))

    else:
        return {
            "ok": False,
            "error": "unsupported_docx_operation",
            "path": str(path),
            "backup_path": backup_path,
        }

    _atomic_save(path, lambda target: doc.save(str(target)))

    return {
        "ok": True,
        "path": str(path),
        "operation": operation,
        "backup_path": backup_path,
        "files_changed": [str(path)],
    }


def _update_workbook(
    path: Path,
    *,
    operation: str,
    payload: dict[str, Any],
    backup_path: str,
    keep_vba: bool,
) -> dict[str, Any]:
    openpyxl = _require("openpyxl", "openpyxl")
    workbook = openpyxl.load_workbook(
        str(path),
        data_only=False,
        keep_vba=keep_vba,
    )

    if operation == "update_cell":
        sheet = workbook[str(payload.get("sheet") or workbook.sheetnames[0])]
        coordinate = str(payload.get("cell") or "")

        if not coordinate:
            return {
                "ok": False,
                "error": "cell_required",
                "path": str(path),
                "backup_path": backup_path,
            }

        current = sheet[coordinate].value
        if (
            isinstance(current, str)
            and current.startswith("=")
            and not bool(payload.get("replace_formula", False))
        ):
            return {
                "ok": False,
                "error": "formula_replacement_requires_explicit_flag",
                "path": str(path),
                "backup_path": backup_path,
            }

        if "formula" in payload:
            formula = str(payload["formula"])
            sheet[coordinate] = formula if formula.startswith("=") else f"={formula}"
        else:
            sheet[coordinate] = payload.get("value")

    elif operation == "append_rows":
        sheet = workbook[str(payload.get("sheet") or workbook.sheetnames[0])]
        for row in payload.get("rows") or []:
            sheet.append(row)

    elif operation == "create_sheet":
        name = str(payload.get("sheet") or "").strip()
        if not name:
            return {
                "ok": False,
                "error": "sheet_required",
                "path": str(path),
                "backup_path": backup_path,
            }

        if name in workbook.sheetnames:
            return {
                "ok": False,
                "error": "sheet_exists",
                "path": str(path),
                "sheet": name,
                "backup_path": backup_path,
            }

        workbook.create_sheet(title=name)

    elif operation == "rename_sheet":
        old = str(payload.get("sheet") or "")
        new = str(payload.get("new_name") or "")

        if old not in workbook.sheetnames:
            return {
                "ok": False,
                "error": "sheet_not_found",
                "path": str(path),
                "sheet": old,
                "backup_path": backup_path,
            }

        workbook[old].title = new

    elif operation == "delete_rows":
        sheet = workbook[str(payload.get("sheet") or workbook.sheetnames[0])]
        sheet.delete_rows(
            int(payload.get("idx") or 1),
            int(payload.get("amount") or 1),
        )

    elif operation in {"write_sheets", "replace_sheets"}:
        content = payload.get("content", payload)
        workbook_payload = _normalize_workbook_payload(content)
        sheets = workbook_payload.get("sheets") or {}

        if operation == "replace_sheets":
            for sheet_name in list(workbook.sheetnames):
                del workbook[sheet_name]

        for sheet_name, sheet_spec in sheets.items():
            if sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
            else:
                sheet = workbook.create_sheet(title=str(sheet_name))

            _write_sheet(sheet, sheet_spec)

    else:
        return {
            "ok": False,
            "error": "unsupported_workbook_operation",
            "path": str(path),
            "operation": operation,
            "backup_path": backup_path,
        }

    _atomic_save(path, lambda target: workbook.save(str(target)))

    verification = _verify_workbook_created(path, keep_vba=keep_vba)

    return {
        "ok": True,
        "path": str(path),
        "operation": operation,
        "backup_path": backup_path,
        "files_changed": [str(path)],
        "verification": verification,
        "warning": (
            "Macros preserved with keep_vba=True; verify workbook macros after editing."
            if keep_vba
            else ""
        ),
    }


def _update_pdf_metadata(
    path: Path,
    *,
    payload: dict[str, Any],
    backup_path: str,
) -> dict[str, Any]:
    pypdf = _require("pypdf", "pypdf")
    reader = pypdf.PdfReader(str(path))
    writer = pypdf.PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    writer.add_metadata({str(key): str(value) for key, value in payload.items()})

    def write_pdf(target: Path) -> None:
        with target.open("wb") as handle:
            writer.write(handle)

    _atomic_save(path, write_pdf)

    return {
        "ok": True,
        "path": str(path),
        "operation": "metadata",
        "backup_path": backup_path,
        "files_changed": [str(path)],
    }


def _create_pdf(
    path: Path,
    *,
    content: Any,
    kind: DocumentFileType,
) -> dict[str, Any]:
    from xml.sax.saxutils import escape

    _require("reportlab", "reportlab")
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    payload = content if isinstance(content, dict) else {"text": str(content or "")}
    title = str(payload.get("title") or path.stem.replace("_", " ").title()).strip()
    subtitle = str(payload.get("subtitle") or "").strip()
    raw_paragraphs = payload.get("paragraphs")
    paragraphs = raw_paragraphs if isinstance(raw_paragraphs, list) else []
    raw_sections = payload.get("sections")
    sections = raw_sections if isinstance(raw_sections, list) else []
    raw_tables = payload.get("tables")
    tables = raw_tables if isinstance(raw_tables, list) else []
    has_body_content = bool(str(payload.get("text") or "").strip())
    has_body_content = has_body_content or any(
        str(item).strip() for item in paragraphs
    )
    has_body_content = has_body_content or any(
        isinstance(section, dict)
        and any(
            (
                str(section.get("heading") or "").strip(),
                *(str(item).strip() for item in (section.get("paragraphs") or [])),
                *(str(item).strip() for item in (section.get("bullets") or [])),
            )
        )
        for section in sections
    )
    has_body_content = has_body_content or bool(tables)
    if not has_body_content:
        return {
            "ok": False,
            "error": "invalid_pdf_content",
            "message": "PDF content must include text, paragraphs, sections, or tables.",
            "path": str(path),
        }
    navy = colors.HexColor("#17324D")
    muted = colors.HexColor("#5E6B75")
    light_blue = colors.HexColor("#EAF1F6")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="PdfTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=29,
        textColor=navy,
        alignment=TA_CENTER,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="PdfSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        textColor=muted,
        alignment=TA_CENTER,
        spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        name="PdfHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=navy,
        spaceBefore=12,
        spaceAfter=7,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="PdfBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.2,
        leading=15.2,
        textColor=colors.HexColor("#20272D"),
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="PdfBullet",
        parent=styles["PdfBody"],
        leftIndent=14,
        firstLineIndent=-9,
        bulletIndent=2,
        spaceAfter=5,
    ))

    def paragraph(value: Any, style_name: str = "PdfBody") -> Any:
        safe = escape(str(value or "")).replace("\n", "<br/>")
        return Paragraph(safe, styles[style_name])

    story: list[Any] = [Spacer(1, 0.12 * inch), paragraph(title, "PdfTitle")]
    if subtitle:
        story.append(paragraph(subtitle, "PdfSubtitle"))

    if not paragraphs and payload.get("text"):
        paragraphs = [item.strip() for item in str(payload["text"]).split("\n\n") if item.strip()]
    story.extend(paragraph(item) for item in paragraphs if str(item).strip())

    for section in sections:
        if not isinstance(section, dict):
            continue
        heading = str(section.get("heading") or "").strip()
        if heading:
            story.append(paragraph(heading, "PdfHeading"))
        section_paragraphs = section.get("paragraphs")
        if isinstance(section_paragraphs, list):
            story.extend(paragraph(item) for item in section_paragraphs if str(item).strip())
        bullets = section.get("bullets")
        if isinstance(bullets, list):
            story.extend(
                Paragraph(escape(str(item)), styles["PdfBullet"], bulletText="-")
                for item in bullets
                if str(item).strip()
            )

    for rows in tables:
        if not isinstance(rows, list) or not rows:
            continue
        table_data = [
            [paragraph(cell) for cell in row]
            for row in rows
            if isinstance(row, list)
        ]
        if not table_data:
            continue
        table = Table(table_data, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), light_blue),
            ("TEXTCOLOR", (0, 0), (-1, 0), navy),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B8CCDA")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.extend([Spacer(1, 6), table, Spacer(1, 8)])

    def draw_footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7E0E6"))
        canvas.line(0.72 * inch, 0.52 * inch, 7.78 * inch, 0.52 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(muted)
        canvas.drawString(0.72 * inch, 0.34 * inch, title[:72])
        canvas.drawRightString(7.78 * inch, 0.34 * inch, f"Page {doc.page}")
        canvas.restoreState()

    def write_pdf(target: Path) -> None:
        document = SimpleDocTemplate(
            str(target),
            pagesize=letter,
            rightMargin=0.72 * inch,
            leftMargin=0.72 * inch,
            topMargin=0.68 * inch,
            bottomMargin=0.68 * inch,
            title=title,
        )
        document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)

    _atomic_save(path, write_pdf)
    pypdf = _require("pypdf", "pypdf")
    page_count = len(pypdf.PdfReader(str(path)).pages)
    return {
        "ok": True,
        "path": str(path),
        "file_type": kind.value,
        "created": True,
        "files_changed": [str(path)],
        "verification": {
            "page_count": page_count,
            "bytes": path.stat().st_size,
            "layout": "styled_report",
        },
    }
