import os
from pathlib import Path
from unittest.mock import MagicMock
import openpyxl
from docx import Document

import router
import langgraph_agent
from artifacts import OUTPUT_DIR, create_artifact, _rows, _parse_cell_value, _generate_xlsx
import tools


def test_router_history_retains_at_least_10_turns():
    """Verify that router.add_to_history does not prematurely purge messages after 3 turns."""
    router.clear_history()
    # Add 8 full turns (16 messages)
    for i in range(1, 9):
        router.add_to_history("user", f"Question {i}")
        router.add_to_history("assistant", f"Answer {i}")

    history = router.get_history()
    assert len(history) == 16
    assert history[0]["content"] == "Question 1"
    assert history[-1]["content"] == "Answer 8"

    # Add up to 12 turns (24 messages). Max messages is max(MAX_TURNS * 2, 20) = 20 (if MAX_TURNS=6)
    for i in range(9, 13):
        router.add_to_history("user", f"Question {i}")
        router.add_to_history("assistant", f"Answer {i}")

    assert len(router.get_history()) >= 20


def test_clear_history_creates_new_session():
    """Verify clear_history clears memory and updates session name."""
    router.add_to_history("user", "Hello")
    old_session = router._CURRENT_SESSION_NAME
    router.clear_history()
    assert len(router.get_history()) == 0
    assert router._CURRENT_SESSION_NAME != ""


def test_referential_topic_resolution_from_history():
    """Verify that 'put it in a word file' resolves subject from prior turn."""
    history = [
        {"role": "user", "content": "Tell me about Quantum Computing and its future"},
        {"role": "assistant", "content": "✅ **quantum_computing_overview.pdf** created successfully!\n📁 Saved to disk."},
    ]
    resolved, source = langgraph_agent._resolve_subject_from_history("Now put it in a word file", history, "docx")
    assert "Quantum Computing" in resolved


def test_referential_topic_resolution_from_user_turn():
    """Verify that 'save this as excel' resolves subject from previous user question."""
    history = [
        {"role": "user", "content": "Analyze renewable solar energy output for 2026"},
        {"role": "assistant", "content": "Here is the analysis on solar energy: panels, efficiency, grid tie-in."},
    ]
    resolved, source = langgraph_agent._resolve_subject_from_history("Save this as excel", history, "xlsx")
    assert "Solar Energy" in resolved or "Renewable Solar" in resolved


def test_excel_rows_parses_markdown_table():
    """Verify that markdown tables are parsed into real multi-column rows."""
    md_table = (
        "| Employee ID | Name | Department | Salary |\n"
        "|-------------|------|------------|--------|\n"
        "| EMP-001     | Alice | Engineering| 95000  |\n"
        "| EMP-002     | Bob   | Marketing  | 78000  |\n"
    )
    rows = _rows(md_table)
    assert len(rows) == 3  # header + 2 data rows
    assert rows[0] == ["Employee ID", "Name", "Department", "Salary"]
    assert rows[1] == ["EMP-001", "Alice", "Engineering", "95000"]
    assert rows[2] == ["EMP-002", "Bob", "Marketing", "78000"]


def test_excel_rows_strips_code_fences_and_preamble():
    """Verify fenced code blocks and conversational preamble are stripped."""
    fenced_content = (
        "Here is the spreadsheet you asked for:\n"
        "```csv\n"
        "Product,Units Sold,Revenue\n"
        "Widget A,150,4500\n"
        "Widget B,80,2400\n"
        "```\n"
        "Hope this helps!"
    )
    rows = _rows(fenced_content)
    assert len(rows) == 3
    assert rows[0] == ["Product", "Units Sold", "Revenue"]
    assert rows[1] == ["Widget A", "150", "4500"]
    assert rows[2] == ["Widget B", "80", "2400"]


def test_parse_cell_value_converts_numeric_types():
    """Verify numeric strings become ints/floats without stripping codes."""
    assert _parse_cell_value("123") == 123
    assert isinstance(_parse_cell_value("123"), int)

    assert _parse_cell_value("45.67") == 45.67
    assert isinstance(_parse_cell_value("45.67"), float)

    assert _parse_cell_value("$1,250") == 1250
    assert _parse_cell_value("true") is True
    assert _parse_cell_value("false") is False

    # Preserves leading zeros on codes and string IDs
    assert _parse_cell_value("01234") == "01234"
    assert _parse_cell_value("STU-001") == "STU-001"
    assert _parse_cell_value("John Doe") == "John Doe"


def test_excel_generation_numeric_cells_and_types():
    """Verify generated XLSX contains real numeric cells, not raw strings."""
    content = (
        "Item,Quantity,Unit Price,In Stock\n"
        "Laptop,15,1200.50,true\n"
        "Monitor,40,250,false\n"
    )
    test_path = OUTPUT_DIR / "test_numeric_types.xlsx"
    _generate_xlsx("Inventory Report", content, test_path)

    wb = openpyxl.load_workbook(test_path)
    ws = wb.active

    # Row 1 is Title bar
    assert ws.cell(1, 1).value.startswith("Inventory Report")

    # Row 2 is Header
    assert ws.cell(2, 1).value == "Item"
    assert ws.cell(2, 2).value == "Quantity"

    # Row 3 is Data
    assert ws.cell(3, 1).value == "Laptop"
    assert ws.cell(3, 2).value == 15
    assert isinstance(ws.cell(3, 2).value, int)
    assert ws.cell(3, 3).value == 1200.50
    assert isinstance(ws.cell(3, 3).value, float)
    assert ws.cell(3, 4).value is True

    # Font should be Calibri
    assert ws.cell(2, 1).font.name == "Calibri"
    assert ws.cell(3, 1).font.name == "Calibri"


def test_tools_generate_xlsx_numeric_types():
    """Verify tools.generate_xlsx also parses numbers into ints/floats."""
    headers = ["ID", "Score", "Active"]
    data = [["STU-1", "98", "true"], ["STU-2", "85.5", "false"]]
    res = tools.generate_xlsx("Student Grades", headers, data, "pytest_tools_grades.xlsx")
    assert "saved to:" in res

    wb = openpyxl.load_workbook(OUTPUT_DIR / "pytest_tools_grades.xlsx")
    ws = wb.active
    assert ws.cell(2, 2).value == 98
    assert isinstance(ws.cell(2, 2).value, int)
    assert ws.cell(3, 2).value == 85.5
    assert isinstance(ws.cell(3, 2).value, float)


def test_word_unclosed_code_block_flushes():
    """Verify that unclosed code blocks at EOF are rendered properly in Word."""
    content = (
        "## Python Implementation\n\n"
        "```python\n"
        "def greet(name):\n"
        "    return f'Hello, {name}'"
        # No closing ```
    )
    artifact = create_artifact("Code Guide", content, "docx", "pytest_unclosed_code.docx")
    doc = Document(str(artifact.path))

    # Should contain the code text in a shaded table
    all_table_text = " ".join(cell.text for t in doc.tables for row in t.rows for cell in row.cells)
    assert "def greet(name):" in all_table_text


def test_tools_write_docx_parity_with_tables_and_callouts():
    """Verify tools.write_docx parses tables and callouts without dumping raw markdown."""
    content = (
        "## Performance Metrics\n\n"
        "| Metric | Value |\n"
        "| Latency | 12ms |\n"
        "| Throughput | 5000 rps |\n\n"
        "> Note: These metrics were measured under load.\n"
    )
    res = tools.write_docx("System Performance", content, "pytest_tools_performance.docx")
    assert "saved to:" in res

    doc = Document(str(OUTPUT_DIR / "pytest_tools_performance.docx"))
    # Table should be created (masthead + data table)
    assert len(doc.tables) >= 2
    assert len(doc.inline_shapes) == 0  # No logo image
