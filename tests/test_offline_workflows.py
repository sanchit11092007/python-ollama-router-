from pathlib import Path

import db
import langgraph_agent
from artifacts import OUTPUT_DIR, create_artifact
from fastapi.testclient import TestClient
from main import app


def test_supported_artifacts_are_created_and_readable():
    for extension in ("txt", "md", "json", "csv", "xlsx", "docx", "pdf"):
        artifact = create_artifact("Test Artifact", "name,value\nagent,1", extension, f"pytest_artifact.{extension}")
        assert Path(artifact.path).is_file()
        assert artifact.size_bytes > 0


def test_agent_completes_requested_file_workflow(monkeypatch):
    monkeypatch.setattr(langgraph_agent, "_llm_content", lambda *_: "def last_non_zero(n):\n    return n")
    result = langgraph_agent.run_agent("Generate Python code for the last non-zero factorial digit and provide it as a PDF.")
    assert result["artifact"] is not None
    assert result["artifact"]["filename"].endswith(".pdf")
    assert any(event["stage"] == "creating_file" for event in result["events"])
    assert (OUTPUT_DIR / result["artifact"]["filename"]).is_file()


def test_sqlite_session_persistence():
    assert db.init_db()
    session = "pytest-offline-session"
    assert db.create_session(session)
    assert db.save_message(session, "user", "remember this")
    messages = db.get_session_messages(session)
    assert messages[-1]["content"] == "remember this"
    assert db.delete_session(session)


def test_agent_api_returns_download_url(monkeypatch):
    monkeypatch.setattr(langgraph_agent, "_llm_content", lambda *_: "A concise report.")
    with TestClient(app) as client:
        response = client.post("/ask/agent", json={"query": "Create a PDF about offline AI."})
        artifact = response.json()["artifact"]
        download = client.get(artifact["download_url"])
    assert response.status_code == 200
    assert artifact["download_url"].startswith("/files/")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/pdf"


def test_capabilities_route_is_available():
    with TestClient(app) as client:
        capabilities = client.get("/capabilities")
    assert capabilities.json()["offline"] is True


def test_derive_clean_title_removes_prompt_instructions():
    from artifacts import derive_clean_title

    # 1. Title extracted from markdown heading inside content
    q1 = "Give me the code to print the fibonaaci series in python and also generatet he pdf"
    c1 = "# Python code to print the Fibonacci series\n\ndef fib(n):\n    pass"
    title1, content1, slug1 = derive_clean_title(q1, c1)
    assert title1 == "Python code to print the Fibonacci series"
    assert not content1.startswith("# Python code")
    assert not title1.lower().startswith("give me")

    # 2. Title cleaned directly from prompt without content heading
    q2 = "Give me the code to print numebrs from 1 to 10 in python also generate the pdf"
    c2 = "for i in range(1, 11):\n    print(i)"
    title2, content2, slug2 = derive_clean_title(q2, c2)
    assert not title2.lower().startswith("give me")
    assert "generate" not in title2.lower()
    assert "pdf" not in title2.lower()
    assert "1 to 10" in title2


def test_is_file_creation_request_detection():
    from ask import is_file_creation_request

    assert is_file_creation_request("Give me the code to print numebrs from 1 to 10 in python also generate the pdf") is True
    assert is_file_creation_request("Give me the code to print the fibonaaci series in python and also generatet he pdf") is True
    assert is_file_creation_request("Create a word document with project plan") is True
    assert is_file_creation_request("Export sales data to excel") is True
    assert is_file_creation_request("How do I reverse a string in python?") is False


def test_router_find_embedded_tool_calls():
    from router import _find_embedded_tool_calls

    model_output = (
        "Sure, here is the code:\n```python\nprint(1)\n```\n"
        "To generate a PDF from this text, use:\n"
        "```json\n"
        '{"name": "generate_pdf_from_text", "arguments": {"title": "Numbers", "content": "1", "filename": "numbers.pdf"}}\n'
        "```"
    )
    tools = _find_embedded_tool_calls(model_output)
    assert len(tools) == 1
    assert tools[0][0] == "generate_pdf_from_text"
    assert tools[0][1]["title"] == "Numbers"
