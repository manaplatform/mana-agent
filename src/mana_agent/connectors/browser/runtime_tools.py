from __future__ import annotations
import json
from typing import Any
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from mana_agent.connectors.browser.session import BrowserConnectorError, default_browser_manager

class _Session(BaseModel): session_id: str
class _Open(_Session): url: str; profile_name: str | None = None
class _Inspect(_Session): tab_id: str | None = None
class _Act(_Inspect): target: str = ""; value: Any = None; observed_page_version: int | None = None; expected_origin: str | None = None; risk: str = "reversible"; confirmation_required: bool = False; approval_token: str | None = None; timeout_ms: int | None = None
class _Screenshot(_Inspect): full_page: bool = True
class _Upload(_Session): ref: str; path: str; observed_page_version: int; tab_id: str | None = None
class _Switch(_Session): tab_id: str
class _CheckLinks(_Inspect): max_links: int = Field(default=50, ge=1, le=100)

def _result(call):
    try: return json.dumps(call(), ensure_ascii=False, default=str)
    except BrowserConnectorError as exc: return json.dumps({"ok": False, "error_code": exc.code, "message": str(exc)})
    except Exception as exc: return json.dumps({"ok": False, "error_code": "browser_error", "message": str(exc)})

def build_browser_langchain_tools() -> list[Any]:
    manager = default_browser_manager()
    def open_page(**kw): return _result(lambda: manager.open(**_Open.model_validate(kw).model_dump()))
    def inspect(**kw): return _result(lambda: manager.inspect(**_Inspect.model_validate(kw).model_dump()))
    def action(name, kw): return _result(lambda: manager.act(action=name, **_Act.model_validate(kw).model_dump()))
    def screenshot(**kw): return _result(lambda: manager.screenshot(**_Screenshot.model_validate(kw).model_dump()))
    def upload(**kw): return _result(lambda: manager.upload(**_Upload.model_validate(kw).model_dump()))
    def tabs(**kw): return _result(lambda: manager.tabs(**_Session.model_validate(kw).model_dump()))
    def switch(**kw): return _result(lambda: manager.switch_tab(**_Switch.model_validate(kw).model_dump()))
    def check_links(**kw): return _result(lambda: manager.check_links(**_CheckLinks.model_validate(kw).model_dump()))
    def close(**kw): return _result(lambda: manager.close(**_Session.model_validate(kw).model_dump()))
    tools = [StructuredTool.from_function(func=open_page,name="browser_open",description="Open an absolute HTTP(S) URL in an isolated model-selected browser session.",args_schema=_Open),StructuredTool.from_function(func=inspect,name="browser_inspect",description="Inspect page text, semantic accessibility snapshot, forms and interactive element refs.",args_schema=_Inspect)]
    for name in ("click","type","select","scroll","wait","back","download"):
        tools.append(StructuredTool.from_function(func=lambda _name=name, **kw: action(_name, kw),name=f"browser_{name}",description=f"Perform validated browser {name}; sensitive terminal actions return confirmation_required.",args_schema=_Act))
    tools += [StructuredTool.from_function(func=screenshot,name="browser_screenshot",description="Save a screenshot in the private session artifact directory.",args_schema=_Screenshot),StructuredTool.from_function(func=upload,name="browser_upload",description="Upload an allowed local file through a current file-input ref.",args_schema=_Upload),StructuredTool.from_function(func=check_links,name="browser_check_links",description="Validate rendered HTTP(S) links without navigating away; return status and broken-link results.",args_schema=_CheckLinks),StructuredTool.from_function(func=tabs,name="browser_tabs",description="List browser tabs and popups.",args_schema=_Session),StructuredTool.from_function(func=switch,name="browser_switch_tab",description="Switch to a model-selected tab id.",args_schema=_Switch),StructuredTool.from_function(func=close,name="browser_close",description="Close and clean an isolated browser session.",args_schema=_Session)]
    return tools
