"""LangChain BaseTool wrappers that call the public MCP servers directly.

These replace the private IBM packages (iotagent, fmsr_agent, tsfmagent, wo_agent)
that AgentHive's tool layer normally imports from github.ibm.com.

Each tool spawns the corresponding MCP server as a subprocess via stdio (the
same mechanism the PlanExecuteRunner uses) and calls the tool synchronously.

Tool groups mirror the AgentHive tool module structure:
  iot_bms_tools   → IoTAgent MCP server  (sites, assets, sensors, history)
  fmsr_tools      → FMSRAgent MCP server (get_failure_modes, get_failure_mode_sensor_mapping)
  tsfm_tools      → TSFMAgent MCP server (get_ai_tasks, get_tsfm_models, run_tsfm_forecasting)
  wo_tools        → Utilities MCP server (current_date_time, json_reader) — WO needs CouchDB
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Type

from langchain.tools import BaseTool  # type: ignore[import]
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

# ── MCP call helper ───────────────────────────────────────────────────────────

def _call_mcp_tool_sync(server_name: str, tool_name: str, args: dict) -> str:
    """Call an MCP tool synchronously, spinning up the server as a subprocess."""
    from workflow.executor import DEFAULT_SERVER_PATHS, _call_tool  # type: ignore[import]

    server_path = DEFAULT_SERVER_PATHS.get(server_name)
    if server_path is None:
        return json.dumps({"error": f"Unknown server: {server_name}"})

    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(_call_tool(server_path, tool_name, args))
        loop.close()
        return result
    except Exception as exc:  # noqa: BLE001
        _log.warning("MCP tool call %s/%s failed: %s", server_name, tool_name, exc)
        return json.dumps({"error": str(exc)})


# ── IoT tools ─────────────────────────────────────────────────────────────────

class _SitesTool(BaseTool):
    name: str = "sites"
    description: str = "Returns list of available IoT sites."

    def _run(self, query: str = "") -> str:
        return _call_mcp_tool_sync("IoTAgent", "sites", {})

    async def _arun(self, query: str = "") -> str:
        return self._run(query)


class _AssetsTool(BaseTool):
    name: str = "assets"
    description: str = (
        "Returns assets at a given site. Input: site name (e.g. 'MAIN')."
    )

    def _run(self, query: str) -> str:
        return _call_mcp_tool_sync("IoTAgent", "assets", {"site_name": query.strip()})

    async def _arun(self, query: str) -> str:
        return self._run(query)


class _SensorsTool(BaseTool):
    name: str = "sensors"
    description: str = (
        "Returns sensors for a given asset. Input JSON: {\"site_name\": \"MAIN\", \"asset_id\": \"CH-1\"}"
    )

    def _run(self, query: str) -> str:
        try:
            args = json.loads(query)
        except json.JSONDecodeError:
            parts = [p.strip() for p in query.split(",")]
            args = {"site_name": parts[0] if parts else "MAIN",
                    "asset_id": parts[1] if len(parts) > 1 else query.strip()}
        return _call_mcp_tool_sync("IoTAgent", "sensors", args)

    async def _arun(self, query: str) -> str:
        return self._run(query)


class _HistoryTool(BaseTool):
    name: str = "history"
    description: str = (
        "Returns historical sensor data for an asset. "
        "Input JSON: {\"site_name\": \"MAIN\", \"asset_id\": \"CH-1\", "
        "\"start\": \"2024-01-01T00:00:00\", \"final\": \"2024-01-02T00:00:00\"}"
    )

    def _run(self, query: str) -> str:
        try:
            args = json.loads(query)
        except json.JSONDecodeError:
            args = {"site_name": "MAIN", "asset_id": query.strip(), "start": "2024-01-01T00:00:00"}
        return _call_mcp_tool_sync("IoTAgent", "history", args)

    async def _arun(self, query: str) -> str:
        return self._run(query)


iot_bms_tools: list[BaseTool] = [_SitesTool(), _AssetsTool(), _SensorsTool(), _HistoryTool()]

iot_agent_name = "IoT Data Download"
iot_agent_description = (
    "Can provide information about IoT sites, asset details, sensor data, and retrieve "
    "historical data and metadata for various assets and equipment"
)
iot_bms_fewshots = (
    "Question: What IoT sites are available?\n"
    "Thought: I need to list the available IoT sites.\n"
    "Action: sites\nAction Input: \n"
    "Observation: {\"sites\": [\"MAIN\"]}\n"
    "Thought: I have the answer.\nFinal Answer: The available IoT site is MAIN.\n"
)
iot_task_examples = ["What IoT sites are available?", "List assets at MAIN site."]


# ── FMSR tools ────────────────────────────────────────────────────────────────

class _GetFailureModesTool(BaseTool):
    name: str = "get_failure_modes"
    description: str = (
        "Returns failure modes for an asset type. Input: asset name (e.g. 'Chiller', 'AHU')."
    )

    def _run(self, query: str) -> str:
        return _call_mcp_tool_sync("FMSRAgent", "get_failure_modes", {"asset_name": query.strip()})

    async def _arun(self, query: str) -> str:
        return self._run(query)


class _GetFailureModeSensorMappingTool(BaseTool):
    name: str = "get_failure_mode_sensor_mapping"
    description: str = (
        "Returns mapping between failure modes and sensors for an asset. "
        "Input JSON: {\"asset_name\": \"Chiller\", \"failure_modes\": [\"fm1\"], \"sensors\": [\"s1\"]}"
    )

    def _run(self, query: str) -> str:
        try:
            args = json.loads(query)
        except json.JSONDecodeError:
            args = {"asset_name": query.strip(), "failure_modes": [], "sensors": []}
        return _call_mcp_tool_sync("FMSRAgent", "get_failure_mode_sensor_mapping", args)

    async def _arun(self, query: str) -> str:
        return self._run(query)


fmsr_tools: list[BaseTool] = [_GetFailureModesTool(), _GetFailureModeSensorMappingTool()]

fmsr_agent_name = "Failure Mode and Sensor Relevancy Expert for Industrial Asset"
fmsr_agent_description = (
    "Can provide information about failure modes, mapping between failure modes and sensors, "
    "and can generate machine learning recipes for specific failures"
)
fmsr_fewshots = (
    "Question: List all failure modes of asset Chiller.\n"
    "Thought: I need to get the failure modes for Chiller.\n"
    "Action: get_failure_modes\nAction Input: Chiller\n"
    "Observation: {\"asset_name\": \"Chiller\", \"failure_modes\": [\"Refrigerant Leak\", \"Compressor Failure\"]}\n"
    "Thought: I have the failure modes.\nFinal Answer: The failure modes for Chiller are: Refrigerant Leak, Compressor Failure.\n"
)
fmsr_task_examples = ["List all failure modes of asset Chiller.", "Get sensor mapping for Chiller failures."]


# ── TSFM tools ────────────────────────────────────────────────────────────────

class _GetAITasksTool(BaseTool):
    name: str = "get_ai_tasks"
    description: str = "Returns the list of available AI task types supported by TSFM."

    def _run(self, query: str = "") -> str:
        return _call_mcp_tool_sync("TSFMAgent", "get_ai_tasks", {})

    async def _arun(self, query: str = "") -> str:
        return self._run(query)


class _GetTSFMModelsTool(BaseTool):
    name: str = "get_tsfm_models"
    description: str = "Returns the list of available pre-trained TSFM model checkpoints."

    def _run(self, query: str = "") -> str:
        return _call_mcp_tool_sync("TSFMAgent", "get_tsfm_models", {})

    async def _arun(self, query: str = "") -> str:
        return self._run(query)


class _RunTSFMForecastingTool(BaseTool):
    name: str = "run_tsfm_forecasting"
    description: str = (
        "Runs zero-shot time series forecasting using a TSFM model. "
        "Input JSON with dataset_path, timestamp_column, target_columns."
    )

    def _run(self, query: str) -> str:
        try:
            args = json.loads(query)
        except json.JSONDecodeError:
            return json.dumps({"error": "Input must be JSON with dataset_path, timestamp_column, target_columns."})
        return _call_mcp_tool_sync("TSFMAgent", "run_tsfm_forecasting", args)

    async def _arun(self, query: str) -> str:
        return self._run(query)


tsfm_tools: list[BaseTool] = [_GetAITasksTool(), _GetTSFMModelsTool(), _RunTSFMForecastingTool()]

tsfm_agent_name = "Time Series Analytics and Forecasting"
tsfm_agent_description = (
    "Can assist with time series analysis, forecasting, anomaly detection, and model selection, "
    "and supports pretrained models, context length specifications, and regression tasks"
)
tsfm_fewshots = (
    "Question: What types of time series analysis are supported?\n"
    "Thought: I need to list the available AI task types.\n"
    "Action: get_ai_tasks\nAction Input: \n"
    "Observation: {\"tasks\": [\"forecasting\", \"anomaly_detection\"]}\n"
    "Thought: I have the answer.\nFinal Answer: The supported task types are: forecasting, anomaly_detection.\n"
)
tsfm_task_examples = ["What types of time series analysis are supported?", "List available TSFM models."]


# ── WO tools (Utilities MCP — date/time helpers; WO data needs CouchDB) ───────

class _CurrentDateTimeTool(BaseTool):
    name: str = "current_date_time"
    description: str = "Returns the current UTC date and time."

    def _run(self, query: str = "") -> str:
        return _call_mcp_tool_sync("Utilities", "current_date_time", {})

    async def _arun(self, query: str = "") -> str:
        return self._run(query)


class _JsonReaderTool(BaseTool):
    name: str = "json_reader"
    description: str = "Reads and parses a JSON file from disk. Input: file path."

    def _run(self, query: str) -> str:
        return _call_mcp_tool_sync("Utilities", "json_reader", {"file_name": query.strip()})

    async def _arun(self, query: str) -> str:
        return self._run(query)


wo_tools: list[BaseTool] = [_CurrentDateTimeTool(), _JsonReaderTool()]

wo_agent_name = "Work Order Management"
wo_agent_description = (
    "Can retrieve and summarize work order information for industrial equipment"
)
wo_fewshots = (
    "Question: What is the current date and time?\n"
    "Thought: I need to get the current date and time.\n"
    "Action: current_date_time\nAction Input: \n"
    "Observation: {\"currentDateTime\": \"2024-01-15T10:30:00Z\"}\n"
    "Thought: I have the answer.\nFinal Answer: The current date and time is 2024-01-15T10:30:00Z.\n"
)
wo_task_examples = ["Get the work order of equipment CWC04013 for year 2017."]
