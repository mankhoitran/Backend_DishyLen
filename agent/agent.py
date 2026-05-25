"""Main agent loop implementing tool-based reasoning."""

from __future__ import annotations

import json
import logging
from typing import Any

from google.genai import types
from sqlalchemy.orm import Session

from agent.parser import parse_input
from agent.prompts import system_prompt, tool_selection_prompt
from agent.tools import AgentTools
from config import get_settings
from services.gemini_client import GeminiClient
from services.search import SearchService

logger = logging.getLogger(__name__)


class FoodAgent:
    """Tool-driven food information agent."""

    def __init__(self, db: Session) -> None:
        self.settings = get_settings()
        self.gemini_client = GeminiClient()
        self.search_service = SearchService(self.gemini_client)
        self.tools = AgentTools(db=db, search_service=self.search_service)

        self.tool_registry = {
            "get_processed_dish": self.tools.get_processed_dish,
            "search_dish": self.tools.search_dish,
            "get_spicy_level": self.tools.get_spicy_level,
            "get_dish_macro": self.tools.get_dish_macro,
            "get_dish_summary": self.tools.get_dish_summary,
            "get_dish_image_url": self.tools.get_dish_image_url,
            "translate": self.tools.translate,
        }

    def run(self, query: str, target_language: str | None = None) -> dict[str, Any]:
        """Execute end-to-end dish information workflow."""

        parsed = parse_input(query)
        logger.info("Parsed input type=%s dish=%s", parsed.input_type, parsed.dish_name)

        tools_declaration = [self._build_tool_declaration()]
        initial_prompt = (
            f"{system_prompt()}\n\n"
            f"{tool_selection_prompt(parsed.normalized_query, parsed.dish_name, target_language)}"
        )

        contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part(text=initial_prompt)]),
        ]

        collected: dict[str, Any] = {
            "dish": parsed.dish_name,
            "spicy_level": "unknown",
            "macros": {},
            "summary": "",
            "image_url": "",
            "source": "search",
        }

        for step in range(self.settings.max_agent_steps):
            logger.info("Agent step %s", step + 1)
            result = self.gemini_client.generate_with_contents(contents=contents, tools=tools_declaration)

            if result.function_calls:
                model_content = result.raw.candidates[0].content
                contents.append(model_content)

                for call in result.function_calls:
                    name = call["name"]
                    args = call.get("args", {})
                    logger.info("Tool call: %s args=%s", name, args)
                    tool_result = self._execute_tool(name, args)
                    collected = self._merge_collected(collected, tool_result)

                    contents.append(
                        types.Content(
                            role="tool",
                            parts=[
                                types.Part.from_function_response(
                                    name=name,
                                    response={"result": tool_result},
                                )
                            ],
                        )
                    )
                continue

            final_payload = self._safe_parse_payload(result.text)
            if final_payload:
                collected = self._merge_collected(collected, final_payload)
            break

        if collected.get("source") == "search":
            self.tools.persist_dish(
                dish_name=collected.get("dish", parsed.dish_name),
                spicy_level=collected.get("spicy_level", "unknown"),
                macros=collected.get("macros", {}),
                summary=collected.get("summary", ""),
            )

        if target_language and collected.get("summary"):
            translated = self.tools.translate(collected["summary"], target_language)
            collected["summary"] = translated.get("translated_text", collected["summary"])

        return {
            "dish": collected.get("dish", parsed.dish_name),
            "spicy_level": collected.get("spicy_level", "unknown"),
            "macros": collected.get("macros", {}),
            "summary": collected.get("summary", ""),
            "image_url": collected.get("image_url", ""),
            "source": collected.get("source", "search"),
        }

    def _execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a function call to the matching tool function."""

        fn = self.tool_registry.get(name)
        if not fn:
            return {"error": f"Unknown tool: {name}"}

        try:
            return fn(**args)
        except TypeError:
            return {"error": f"Invalid arguments for {name}", "args": args}
        except Exception as exc:  # pragma: no cover - defensive catch for API/runtime failures
            logger.exception("Tool execution failed: %s", name)
            return {"error": str(exc)}

    @staticmethod
    def _merge_collected(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        """Merge non-empty tool outputs into collected state."""

        merged = dict(current)
        for key in ("dish", "spicy_level", "summary", "image_url", "source"):
            value = incoming.get(key)
            if value:
                merged[key] = value

        incoming_macros = incoming.get("macros")
        if isinstance(incoming_macros, dict) and incoming_macros:
            merged["macros"] = incoming_macros

        if incoming.get("found") is True:
            merged["source"] = "database"

        return merged

    @staticmethod
    def _safe_parse_payload(text: str) -> dict[str, Any]:
        """Best-effort parse final JSON object."""

        payload = text.strip() if text else ""
        if not payload:
            return {}

        if payload.startswith("```"):
            payload = payload.strip("`")
            if payload.lower().startswith("json"):
                payload = payload[4:].strip()

        try:
            value = json.loads(payload)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _build_tool_declaration() -> types.Tool:
        """Declare function signatures exposed to Gemini."""

        schema_dish = {
            "type": "object",
            "properties": {"dish_name": {"type": "string"}},
            "required": ["dish_name"],
        }
        schema_translate = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "target_language": {"type": "string"},
            },
            "required": ["text", "target_language"],
        }

        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="get_processed_dish",
                    description="Retrieve dish data from local DB if present",
                    parameters=schema_dish,
                ),
                types.FunctionDeclaration(
                    name="search_dish",
                    description="Search web data for dish info",
                    parameters=schema_dish,
                ),
                types.FunctionDeclaration(
                    name="get_spicy_level",
                    description="Get dish spicy level",
                    parameters=schema_dish,
                ),
                types.FunctionDeclaration(
                    name="get_dish_macro",
                    description="Get dish nutrition macro estimates",
                    parameters=schema_dish,
                ),
                types.FunctionDeclaration(
                    name="get_dish_summary",
                    description="Get concise dish summary",
                    parameters=schema_dish,
                ),
                types.FunctionDeclaration(
                    name="get_dish_image_url",
                    description="Get one best representative image URL for a dish",
                    parameters=schema_dish,
                ),
                types.FunctionDeclaration(
                    name="translate",
                    description="Translate a text into target language",
                    parameters=schema_translate,
                ),
            ]
        )
