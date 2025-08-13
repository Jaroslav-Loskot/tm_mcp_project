import os
import re
import json
import datetime
from functools import lru_cache
from typing import Annotated, Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from simple_salesforce import Salesforce, SalesforceMalformedRequest
from typing_extensions import TypedDict

from langgraph.prebuilt import ToolNode
from langchain_aws.chat_models.bedrock import ChatBedrock
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

import datetime

import mcp_salesforce.helpers as helpers
import mcp_salesforce.core_schema as core_schema

# --------------------------------------------------------------------------------------
# ENV & CONNECTION
# --------------------------------------------------------------------------------------
load_dotenv(override=True)

SF: Optional[Salesforce] = None
try:
    SF = helpers.get_sf_connection()
except Exception:
    SF = None  # connect lazily on first tool use


def init_chat_model(model_key: str = "NOVA_LITE_MODEL_ID") -> ChatBedrock:
    """Initialize the Bedrock chat model."""
    model_id = os.environ[model_key]
    region = os.environ["AWS_REGION"]
    return ChatBedrock(model_id=model_id, region_name=region, model_kwargs={"temperature": 0})


# --------------------------------------------------------------------------------------
# STATE
# --------------------------------------------------------------------------------------
class AgentContext(TypedDict):
    pass


class State(TypedDict):
    messages: Annotated[list, add_messages]




def _trace_as_text(state: Dict[str, Any]) -> str:
    """Return a readable transcript string (no printing)."""
    messages = state["messages"] if isinstance(state, dict) and "messages" in state else state
    lines: List[str] = []
    lines.append("\n💬 Conversation Trace\n" + "-" * 80)

    for idx, m in enumerate(messages, 1):
        if isinstance(m, HumanMessage):
            lines.append(f"\n[{idx}] 🧑‍💻 User:\n  {str(m.content).strip()}")

        elif isinstance(m, AIMessage):
            lines.append(f"\n[{idx}] 🤖 AI:")
            content = m.content
            if isinstance(content, list):  # Claude/Bedrock blocks
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        t = (block.get("text") or "").strip()
                        if t:
                            lines.append("  " + t.replace("\n", "\n  "))
                    elif btype == "tool_use":
                        name = block.get("name")
                        args = block.get("input")
                        lines.append(f"  🔧 Tool Call → {name}({json.dumps(args, ensure_ascii=False)})")
            elif isinstance(content, str):
                if content.strip():
                    lines.append("  " + content.strip().replace("\n", "\n  "))
                if getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        lines.append(f"  🔧 Tool Call → {tc['name']}({json.dumps(tc.get('args', {}), ensure_ascii=False)})")
            else:
                lines.append(f"  ⚠️ Unknown AIMessage content format: {type(content)}")

        elif isinstance(m, ToolMessage):
            status = getattr(m, "status", "success")
            lines.append(f"\n[{idx}] 🛠️ Tool Response [{status}]:")
            c = m.content
            if isinstance(c, (dict, list)):
                lines.append("  " + json.dumps(c, indent=2, ensure_ascii=False).replace("\n", "\n  "))
            elif isinstance(c, str):
                try:
                    loaded = json.loads(c)
                    lines.append("  " + json.dumps(loaded, indent=2, ensure_ascii=False).replace("\n", "\n  "))
                except Exception:
                    lines.append("  " + str(c).strip().replace("\n", "\n  "))
            else:
                lines.append("  " + str(c))

        elif isinstance(m, SystemMessage):
            lines.append(f"\n[{idx}] ⚙️ SystemPrompt: (omitted)")

        else:
            lines.append(f"\n[{idx}] ⚠️ {type(m).__name__}: {m}")

    lines.append("-" * 80 + "\n")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# CORE FIELD INDEX (tiny whitelist to keep tokens low)
# --------------------------------------------------------------------------------------
def _find(desc: Dict[str, Any], candidates: List[str]) -> Optional[Dict[str, Any]]:
    for f in desc.get("fields", []):
        if f.get("name") in candidates:
            return f
    return None


def _schema_entry(f: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "api": f["name"],
        "label": f.get("label"),
        "type": f.get("type"),
        "filterable": bool(f.get("filterable")),
        "sortable": bool(f.get("sortable")),
    }
    if f.get("type") == "picklist":
        out["values"] = [pv["value"] for pv in f.get("picklistValues", []) if pv.get("active", True)]
    if f.get("referenceTo"):
        out["references"] = f.get("referenceTo")
        if f.get("relationshipName"):
            out["relationshipName"] = f["relationshipName"]
    return out


# --------------------------------------------------------------------------------------
# TOOLS
# --------------------------------------------------------------------------------------
@tool
def parse_salesforce_date_tool(natural_input: str, want_datetime: bool = False) -> str:
    """
    Convert natural language date/time to an ISO string for SOQL:
      - Date: YYYY-MM-DD
      - Datetime: YYYY-MM-DDThh:mm:ssZ (UTC)
    """
    today = datetime.datetime.utcnow()
    system_prompt = (
        "You convert natural language date/time expressions into a single ISO string for Salesforce SOQL.\n"
        f"Current UTC datetime is: {today.strftime('%Y-%m-%dT%H:%M:%SZ')}\n\n"
        "Output rules:\n"
        "- If a time-of-day is provided or implied, output a UTC datetime in format YYYY-MM-DDThh:mm:ssZ.\n"
        "- Otherwise output a date in YYYY-MM-DD.\n"
        "- Durations: week=7 days, month=30 days, quarter=90 days, year=365 days.\n"
        "- Output only the ISO string; no explanations."
    )
    llm = init_chat_model("NOVA_LITE_MODEL_ID")
    resp = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=f"Convert: {natural_input}\nForce datetime: {bool(want_datetime)}")]
    ).content
    mdt = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", resp)
    if mdt:
        return mdt.group(0)
    md = re.search(r"\d{4}-\d{2}-\d{2}", resp)
    if md:
        return md.group(0)
    raise ValueError(f"Could not parse date from: {resp}")


@tool
def validate_soql_tool(soql: str) -> Dict[str, Any]:
    """
    Execute the SOQL with LIMIT 1 to validate syntax; also try COUNT() estimate for simple selects.
    Returns: { "soql": "<query>", "valid": bool, "sample": {...}|None, "count": int|None, "error": str|None }
    """
    sf = helpers.get_sf_connection()
    try:
        test_query = soql if re.search(r"\blimit\s+\d+\b", soql, re.IGNORECASE) else soql.rstrip() + " LIMIT 1"
        sample_res = sf.query(test_query)
        sample_record = None
        if sample_res.get("records"):
            sample_record = helpers._strip_attributes([sample_res["records"][0]])[0]

        count_val = None
        simple_match = re.match(
            r"^\s*select\s+.+?\s+from\s+([a-zA-Z0-9_]+)\s*(where\s+.+?)?(order\s+by\s+.+?)?$",
            soql,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if simple_match and " group by " not in soql.lower():
            obj = simple_match.group(1)
            where = simple_match.group(2) or ""
            count_query = f"SELECT COUNT() FROM {obj} {where}"
            count_res = sf.query(count_query)
            count_val = count_res.get("totalSize", None)

        return {"soql": soql, "valid": True, "sample": sample_record, "count": count_val, "error": None}
    except SalesforceMalformedRequest as e:
        return {"soql": soql, "valid": False, "sample": None, "count": None, "error": f"MALFORMED_QUERY: {getattr(e, 'content', e)}"}
    except Exception as e:
        return {"soql": soql, "valid": False, "sample": None, "count": None, "error": str(e)}


@tool
def list_stage_names_tool(active_only: bool = True) -> List[str]:
    """Return ONLY the available values for Opportunity.StageName (active by default)."""
    sf = helpers.get_sf_connection()
    meta = sf.Opportunity.describe()
    for field in meta["fields"]:
        if field["name"] == "StageName":
            vals = []
            for pv in field.get("picklistValues", []):
                if (not active_only) or pv.get("active", False):
                    vals.append(pv["value"])
            return vals
    raise ValueError("StageName field not found on Opportunity.")



@tool
def get_salesforce_field_schema(field_api: str) -> Dict[str, Any]:
    """
    Compact schema for ONE allow-listed field.

    Input:
      - field_api: '<Object>.<Field>' (e.g., 'Account.BillingCountry', 'Opportunity.StageName').

    Behavior:
      - Looks up the field strictly in the allow-list (core_schema).
      - Case-insensitive matching for both the friendly suffix and the API name.
      - Returns:
        { "object": <Object>, "field": <resolved API>, "attribute": "<Friendly.Key>", "schema": {...} }
        or { "object": <Object>, "field": "<input suffix>", "error": "Field not in allow-list." }
    """
    try:
        key_raw = (field_api or "").strip()
        if "." not in key_raw:
            return {
                "field": key_raw,
                "error": "Use '<Object>.<Field>' format, e.g., 'Account.BillingCountry'."
            }

        obj_part, fld_part = key_raw.split(".", 1)
        obj_norm = obj_part.strip().capitalize()   # normalize 'account' → 'Account'
        fld_raw  = fld_part.strip()

        # 1) Try direct friendly key
        friendly = f"{obj_norm}.{fld_raw}"
        sch = core_schema.get_schema_for_attribute(friendly)
        if sch:
            return {
                "object": obj_norm,
                "field": sch["api"],
                "attribute": friendly,
                "schema": sch,
            }

        # 2) Case-insensitive / API-name lookup within same object
        idx = core_schema.build_core_field_index_cached()
        prefix = f"{obj_norm}."
        fld_lower = fld_raw.lower()
        for attr, s in idx.items():
            if not attr.startswith(prefix):
                continue
            suffix = attr.split(".", 1)[1].lower()          # friendly suffix
            api_lower = (s.get("api") or "").lower()        # actual API name
            if fld_lower == suffix or fld_lower == api_lower:
                return {
                    "object": obj_norm,
                    "field": s["api"],
                    "attribute": attr,
                    "schema": s,
                }

        return {
            "object": obj_norm,
            "field": fld_raw,
            "error": "Field not in allow-list."
        }

    except Exception as e:
        return {"field": field_api, "error": str(e)}
    




@tool
def list_core_attribute_names_tool() -> List[str]:
    """Return the friendly allow-list like ['Opportunity.Id', 'Account.Type', ...]."""
    return core_schema.list_allowed_attribute_names()


@tool
def llm_pick_best_name_matches_tool(
    query: str,
    k: int = 5,
    max_per_type: int = 400,
    shortlist_cap: int = 80,
    force: bool = True,
) -> List[Dict[str, Any]]:
    """
    Let the LLM pick top-K best matches across Account + Opportunity.
    Always returns both IDs (non-applicable one is None).

    Output rows:
    {
      "account_id": "001... or None",
      "opportunity_id": "006... or None",
      "name": "<Name>",
      "type": "Account" | "Opportunity",
      "match_score": 0..100
    }
    """
    
    return helpers.llm_pick_best_name_matches_tool(query, k, max_per_type, shortlist_cap, force)


@tool
def resolve_owner_names_tool(owner_ids: List[str]) -> List[Dict[str, Optional[str]]]:
    """
    Translate a list of OwnerId values into [{id, name}] pairs.
    Supports Users (005...) and Groups/Queues (00G...). Missing/unknown -> name=None.
    Preserves input order and de-duplicates internally.
    """
    return helpers.resolve_owner_names_tool(owner_ids)


# Register tool list (unique, ordered)
tools = [
    parse_salesforce_date_tool,
    validate_soql_tool,
    list_stage_names_tool,
    get_salesforce_field_schema,
    list_core_attribute_names_tool,
    llm_pick_best_name_matches_tool,
    resolve_owner_names_tool,
]


# --------------------------------------------------------------------------------------
# BOT MANAGER
# --------------------------------------------------------------------------------------
def bot_manager(state: State):
    llm = init_chat_model("NOVA_LITE_MODEL_ID")
    llm_with_tools = llm.bind_tools(tools)

    SYSTEM_PROMPT = """You are a SOQL Builder Agent. Be terse. Minimize tokens and tool calls.

    # Today's date is: {today}

STRICT CADENCE (one tool call per assistant turn)
- Do NOT emit more than one tool call in a single message.
- Do NOT include any free text when calling a tool.
- Reuse prior results; never call the same tool with identical args twice.

HARD CONSTRAINT: SCHEMA CALLS ONLY FROM WHITELIST
- Call list_core_attribute_names_tool() once at the start of the request; treat the returned array as the *only* allowed attributes.
- When calling get_salesforce_field_schema, you MUST derive arguments from a whitelist item:
  • Parse "<Object>.<Field>" → object_api = "<Object>", field_api = "<Field>".
  • Examples:
      - "Opportunity.StageName" → get_salesforce_field_schema("Opportunity", "StageName")
      - "Account.BillingCountry" → get_salesforce_field_schema("Account", "BillingCountry")
- NEVER pass user free-text or non-whitelisted names to get_salesforce_field_schema.
- Pick “important” attributes ONLY from the whitelist; do not invent or probe outside it.

TURN CACHE (internal only; do not print)
TOOL_CACHE = {
  names: [...],                  # from list_core_attribute_names_tool()
  important_attrs: [...],        # chosen subset from names
  schema: {attr -> schema},      # get_salesforce_field_schema(object_api, field_api) for used attrs only
  stage_names_true: [...],       # list_stage_names_tool(True) if needed
  matches: [...],                # from llm_pick_best_name_matches_tool
  date_literals: {...}           # e.g., {"created": "LAST_QUARTER"}
}

WORKFLOW (mandatory)
0) ENTITY RESOLUTION (if a specific name is present):
   - If the request mentions a concrete account/opportunity name (e.g., “Allica Bank”, “ACME Renewal”),
     call llm_pick_best_name_matches_tool(query=<name>, k=3, force=True) and cache to TOOL_CACHE.matches.
   - Use the TOP match for filtering:
     • If Opportunity requested or top match.type == "Opportunity" → WHERE Id = '<opportunity_id>'.
     • Else (Account) → for Opportunity queries use WHERE AccountId = '<account_id>'; for Account queries WHERE Id = '<account_id>'.
   - Proceed even if imperfect; note the chosen entity in final JSON notes. Do NOT ask for clarification.

1) LIST NAMES (once):
   - Call list_core_attribute_names_tool(); cache as TOOL_CACHE.names. Do not print.

2) PICK IMPORTANT ATTRIBUTES (subset of whitelist only):
   - SELECT: only requested; default Opportunity.Id, Opportunity.Name.
   - WHERE/ORDER BY: only fields required (CreatedDate, StageName/IsClosed/IsWon, Account.* country, Currency, Owner, Amount, CloseDate, etc.).
   - Save into TOOL_CACHE.important_attrs.

3) FETCH SCHEMA (only for used fields; whitelist-derived calls only):
   - For each attribute used in WHERE/ORDER BY or needing type/picklist/relationship:
     • Split "<Object>.<Field>" → get_salesforce_field_schema("<Object>", "<Field>").
     • One attribute per turn; cache in TOOL_CACHE.schema[attr].
   - Do NOT fetch schema for unused SELECT-only fields unless needed for type/picklist validation.

4) DATES (type-aware, minimal):
   - Prefer Salesforce date literals (no tool): LAST_QUARTER, LAST_N_DAYS:30/90, THIS_YEAR, LAST_YEAR.
   - If no literal applies, call parse_salesforce_date_tool once:
     • Datetime (CreatedDate, LastModifiedDate, SystemModstamp): YYYY-MM-DDThh:mm:ssZ (no quotes).
     • Date-only (CloseDate): YYYY-MM-DD (no quotes).
   - Never quote date/datetime literals.

5) STAGE LOGIC (only if relevant):
   - If stage/open/won/lost/closed/pipeline or specific labels are mentioned:
     • Prefer booleans: open→IsClosed=false; won→IsWon=true; lost→IsClosed=true AND IsWon=false.
     • Call list_stage_names_tool(True) once to record labels; for explicit labels, use StageName IN/NOT IN with EXACT values.

6) BUILD MINIMAL SOQL:
   - SELECT minimal fields; WHERE only necessary filters (include entity filter from step 0); no quotes for date/datetime literals.
   - ORDER BY only sortable fields; LIMIT when requested.
   - Use relationship paths only if consistent with known whitelist attributes/schemas.

7) VALIDATE → RETURN (with safe fallback):
   - Call validate_soql_tool once.
   - If MALFORMED_QUERY indicates date/datetime mismatch, correct literal shape or switch to a date literal and retry exactly once (no new discovery).
   - If validation succeeds:
       • Use validate_soql_tool.count for approx_row_count.
       • If validate_soql_tool.sample exists, include it as `sample`.
       • Return the SOQL verbatim.
   - If still invalid or no valid SOQL can be formed:
       • Return "soql": "" and "approx_row_count": null, with `comment` starting `!comment!` describing the blocking reason.
       • Do NOT ask the user to clarify.

FINAL OUTPUT (JSON only; no extra text):
{
  "soql": "<final SOQL or empty string>",
  "approx_row_count": <number or null>,
  "resolved_stage_names": ["<StageA>", "<StageB>", ...],
  "sample": { ... },
  "comment": "!comment! <reason if soql is empty>",
  "notes": "<=1 short sentence (include chosen entity if any)>"
}
"""
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    new_message = llm_with_tools.invoke(messages)
    return {"messages": messages + [new_message]}


# --------------------------------------------------------------------------------------
# ROUTING
# --------------------------------------------------------------------------------------
def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


graph_builder = StateGraph(State, context_schema=AgentContext)
graph_builder.add_node("bot_manager", bot_manager)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.set_entry_point("bot_manager")
graph_builder.add_conditional_edges("bot_manager", route_tools, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "bot_manager")
graph = graph_builder.compile()


# --------------------------------------------------------------------------------------
# PUBLIC ENTRY
# --------------------------------------------------------------------------------------
def call_agent_generate_soql(
    human_input: str,
    *,
    stream: bool = False,
    with_trace: bool = False,
    return_state: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    Run the agent and return only the final LLM output by default.

    Args:
      human_input: natural language request
      stream: if True, stream graph execution (trace is accumulated if with_trace=True)
      with_trace: if True, include a pretty-printed tool/step trace (string)
      return_state: if True, include the raw final graph state

    Returns:
      - If with_trace=False and return_state=False:
          -> last assistant message content (string)
      - Otherwise:
          -> dict with keys { "output": ..., "trace": ..., "state": ... }
    """
    user_message = HumanMessage(content=human_input)
    state: Dict[str, Any] = {"messages": [user_message]}

    final_state: Optional[Dict[str, Any]] = None
    trace_text = ""

    if stream:
        trace_chunks: List[str] = []
        for s in graph.stream(state, stream_mode="values"):
            final_state = s
            if with_trace:
                trace_chunks.append(_trace_as_text(s))
        if with_trace:
            trace_text = "\n".join(trace_chunks)
    else:
        final_state = graph.invoke(state)
        if with_trace:
            trace_text = _trace_as_text(final_state)

    # Extract final assistant content safely
    messages = final_state["messages"]
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    output = getattr(last_ai, "content", None) if last_ai is not None else getattr(messages[-1], "content", "")

    if not with_trace and not return_state:
        return output

    return {"output": output, "trace": trace_text, "state": final_state if return_state else None}
