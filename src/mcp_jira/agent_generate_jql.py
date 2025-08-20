import datetime
import os
import json
import re
import warnings
from dotenv import load_dotenv
from jira import JIRA
from typing import Annotated, Dict, List

from typing_extensions import TypedDict

from langgraph.prebuilt import ToolNode
from langchain_aws.chat_models.bedrock import ChatBedrock
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import mcp_jira.helpers as helpers

# warnings.filterwarnings(action="ignore", message=r"datetime.datetime.utcnow")
warnings.filterwarnings(
    action="error",
    category=DeprecationWarning,
    message="`config_type` is deprecated" # Optional, but good for being specific
)

load_dotenv(override=True)
JIRA_URL = os.getenv("JIRA_BASE_URL")
JIRA_USER = os.getenv("JIRA_EMAIL","")
JIRA_TOKEN = os.getenv("JIRA_API_TOKEN","")

DEFAULT_CATEGORY = os.getenv("DEFAULT_PROJECT_CATEGORY", "")
EXCLUDED_KEYS = [k.strip() for k in os.getenv("EXCLUDED_PROJECT_KEYS", "").split(",") if k.strip()]

jira = JIRA(server=JIRA_URL, basic_auth=(JIRA_USER, JIRA_TOKEN))

def pretty_print_messages(state):
    print("💬 Conversation:\n" + "-" * 60)
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            print(f"🧑‍💻 User:\n  {m.content}\n")

        elif isinstance(m, AIMessage):
            print("🤖 Bot:")

            content = m.content

            # Claude structured response (list of blocks)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):  # ✅ safeguard
                        block_type = block.get("type")
                        if block_type == "text":
                            text = block.get("text", "")
                            print("  " + text.strip().replace("\n", "\n  "))
                        elif block_type == "tool_use":
                            print(f"  🔧 Tool Call → {block.get('name')}({json.dumps(block.get('input', {}))})\n")
                    else:
                        print(f"⚠️ Unexpected block type: {type(block)} {block}")

            # Nova Lite / plain string response
            elif isinstance(content, str):
                print(f"  {content.strip()}")
                if getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        name = tc.get("name", "unknown")
                        args = json.dumps(tc.get("args", {}))
                        print(f"  🔧 Tool Call → {name}({args})\n")

            # Unknown content format
            else:
                print(f"⚠️ Unknown AIMessage content format: {content}")

            print()

        elif isinstance(m, ToolMessage):
            status = getattr(m, "status", "success")
            print(f"🛠️ Tool Response [{status}]:")
            print(f"  {m.content}\n")

        else:
            print(f"⚠️ Unknown message type: {m}\n")

def init_chat_model(model_key: str = "CLAUDE_MODEL_ID") -> ChatBedrock:
    model_id = os.environ[model_key]
    region = os.environ["AWS_REGION"]

    return ChatBedrock(
        model=model_id,
        region=region,
        model_kwargs={"temperature": 0}
    )

class AgentContext(TypedDict):
    pass

class State(TypedDict):
    messages: Annotated[list, add_messages]

#### TOOLS ---------------------------------------------------------------------------------------------------------------

@tool
def list_issue_type_statuses_tool(project_key: str) -> List[Dict[str, List[str]]]:
    """
    Lists valid Jira statuses grouped by issue type for a given project.

    Returns:
      [
        { "type": "Bug", "available_statuses": ["To Do", "In Progress", "Done"] },
        { "type": "Story", "available_statuses": ["Backlog", "Selected", "Done"] },
        ...
      ]
    """
    try:
        # Assuming helpers.jira.issue_types_for_project exists and works.
        issue_types = helpers.jira.issue_types_for_project(project_key)
        result = []
        for it in issue_types:
            statuses = getattr(it, "statuses", [])
            names = [s.name for s in statuses]
            result.append({"type": it.name, "available_statuses": names})
        return result
    except Exception as e:
        raise ValueError(f"Failed to retrieve issue-type statuses for project '{project_key}': {e}")


@tool
def parse_jira_date_tool(input_str: str) -> str:
    """
    Uses LLM to convert a natural language date expression into a Jira-compatible ISO date (YYYY-MM-DD).

    Example:
        input_str = "last month"
        → "2025-07-01"

    Returns:
        Date string like "2025-07-01"
    """
    today = datetime.date.today().isoformat()
    system_prompt = (
        f"You are a date conversion assistant for Jira JQL queries.\n"
        f"Today is: {today}\n\n"
        "Your task is to convert natural language time filters like 'last month', 'past 2 quarters', or "
        "'updated 3 weeks ago' into an absolute date in ISO format (YYYY-MM-DD).\n\n"
        "Guidelines:\n"
        "- Use 'updated >= <date>' if the input includes words like 'updated', 'changed', or 'modified'.\n"
        "- Otherwise, use 'created >= <date>'.\n"
        "- Convert durations as follows:\n"
        "  - weeks -> 7 days per week\n"
        "  - months -> 30 days per month\n"
        "  - quarters -> 90 days per quarter\n"
        "  - years -> 365 days per year\n"
        "- Only return a single date string in the format YYYY-MM-DD.\n"
        "- Do not explain your reasoning. Output only the computed date.\n"
    )

    user_prompt = f"Convert to date: {input_str}"
    
    llm = init_chat_model("NOVA_LITE_MODEL_ID") 

    # ✅ Pass as a list of messages
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    text = str(response.content if hasattr(response, "content") else response)

    match = re.search(r"\d{4}-\d{2}-\d{2}", text)

    if match:
        return match.group(0)
    else:
        raise ValueError(f"Could not parse a valid date from response: {text}")

@tool
def resolve_project_name_tool(human_input: str) -> List[Dict[str, str]]:
    """
    Resolve a Jira project name from human-friendly input.
    Fetches available Jira projects and chooses the best match.

    Parameters:
    - human_input: Human-friendly name of the project (e.g., 'website revamp').

    Returns:
    - The matching Jira project name (e.g., 'Website Comapny'), or raises error if not found or invalid.
    """
    # Assuming helpers._resolve_project_name exists and returns a list of dicts.
    return helpers._resolve_project_name(human_input, os.environ["DEFAULT_PROJECT_CATEGORY"])


@tool
def check_jql(jql: str) -> Dict:
    """
    Executes a JQL query and returns an approximate count of matching Jira issues.

    Parameters:
    - jql: A valid Jira Query Language string (e.g., 'project = PROJ AND status = "To Do"')

    Returns:
    {
        "jql": "<your input>",
        "approximate_count": <integer>
    }

    Or in case of error:
    {
        "error": "...",
        "jql": "<your input>"
    }
    """
    try:
        count = jira.approximate_issue_count(jql_str=jql)
        return {"jql": jql, "approximate_count": count}
    except Exception as e:
        return {"error": str(e), "jql": jql}


# **IMPORTANT FIX**: Add the resolve_project_name_tool to the main tools list
tools = [
    parse_jira_date_tool, 
    check_jql, 
    list_issue_type_statuses_tool, 
    resolve_project_name_tool
    ]




def bot_manager(state: State):
    llm = init_chat_model("NOVA_LITE_MODEL_ID")
    # Bind all tools, including the project resolver
    llm_with_tools = llm.bind_tools(tools)

    SYSTEM_PROMPT = """You are a helpful assistant that converts natural language into valid Jira JQL queries.

    Core Rules:
    - Always resolve project names from the user input *before* generating JQL. Use the `resolve_project_name_tool` for this.
    - If the input contains any project names (e.g., 'UniCredit Italy', 'Austria DevOps'), call the `resolve_project_name_tool` with the project name.
    - Only proceed to generate JQL after all names are resolved.
    - IMPORTANT: If a project name is not found by the tool, or you consider the name too different, respond to the user that the project could not be found and ask for clarification.

    Tool Usage:
    - Always use tools to resolve project names, date expressions, and available issue types/statuses.
    - Never ask the user to clarify; use tools to disambiguate.
    - Once JQL is generated, call the `check_jql` tool:
    - If it errors -> fix and regenerate.
    - If 0 results -> reconsider filters (e.g. status, date, type).
    - If valid with results -> return the JQL and count in the specified JSON format.

    JQL Construction Guidelines:
    - Use only known project keys and priorities (from tools).
    - Use issue types and statuses only after confirming their availability via tools.
    - Only call the issue-type/status tool once per project unless the input changes.
    - Use `resolution in (Unresolved, EMPTY)` for open/unresolved/incomplete.
    - Use `resolution not in (Unresolved, EMPTY)` for closed/resolved/completed.
    - If filtering by project, date, or priority alone -> omit the resolution clause.
    - If priority is specified -> include it.
    - For specific types or statuses:
    - Use `type IN (...)`, `status IN (...)`, or their `NOT IN` counterparts.
    - **Always enclose all list items in single quotes**: 
        ✅ `type IN ('Bug', 'Story')` 
        ❌ `type IN (Bug, Story)`

    Date Handling:
    - Use `updated >=` if the input mentions update/change/modification.
    - Else use `created >=`.
    - For durations:
    - `1 month` -> `30d`, `1 quarter` -> `90d`, `1 year` -> `365d`.

    Always substitute exact tool outputs in the final JQL (e.g., if a tool returns `"2025-07-02"`, use that date verbatim).

    Return only JSON:
    ```json
    { "jql": "<generated JQL>", "approx_query_results": <number> }
    """
    # ensure messages is always a list[BaseMessage]
    messages: list = state.get("messages", [])
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    # invoke with messages (list of BaseMessage)
    new_message = llm_with_tools.invoke(messages)  # type: ignore[arg-type]

    return {"messages": messages + [new_message]}


def route_tools(state: State):
    messages: list = state.get("messages", [])
    if not messages:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    ai_message = messages[-1]
    if isinstance(ai_message, AIMessage) and getattr(ai_message, "tool_calls", None):
        if len(ai_message.tool_calls) > 0:
            return "tools"
    return END


graph_builder = StateGraph(State, context_schema=AgentContext)

graph_builder.add_node("bot_manager", bot_manager)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.set_entry_point("bot_manager")
graph_builder.add_conditional_edges(
    "bot_manager",
    route_tools,
    {"tools": "tools", END: END}
)
graph_builder.add_edge("tools", "bot_manager")
graph = graph_builder.compile()


from typing import cast

def call_agent_generate_jql(human_input: str):
    user_message = HumanMessage(content=human_input)
    state: State = {"messages": [user_message]}  # build correctly typed State
    return graph.invoke(cast(State, state))




