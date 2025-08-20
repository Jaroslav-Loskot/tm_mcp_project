import datetime
import os
import json
import re
import warnings
from jira import JIRA
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_aws.chat_models.bedrock import ChatBedrock
from typing import Annotated, Any, Dict, List, Optional, Union
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict
from langchain_core.messages import convert_to_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from mcp_common.utils.bedrock_wrapper import call_nova_lite
import mcp_jira.helpers as helpers
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph_supervisor import create_supervisor
from langchain_core.language_models import BaseChatModel


warnings.filterwarnings(action="ignore", message=r"datetime.datetime.utcnow")


load_dotenv(override=True)
JIRA_URL = os.getenv("JIRA_BASE_URL")
JIRA_USER = os.getenv("JIRA_EMAIL", "")
JIRA_TOKEN = os.getenv("JIRA_API_TOKEN", "")

DEFAULT_CATEGORY = os.getenv("DEFAULT_PROJECT_CATEGORY", "")
EXCLUDED_KEYS = [k.strip() for k in os.getenv("EXCLUDED_PROJECT_KEYS", "").split(",") if k.strip()]

jira = JIRA(server=JIRA_URL, basic_auth=(JIRA_USER, JIRA_TOKEN))


def pretty_print_messages(update):
    import re

    def extract_text_blocks(content):
        if isinstance(content, str):
            return [content]
        elif isinstance(content, list):
            return [b["text"] for b in content if isinstance(b, dict) and "text" in b]
        return []

    def extract_thoughts(content):
        texts = extract_text_blocks(content)
        return [m.strip() for t in texts for m in re.findall(r"<thinking>(.*?)</thinking>", t, re.DOTALL)]

    def extract_response(content):
        texts = extract_text_blocks(content)
        return [m.strip() for t in texts for m in re.findall(r"<response>(.*?)</response>", t, re.DOTALL)]

    # Check for subgraph updates
    agent_name = "supervisor"
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) > 0:
            agent_name = ns[-1].split(":")[0]

    for node_name, node_update in update.items():
        if node_update is None or "messages" not in node_update:
            continue

        messages = convert_to_messages(node_update["messages"])

        for message in messages:
            if isinstance(message, HumanMessage):
                print(f"\n👤 User: {message.content}")

            elif isinstance(message, AIMessage):
                print(f"\n🤖 Agent '{agent_name}' is thinking...")

                thoughts = extract_thoughts(message.content)
                for thought in thoughts:
                    print(f"   💭 {thought}")

                response = extract_response(message.content)
                for resp in response:
                    print(f"   ✅ Final Answer:\n{resp}")

                if message.tool_calls:
                    print(f"   🛠️ Wants to call tool(s):")
                    for tool_call in message.tool_calls:
                        print(f"      • Tool: {tool_call['name']}")
                        print(f"        Args: {json.dumps(tool_call['args'], indent=10)}")

            elif isinstance(message, ToolMessage):
                print(f"\n🔧 Tool '{message.name}' returned:")
                try:
                    content = message.content
                    if isinstance(content, str):
                        parsed = json.loads(content)  # only parse strings
                    elif isinstance(content, (dict, list)):
                        parsed = content              # already structured
                    else:
                        parsed = str(content)         # fallback

                    print(json.dumps(parsed, indent=4, ensure_ascii=False))
                except Exception:
                    print(str(message.content))

            elif isinstance(message, SystemMessage):
                print(f"\n⚙️ System: {message.content}")

    print("\n" + "=" * 100 + "\n")




def init_chat_model(model_key: str = "CLAUDE_MODEL_ID") -> BaseChatModel:
    model_id = os.environ[model_key]
    region = os.environ["AWS_REGION"]

    return ChatBedrock(
        model=model_id,
        region=region,
        model_kwargs={"temperature": 0}
    )




#### TOOLS ---------------------------------------------------------------------------------------------------------------



@tool
def parse_jira_date_tool(input_str: str) -> str:
    """
    Converts a natural language date expression into a Jira-compatible JQL filter clause.

    ```
    Example:
        input_str = "updated in the last 2 months"
        → "updated >= '2025-06-01'"

    Returns:
        JQL string like "created >= '2025-07-01'" or "updated BETWEEN '2025-05-01' AND '2025-06-01'"
    """
    today = datetime.date.today().isoformat()

    system_prompt = f"""
        You are a date conversion assistant for Jira JQL queries.
        Today is: {today}

        Your task is to convert natural language date expressions into a **JQL filter clause**.

        🎯 Your output must be a **valid JQL fragment** like:

        * `created >= '2025-05-01'`
        * `updated <= '2025-06-15'`
        * `created BETWEEN '2025-05-01' AND '2025-06-01'`

        📘 Rules:

        1. **Which field to use**:

        * Use `updated` if the input includes words like: updated, modified, changed
        * Otherwise, use `created`

        2. **Time range handling**:

        * 'last N weeks', 'past N months' → use `>=` and subtract duration from today
        * 'before X', 'until X' → use `<=` and subtract duration from today
        * 'between 2 and 4 weeks ago' → compute both dates and use `BETWEEN` clause
        * Always round to full days

        3. **Duration mapping**:

        * 1 week = 7 days
        * 1 month = 30 days
        * 1 quarter = 90 days
        * 1 year = 365 days

        🧾 Format:

        * Always use ISO format YYYY-MM-DD
        * Wrap dates in single quotes: ✅ `'2025-06-01'`
        * Do NOT output any explanation or justification.
        * Return only the JQL clause as plain text.
        """

    user_prompt = f"Convert to JQL: {input_str}"
    response = call_nova_lite(system_prompt + "\n" + user_prompt)
    return response.strip()

@tool
def resolve_types_and_statuses(
    project_key: Optional[str] = None,
    project_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Resolves available issue types and statuses across Jira projects.

    Args:
        full_user_input: The original user message (can be used for future NLP features).
        project_key: Optional single Jira project key.
        project_names: Optional list of Jira project names.

    Returns:
        A dictionary with:
        {
            "available_issue_types": [...],
            "available_statuses": [...]
        }
    """
    # Resolve project keys
    if project_key:
        keys = [project_key]
    elif project_names:
        all_projects = helpers._list_projects()
        name_to_key = {p["name"].lower(): p["key"] for p in all_projects}
        keys = [name_to_key[name.lower()] for name in project_names if name.lower() in name_to_key]
    else:
        keys = [p["key"] for p in helpers._list_projects()]

    if not keys:
        raise ValueError("Could not resolve any project keys.")

    issue_type_set = set()
    status_set = set()

    for key in keys:
        for issue_type in jira.issue_types_for_project(key):
            issue_type_set.add(issue_type.name)
            for status in getattr(issue_type, "statuses", []):
                status_set.add(status.name)

    return {
        "available_issue_types": sorted(issue_type_set),
        "available_statuses": sorted(status_set)
    }


@tool
def resolve_project_names_tool(human_input: str) -> List[Dict[str, str]]:
    """
    Resolves Jira project keys and names from human-friendly input using LLM.

    Input:
        A string that may contain one or more project names, abbreviations, or partial names.

    Output:
        A list of dictionaries, each containing a matching project key and name.
        Example: [{ "key": "PROJ", "name": "Project" }]

    Uses the Claude LLM (via call_nova_lite) to match fuzzy input to known projects.
    """
    # Load all known projects
    all_projects = helpers._list_projects()
    project_map_str = "\n".join([f"{p['key']}: {p['name']}" for p in all_projects])

    system_prompt = f"""
        You are a Jira assistant that resolves human-friendly project names into exact Jira project keys.

        RULES:
        - A list of valid projects is provided below in the format '<KEY>: <NAME>'.
        - The user may refer to a project by either its name or an approximation of its name (e.g. "Project" or "PROJ").
        - Your task is to find the best matching project keys.
        - Only use keys from the provided list. Never make up a key.
        - If multiple projects match, return them all in ranked order.
        - If nothing matches reasonably, return an empty list.
        - Return only valid JSON like:
        ```json
        {{ "matches": [{{ "key": "PROJ", "name": "Project" }}] }}
        ````

        Available Projects:
        {project_map_str}
    """

    user_prompt = f"Resolve project names: {human_input}"
    response = call_nova_lite(system_prompt + "\n\n" + user_prompt)

    # Try to parse valid JSON from the response
    try:
        match_data = json.loads(response)
        return [{"content": json.dumps({"matches": match_data.get("matches", [])})}]

    except Exception as e:
        raise ValueError(f"Failed to parse response from LLM: {e}\nRaw response:\n{response}")

# @tool
# def resolve_types_and_statuses_jql(
#     full_user_input: str,
#     project_key: Optional[str] = None,
#     project_names: Optional[List[str]] = None,
# ) -> Dict[str, str]:
#     """
#     Generates a JQL filter fragment based on available issue types and statuses
#     across one or more Jira projects, using the full natural language request.

#     Args:
#         full_user_input: The full original user message (not just issue types)
#         project_key: Optional single Jira project key
#         project_names: Optional list of Jira project names

#     Returns:
#         {
#             "content": "{ \"jql_fragment\": \"type IN (...) AND ...\" }"
#         }
#     """
#     # Resolve project keys
#     if project_key:
#         keys = [project_key]
#     elif project_names:
#         all_projects = helpers._list_projects()
#         name_to_key = {p["name"].lower(): p["key"] for p in all_projects}
#         keys = [name_to_key[name.lower()] for name in project_names if name.lower() in name_to_key]
#     else:
#         keys = [p["key"] for p in helpers._list_projects()]

#     if not keys:
#         raise ValueError("Could not resolve any project keys.")

#     # De-duplicate issue types and statuses across all keys
#     seen = set()
#     combined_options = []
#     for key in keys:
#         for it in helpers.jira.issue_types_for_project(key):
#             type_name = it.name
#             status_names = tuple(sorted(s.name for s in getattr(it, "statuses", [])))
#             key_tuple = (type_name, status_names)
#             if key_tuple not in seen:
#                 seen.add(key_tuple)
#                 combined_options.append({
#                     "type": type_name,
#                     "available_statuses": list(status_names)
#                 })

#     system_prompt = f"""
#         You are a Jira assistant. Based on a user request and the provided issue types and statuses, return a minimal JQL fragment.

#         Rules:
#         - Only filter by issue type or status if the user **explicitly** mentions them.
#         - If the user says things like "open", "active", or "not closed" → add: `resolution IN ('Unresolved', 'EMPTY')`
#         - If they say "closed", "resolved", or "completed" → add: `resolution NOT IN ('Unresolved', 'EMPTY')`
#         - If they say "including resolved/closed" → do not add any resolution filter.
#         - If they ask for “all issues” or don’t specify anything → return an empty fragment.
#         - If not specifically asked for updated issues, always use created. 

#         Never add date filters.

#         Format:
#         Return only this JSON:
#         ```json
#         {{ "jql_fragment": "..." }}
#         Available Options:
#         {json.dumps(combined_options, indent=2)}

#         User Request:
#         {full_user_input}
#         """

#     response = call_nova_lite(system_prompt.strip())

#     try:
#         parsed = json.loads(response)
#         if "jql_fragment" not in parsed:
#             raise ValueError("Missing 'jql_fragment' in response")
#         return {"content": json.dumps(parsed)}
#     except Exception as e:
#         raise ValueError(f"Invalid LLM response: {e}\nRaw response:\n{response}")





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


@tool
def think(thought: str) -> str:
    """
    Allows the assistant to express its reasoning step-by-step during complex tasks.
    The input is a single string describing what it is thinking or planning to do.
    """
    return f"[Thinking] {thought}"


tools = [parse_jira_date_tool, check_jql, resolve_project_names_tool, resolve_types_and_statuses, think]

 

def init_supervisor() : 
    SYSTEM_PROMPT = """You are a Jira assistant that converts natural language into correct, minimal JQL.

🔧 Tool Usage:
- `resolve_project_names_tool`: resolve all project names → project keys. Prefer using project keys.
  - ❗If the user doesn't mention any project, assume they want to query **all projects**. Use your tool to pull all projects.
- `resolve_types_and_statuses`: input = list of keys and returns available task types and statuses.
- `parse_jira_date_tool`: convert date expressions.
- Wait for all tool results before assembling JQL.
- `check_jql`: validate final query and return count.
- If needed, ask the user for clarification.

🧠 Assembly Logic:
- Always use: `project IN ('KEY1', 'KEY2')`.
- Add `type IN (...)` or `status IN (...)` **only if user explicitly mentions them**.
- Add priority filters only if explicitly mentioned.

🧹 Formatting:
- Always use single quotes: `'Bug'`, `'Open'`, etc.
- Final JQL should be readable:
```jql

✅ Validation & Output:
Call check_jql.
If error → fix and retry.
If 0 results → try relaxing filters.

If you are concerned or you have open questions like unsure about the JQL, mention it in the "agent_comment" output filed.
Always emphasize if the query returns 0 results. 

Once the query is validated, output this final response exactly:

```json
{
  "jql": "<final assembled query here>",
  "approx_query_results": <count from check_jql>,
  "agent_comment" : "<your comment if applicable>"
}
"""    

    supervisor = create_supervisor(
        model=init_chat_model("NOVA_LITE_MODEL_ID"),
        agents=[],
        prompt=(SYSTEM_PROMPT),
        tools=tools,
        add_handoff_back_messages=True,
        output_mode="full_history",
    )

    return supervisor



def ask_agent_to_generate_jql(human_input: str) -> Optional[Dict[str, Union[str, int]]]:
    """
    Calls the LangGraph supervisor with the given human input.
    Returns the final structured JQL output as a Python dict.
    """
    print("Starting the supervisor stream...")
    print("=" * 80)

    final_chunk = None
    supervisor_instance = init_supervisor().compile()

    for chunk in supervisor_instance.stream({"messages": [{"role": "user", "content": human_input}]}):
        pretty_print_messages(chunk)
        final_chunk = chunk  # Keep track of the last update

    print("\n" + "=" * 80)
    print("Stream finished.")

    if not final_chunk or "supervisor" not in final_chunk:
        return None

    final_messages = final_chunk["supervisor"].get("messages", [])
    final_response = next(
        (msg for msg in reversed(final_messages) if isinstance(msg, AIMessage)),
        None
    )

    if not final_response:
        return None

    # Extract and parse the final response
    try:
        if isinstance(final_response.content, list):
            content = " ".join(
                block.get("text", "")
                for block in (final_response.content or [])
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            content = final_response.content

        import re

        # Extract the first JSON code block from Markdown-style content
        matches = re.findall(r"```json\n(.*?)```", content, re.DOTALL)
        if matches:
            json_block = matches[0].strip()
            parsed = json.loads(json_block)
            return parsed
        else:
            raise ValueError("No JSON block found in agent response.")


    except Exception as e:
        print("❌ Could not parse final response as JSON:", e)
        return None