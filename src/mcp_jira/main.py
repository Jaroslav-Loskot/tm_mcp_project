import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


import json
import os
import re
import textwrap
from typing import Dict, List, Optional
from fastapi import HTTPException, APIRouter 
from fastmcp import FastMCP
from jira import JIRA
from dotenv import load_dotenv

from mcp_common.utils.bedrock_wrapper import call_claude
from mcp_jira.helpers import _generate_jql_from_input, _parse_jira_date, _resolve_project_name, extract_issue_fields


load_dotenv(override=True)

JIRA_URL = os.getenv("JIRA_BASE_URL")
JIRA_USER = os.getenv("JIRA_EMAIL")
JIRA_TOKEN = os.getenv("JIRA_API_TOKEN")

DEFAULT_CATEGORY = os.getenv("DEFAULT_PROJECT_CATEGORY", "")
EXCLUDED_KEYS = [k.strip() for k in os.getenv("EXCLUDED_PROJECT_KEYS", "").split(",") if k.strip()]

jira = JIRA(server=JIRA_URL, basic_auth=(JIRA_USER, JIRA_TOKEN))



mcp = FastMCP("Jira MCP Server", auth=None)


@mcp.tool()
def search_issues(jql: str, max_results: int = 5) -> list[dict]:
    """
    Search Jira issues using a JQL query.
    Returns a list of issue keys and summaries.
    """
    issues = jira.search_issues(jql, maxResults=max_results)
    return [{"key": issue.key, "summary": issue.fields.summary} for issue in issues]



@mcp.tool()
def get_issue(key: str) -> dict:
    """
    Retrieve full details for a Jira issue by key.
    """
    try:
        issue = jira.issue(key)
        return extract_issue_fields(issue, include_comments=True, jira_client=jira)
    except Exception as e:
        return {"error": str(e)}





@mcp.tool()
def get_available_issue_statuses(key: str) -> list[str]:
    """
    Get the list of available statuses the given issue can transition to.
    This returns the display names of valid transitions for the issue's current workflow state.
    """
    try:
        transitions = jira.transitions(key)
        return [t['to']['name'] for t in transitions]
    except Exception as e:
        return [f"Error: {str(e)}"]

@mcp.tool()
def list_projects() -> list[dict]:
    """
    List Jira projects visible to the current user.

    """
    try:
        projects = jira.projects()
        filtered_projects = []

        for p in projects:
            # Exclude if key is in EXCLUDED_KEYS
            if p.key in EXCLUDED_KEYS:
                continue

            # Filter by category if specified
            category = getattr(p, 'projectCategory', None)
            if DEFAULT_CATEGORY:
                if category and getattr(category, 'name', '') == DEFAULT_CATEGORY:
                    filtered_projects.append({
                        "key": p.key,
                        "name": p.name,
                        "category": category.name
                    })
            else:
                filtered_projects.append({
                    "key": p.key,
                    "name": p.name,
                    "category": getattr(category, 'name', '') if category else None
                })

        return filtered_projects
    except Exception as e:
        return [{"error": str(e)}]



@mcp.tool
def get_all_issue_types() -> List[str]:
    """
    Fetches all globally available issue types (task types) from Jira.
    Returns a de-duplicated, sorted list of issue type names.
    """
    try:
        global_issue_types = jira.issue_types()
        issue_type_set = {it.name for it in global_issue_types}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch global issue types: {e}")

    return sorted(issue_type_set)




@mcp.tool()
def resolve_project_name(human_input: str) -> List[Dict[str, str]]:
    """
    Resolve a Jira project name from human-friendly input.
    Fetches available Jira projects and chooses the best match.

    Parameters:
    - human_input: Human-friendly name of the project (e.g., 'website revamp').

    Returns:
    - The matching Jira project name (e.g., 'Website Comapny'), or raises error if not found or invalid.
    """

    return _resolve_project_name(human_input, DEFAULT_CATEGORY)



@mcp.tool()
def get_all_statuses_for_project(project_key: str) -> list[str]:
    """
    Get all available issue statuses in a Jira project.

    This function fetches all issue types configured for the project, then inspects their metadata
    to extract all allowed status values (e.g., To Do, In Progress, Done).

    Parameters:
    - project_key: The key of the Jira project (e.g., "APP", "SEC").

    Returns:
    - A sorted list of unique status display names used in the project's workflows.
      If an error occurs, returns a list with a single error message.
    """
    try:
        project = jira.project(project_key)
        issue_types = project.issueTypes
        statuses = set()

        for issue_type in issue_types:
            try:
                meta = jira.createmeta(
                    projectKeys=project_key,
                    issuetypeNames=issue_type.name,
                    expand="projects.issuetypes.fields"
                )
                for project_meta in meta.get('projects', []):
                    for itype in project_meta.get('issuetypes', []):
                        if itype.get('name') == issue_type.name:
                            fields = itype.get('fields', {})
                            status_field = fields.get('status')
                            if status_field and 'allowedValues' in status_field:
                                for s in status_field['allowedValues']:
                                    statuses.add(s['name'])
            except Exception:
                continue  # Continue with next issue type if this one fails

        return sorted(statuses)

    except Exception as e:
        return [f"Error: {str(e)}"]



@mcp.tool()
def parse_jira_date(input_str: str) -> str:
    """
    Parses flexible date input into Jira-compatible YYYY-MM-DD format.

    Supports:
    - Relative dates: -1w, -3d, -2m, -1y
    - Keywords: today, yesterday
    - Absolute dates: 2025-07-01, 07/01/2025, 1 Jul 2025, July 1, 2025, etc.

    Parameters:
    - input_str: Human-readable date string.

    Returns:
    - A formatted date string in YYYY-MM-DD format.
    """
    return _parse_jira_date(input_str)

@mcp.tool
def generate_jql_from_input(user_input: str) -> dict:
    """
    Generates a JSON object containing:
    - a valid JQL query using only project, priority, and resolution status (resolved/unresolved/all)
    - an optional max_results value if the user requests a limit
    """
    result = _generate_jql_from_input(
        user_input="open high priority tickets",
        category_filter=DEFAULT_CATEGORY or None,
        exclude_projects=EXCLUDED_KEYS or None
)


@mcp.tool
def execute_jql_query(jql: str) -> List[Dict]:
    """
    Executes a JQL query and returns up to 100 matching issues (paginated internally).
    
    Fields returned per issue:
    - key
    - summary
    - issue_type
    - status
    - assignee
    - created
    - updated
    - project
    - resolution
    - priority

    Parameters:
    - jql: The Jira Query Language string.

    Returns:
    - List of up to 100 issues in compact format.
    """
    try:
        start_at = 0
        page_size = 50  # can be tuned if needed
        total_collected = 0
        max_limit = 100
        results = []

        while total_collected < max_limit:
            remaining = max_limit - total_collected
            page = jira.search_issues(
                jql,
                startAt=start_at,
                maxResults=min(page_size, remaining),
                expand="names"
            )

            for issue in page:
                fields = issue.fields
                results.append({
                    "key": issue.key,
                    "summary": fields.summary,
                    "issue_type": getattr(fields.issuetype, "name", None),
                    "status": getattr(fields.status, "name", None),
                    "assignee": getattr(fields.assignee, "displayName", None) if fields.assignee else None,
                    "created": fields.created,
                    "updated": fields.updated,
                    "project": getattr(fields.project, "key", None),
                    "resolution": getattr(fields.resolution, "name", None) if fields.resolution else None,
                    "priority": getattr(fields.priority, "name", None) if fields.priority else None,
                })
                total_collected += 1

                if total_collected >= max_limit:
                    break

            if len(page) < page_size:
                break  # no more pages

            start_at += page_size

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute JQL: {e}")




@mcp.tool()
def get_issue_with_comments(key: str) -> dict:
    """
    Retrieve full Jira issue info with cleaned comments, priority, and task type.
    """
    try:
        issue = jira.issue(key)
        return extract_issue_fields(issue, include_comments=True, jira_client=jira)
    except Exception as e:
        return {"error": str(e)}



@mcp.tool()
def search_advanced_issues(
    projects: list[str] = [],
    statuses: list[str] = [],
    priorities: list[str] = [],
    assignees: list[str] = [],
    created_after: str = "",
    updated_after: str = "",
    max_results: int = 10,
    sort_by: str = "created",
    sort_order: str = "DESC"
) -> list[dict]:
    """
    Search Jira issues using multiple filters:
    - Accepts lists for projects, statuses, priorities, assignees
    - Accepts created/updated date ranges in 'YYYY-MM-DD'
    - Supports sorting by any Jira field

    Returns a list of matching issues with key, summary, status, assignee, priority, created, updated.
    """
    jql_parts = []

    if projects:
        quoted = [f'"{p}"' for p in projects]
        jql_parts.append(f'project IN ({", ".join(quoted)})')
    if statuses:
        quoted = [f'"{s}"' for s in statuses]
        jql_parts.append(f'status IN ({", ".join(quoted)})')
    if priorities:
        quoted = [f'"{p}"' for p in priorities]
        jql_parts.append(f'priority IN ({", ".join(quoted)})')
    if assignees:
        quoted = [f'"{a}"' for a in assignees]
        jql_parts.append(f'assignee IN ({", ".join(quoted)})')
    if created_after:
        jql_parts.append(f'created >= "{created_after}"')
    if updated_after:
        jql_parts.append(f'updated >= "{updated_after}"')

    # Validate sort order
    order = sort_order.upper()
    if order not in ["ASC", "DESC"]:
        order = "DESC"

    jql = " AND ".join(jql_parts) if jql_parts else ""
    jql += f' ORDER BY {sort_by} {order}'

    try:
        issues = jira.search_issues(jql, maxResults=max_results)
        return [
            {
                extract_issue_fields(issue)
            }
            for issue in issues
        ]
    except Exception as e:
        return [{"error": str(e), "jql": jql}]



@mcp.tool
def summarize_jira_tickets(ticket_keys: List[str]) -> Dict:
    """
    Fetches key details and comments for each Jira ticket, then summarizes them using LLM.

    Returns:
    - executive_summary: high-level overview of all tickets
    - ticket_summaries: mapping of ticket key to its summary
    """
    try:
        ticket_data = []

        for key in ticket_keys:
            try:
                issue = jira.issue(key, expand="renderedFields")
                comments = jira.comments(key)

                summary = issue.fields.summary
                status = issue.fields.status.name
                priority = getattr(issue.fields.priority, "name", None)
                assignee = getattr(issue.fields.assignee, "displayName", None)
                created = issue.fields.created
                updated = issue.fields.updated
                description = issue.fields.description or ""

                comment_text = "\n".join(
                    f"{c.author.displayName}: {c.body}" for c in comments
                )

                ticket_data.append({
                    "key": key,
                    "summary": summary,
                    "status": status,
                    "priority": priority,
                    "assignee": assignee,
                    "created": created,
                    "updated": updated,
                    "description": description,
                    "comments": comment_text
                })
            except Exception as e:
                ticket_data.append({
                    "key": key,
                    "error": f"Failed to fetch ticket: {str(e)}"
                })

        # Prepare input for LLM
        formatted_input = "\n\n".join([
            textwrap.dedent(f"""
            Ticket {t['key']}:
            Summary: {t.get('summary', '')}
            Status: {t.get('status', '')}
            Priority: {t.get('priority', '')}
            Assignee: {t.get('assignee', '')}
            Created: {t.get('created', '')}
            Updated: {t.get('updated', '')}

            Description:
            {t.get('description', '')}

            Comments:
            {t.get('comments', '')}
            """).strip()
            for t in ticket_data if "error" not in t
        ])

        system_prompt = (
            "You are an expert Jira analyst. The user will provide raw issue data including "
            "summaries, statuses, priorities, and comments.\n\n"
            "Your task:\n"
            "1. Write a high-level 'executive_summary' that captures important patterns, updates, progress, blockers, or risks across all tickets.\n"
            "2. For each ticket, return a concise summary (2–4 sentences) under 'ticket_summaries' keyed by ticket ID.\n\n"
            "FORMAT:\n"
            "{\n"
            "  \"executive_summary\": \"...\",\n"
            "  \"ticket_summaries\": {\n"
            "    \"TICKET-123\": \"summary...\",\n"
            "    \"TICKET-456\": \"summary...\"\n"
            "  }\n"
            "}\n"
            "- Return ONLY valid JSON. Do not include markdown, comments, or explanation.\n"
            "- Ensure all JSON is syntactically correct (no trailing commas, correct quoting, etc.)."
        )

        user_input = f"""
        Here is the data for multiple Jira tickets:

        {formatted_input}
        """

        response = call_claude(system_prompt=system_prompt, user_input=user_input)
        fenced = re.search(r"\{.*\}", response, re.DOTALL)
        response_json = fenced.group(0) if fenced else response

        return json.loads(response_json)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to summarize Jira tickets: {e}")
    



if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8100)  # run 'fastmcp run main.py --transport sse --port 8100'