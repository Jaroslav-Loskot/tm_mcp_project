import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


import json
import os
import re
import textwrap
from typing import Counter, Dict, List, Optional
from fastapi import HTTPException, APIRouter 
from fastmcp import FastMCP
from jira import JIRA
from dotenv import load_dotenv


from mcp_common.utils.bedrock_wrapper import call_claude
from mcp_jira.helpers import _advanced_search_issues, _analyze_jira_issues_from_jql, _approximate_jira_issue_count, _execute_jql_query, _generate_jql_from_input, _list_projects, _parse_jira_date, _resolve_project_name, _summarize_jira_tickets, _summarize_jql_query, extract_issue_fields


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
    return _list_projects()


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
    Generates a valid JQL query using only project, priority, and resolution status (resolved/unresolved/all).
    """
    raw = _generate_jql_from_input(
        user_input=user_input,
        category_filter=DEFAULT_CATEGORY or None,
        exclude_projects=EXCLUDED_KEYS or None
    )
    
    return {
        "jql": raw["jql"]
    }

@mcp.tool
def execute_jql_query(jql: str) -> List[Dict]:
    """
    Executes a JQL query and returns all matching issues using Jira Cloud's enhanced search via automatic pagination.

    Returns the following fields:
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
    """
    try:
        return _execute_jql_query(jql)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute JQL: {e}")




@mcp.tool
def summarize_jql_query(jql: str) -> Dict:
    """
    Executes a JQL query and returns a summary including:
    - total issue count
    - number of issues per status
    """
    try:
        return _summarize_jql_query(jql)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to summarize JQL: {e}")


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
def advanced_search_issues(
    projects: list[str] = [],
    priorities: list[str] = [],
    resolved: Optional[bool] = None,
    created_after: str = "",
    updated_after: str = "",
    sort_by: str = "created",
    sort_order: str = "DESC"
) -> list[dict]:
    """
    Search Jira issues using simplified filters.

    This tool queries Jira using enhanced search and returns issues
    filtered by project, priority, resolution state, and date ranges.

    Parameters:
    - projects: List of Jira project keys
    - priorities: List of priorities to include
    - resolved: True = resolved, False = unresolved, None = all
    - created_after: Filter by created >= this date (YYYY-MM-DD)
    - updated_after: Filter by updated >= this date (YYYY-MM-DD)
    - sort_by: Jira field to sort by
    - sort_order: ASC or DESC (default DESC)

    Returns:
    - A list of issue dicts with basic fields
    """
    try:
        return _advanced_search_issues(
            projects=projects,
            priorities=priorities,
            resolved=resolved,
            created_after=created_after,
            updated_after=updated_after,
            sort_by=sort_by,
            sort_order=sort_order
        )
    except Exception as e:
        return [{"error": str(e)}]



@mcp.tool()
def summarize_jira_tickets(ticket_keys: List[str]) -> Dict:
    """
    Summarizes a list of Jira tickets with structured headers and intelligent insights.

    For each ticket, returns:
    - A structured header with Summary, Status, Priority, Assignee, Created, and Updated
    - A short summary of the ticket content
    - Latest update
    - Suggested next step

    Parameters:
    - ticket_keys: List of Jira ticket keys (e.g., ["PROJ-123", "PROJ-456"])

    Returns:
    A dictionary where each key is a ticket key and the value is the generated summary string.
    """
    try:
        return _summarize_jira_tickets(ticket_keys)
    except Exception as e:
        return {"error": f"Failed to summarize Jira tickets: {str(e)}"}


@mcp.tool()
def approximate_jira_issue_count(jql: str) -> Dict:
    """
    Executes a JQL query and returns an approximate count of matching Jira issues.

    Parameters:
    - jql: A valid Jira Query Language string (e.g., 'project = PROJ AND status = "To Do"')

    Returns:
    - A dictionary with:
      {
        "jql": "<original input>",
        "approximate_count": <int>
      }

    Or if there's an error:
      {
        "error": "...",
        "jql": "<original input>"
      }
    """
    return _approximate_jira_issue_count(jql)


@mcp.tool
def analyze_jira_issues(jql: str) -> Dict:
    """
    Analyze Jira issues matching a JQL query and return aggregated statistics.

    Parameters:
    - jql: Jira Query Language string (e.g., 'project = ASEAT AND resolution IS NOT EMPTY')

    Returns:
    A dictionary with the following fields:
    - jql: the input JQL string
    - total_issues: total number of matching issues
    - status_counts: number of issues grouped by status
    - priority_counts: number of issues grouped by priority
    - assignee_counts: number of issues grouped by assignee
    - average_resolution_time_by_priority_for_incident_sla: average resolution time in **days**
      grouped by priority, but only for issues of type "Incident SLA"

    Example:
    {
        "jql": "...",
        "total_issues": 134,
        "status_counts": {"To Do": 42, "In Progress": 54, ...},
        "priority_counts": {"High": 70, "Medium": 50, ...},
        "assignee_counts": {"John Doe": 30, "Unassigned": 20, ...},
        "average_resolution_time_by_priority_for_incident_sla": {"High": 2.8, "Medium": 5.1}
    }

    Notes:
    - Resolution time is calculated as the time between `created` and `resolutiondate`
    - Only resolved issues contribute to average resolution time
    - The resolution time by priority is limited to issues of type "Incident SLA"
    """
    return _analyze_jira_issues_from_jql(jql)


def summarize_jira_tickets_sequential(ticket_keys: List[str], delay: float = 1.5) -> Dict:
    """
    Summarize multiple Jira tickets one-by-one with a delay between LLM calls to avoid overload.

    Parameters:
    - ticket_keys: List of Jira ticket keys (e.g., ["ASEAT-123", "ASUCIT-456"])
    - delay: Optional delay in seconds between calls (default: 1.5s)

    Returns:
    A dictionary mapping ticket keys to summaries, or an error dictionary if a problem occurred.

    Each summary includes:
    - Summary
    - Status
    - Priority
    - Assignee
    - Created
    - Updated
    - Short description of issue
    - Latest comment or update
    - Suggested next step

    Example:
    {
        "ASEAT-123": "Summary: ...\nStatus: ...\nPriority: ...\n...",
        ...
    }

    Notes:
    - Each ticket is summarized individually via LLM with a short wait between them.
    - Invalid or failed tickets are skipped.
    - This is useful when you want stable results across many tickets.
    """
    return summarize_jira_tickets_sequential(ticket_keys, delay=delay)


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8100)  # run 'fastmcp run main.py --transport sse --port 8100'


