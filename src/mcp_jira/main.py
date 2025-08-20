import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


import mcp_jira.helpers as helpers


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

load_dotenv(override=True)

JIRA_URL = os.getenv("JIRA_BASE_URL")
JIRA_USER = os.getenv("JIRA_EMAIL","")
JIRA_TOKEN = os.getenv("JIRA_API_TOKEN","")

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
        return helpers._extract_issue_fields(issue, include_comments=True, jira_client=jira)
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
    return helpers._list_projects()


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
    
    return helpers._resolve_project_name(human_input, DEFAULT_CATEGORY)



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
    return helpers._parse_jira_date(input_str)

@mcp.tool
def generate_jql_from_input(user_input: str) -> dict:
    """
    Generate a valid Jira JQL string from natural language input using AI assistance,
    and estimate how many issues match that JQL.

    Parameters:
    - user_input: Free-form text like "all high priority tickets from last month"

    Returns:
    A dictionary with:
    - jql: the generated JQL string with filters applied
    - approx_query_results: estimated number of matching issues
    - comment: AI agent comment or explanation, if any
    """
    result = helpers._generate_jql_from_input(user_input=user_input)

    return {
        "jql": result.get("jql", ""),
        "approx_query_results": result.get("approx_query_results", -1),
        "comment": result.get("comment", "")
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
        return helpers._execute_jql_query(jql)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute JQL: {e}")





@mcp.tool
def summarize_and_analyze_jira_issues(jql: str) -> Dict:
    """
    Summarize Jira issues matching a JQL query, grouped per project.

    Parameters:
    - jql: A Jira Query Language string (e.g., 'project = ASEAT AND resolution = Unresolved')

    Returns:
    A dictionary where each key is a Jira project key, and value includes:
    - project_name: name of the project
    - total_issues: total tickets in that project
    - unresolved_issues: number of unresolved tickets
    - by_priority: all issues grouped by priority
    - by_status: all issues grouped by status
    - by_assignee: all issues grouped by assignee
    - incident_sla_count_by_priority: number of Incident SLA tickets by priority
    - incident_sla_avg_resolution_by_priority: average resolution time in days for Incident SLA by priority
    """
    return helpers._summarize_and_analyze_jql(jql)


# @mcp.tool()
# def advanced_search_issues(
#     projects: list[str] = [],
#     priorities: list[str] = [],
#     resolved: Optional[bool] = None,
#     created_after: str = "",
#     updated_after: str = "",
#     sort_by: str = "created",
#     sort_order: str = "DESC"
# ) -> list[dict]:
#     """
#     Search Jira issues using simplified filters.

#     This tool queries Jira using enhanced search and returns issues
#     filtered by project, priority, resolution state, and date ranges.

#     Parameters:
#     - projects: List of Jira project keys
#     - priorities: List of priorities to include
#     - resolved: True = resolved, False = unresolved, None = all
#     - created_after: Filter by created >= this date (YYYY-MM-DD)
#     - updated_after: Filter by updated >= this date (YYYY-MM-DD)
#     - sort_by: Jira field to sort by
#     - sort_order: ASC or DESC (default DESC)

#     Returns:
#     - A list of issue dicts with basic fields
#     """
#     try:
#         return _advanced_search_issues(
#             projects=projects,
#             priorities=priorities,
#             resolved=resolved,
#             created_after=created_after,
#             updated_after=updated_after,
#             sort_by=sort_by,
#             sort_order=sort_order
#         )
#     except Exception as e:
#         return [{"error": str(e)}]



@mcp.tool()
def get_tickets_insights(ticket_keys: List[str]) -> Dict:
    """
    Generates intelligent insights and structured summaries for a list of Jira tickets.

    For each ticket, the summary includes:
    - A structured header with key metadata (Summary, Status, Priority, Assignee, Created, Updated)
    - A short summary of the ticket's purpose or issue
    - The most recent update (based on comments)
    - A suggested next step

    Parameters:
    - ticket_keys: List of Jira ticket keys (e.g., ["PROJ-123", "PROJ-456"])

    Returns:
    A dictionary where each key is the ticket key and the value is the generated summary string.
    If a ticket fails to summarize, an error message is returned instead.
    """
    return helpers._get_tickets_insights(ticket_keys)




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
    return helpers._approximate_jira_issue_count(jql)


# @mcp.tool
# def analyze_jira_issues(jql: str) -> Dict:
#     """
#     Analyze Jira issues matching a JQL query and return aggregated statistics.

#     Parameters:
#     - jql: Jira Query Language string (e.g., 'project = ASEAT AND resolution IS NOT EMPTY')

#     Returns:
#     A dictionary with the following fields:
#     - jql: the input JQL string
#     - total_issues: total number of matching issues
#     - status_counts: number of issues grouped by status
#     - priority_counts: number of issues grouped by priority
#     - assignee_counts: number of issues grouped by assignee
#     - average_resolution_time_by_priority_for_incident_sla: average resolution time in **days**
#       grouped by priority, but only for issues of type "Incident SLA"

#     Example:
#     {
#         "jql": "...",
#         "total_issues": 134,
#         "status_counts": {"To Do": 42, "In Progress": 54, ...},
#         "priority_counts": {"High": 70, "Medium": 50, ...},
#         "assignee_counts": {"John Doe": 30, "Unassigned": 20, ...},
#         "average_resolution_time_by_priority_for_incident_sla": {"High": 2.8, "Medium": 5.1}
#     }

#     Notes:
#     - Resolution time is calculated as the time between `created` and `resolutiondate`
#     - Only resolved issues contribute to average resolution time
#     - The resolution time by priority is limited to issues of type "Incident SLA"
#     """
#     return _analyze_jira_issues_from_jql(jql)



@mcp.tool
def get_issue_keys(jql: str) -> List[str]:
    """
    Get a list of issue keys matching a given JQL query.

    Parameters:
    - jql: A Jira Query Language string (e.g., 'project = ASEAT AND resolution = Unresolved')

    Returns:
    A list of issue keys that match the query.
    """
    return helpers._get_issue_keys(jql)


@mcp.tool
def extract_issue_fields(ticket_key: str, include_comments: bool = False) -> dict:
    """
    Parameters:
    - ticket_key: Jira issue key (e.g., "PROJ-123")
    - include_comments: Whether to include cleaned comments

    Returns:
    A dictionary with issue metadata and optionally comments.
    """
    try:
        issue = jira.issue(ticket_key)
        return helpers._extract_issue_fields(issue, include_comments=include_comments, jira_client=jira if include_comments else None)
    except Exception as e:
        return {"error": f"Failed to extract fields for {ticket_key}: {e}"}


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8100, path="/mcp")


