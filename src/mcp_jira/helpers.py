from collections import defaultdict
from dataclasses import dataclass
import json
import logging
import textwrap
import time
from typing import Counter, Dict, List, Optional
from difflib import get_close_matches
from fastapi import HTTPException
from datetime import datetime, timedelta
import re
from jira import JIRA  # Atlassian Python client
import os
from dotenv import load_dotenv
import re
from datetime import datetime, timezone

import pytz
from mcp_common.utils.bedrock_wrapper import call_claude, call_nova_lite
from mcp_jira.main import extract_issue_fields
# from mcp_jira.helpers import get_clean_comments_from_issue


load_dotenv(override=True)


JIRA_URL = os.getenv("JIRA_BASE_URL")
JIRA_USER = os.getenv("JIRA_EMAIL")
JIRA_TOKEN = os.getenv("JIRA_API_TOKEN")

DEFAULT_CATEGORY = os.getenv("DEFAULT_PROJECT_CATEGORY", "")
EXCLUDED_KEYS = [k.strip() for k in os.getenv("EXCLUDED_PROJECT_KEYS", "").split(",") if k.strip()]

jira = JIRA(server=JIRA_URL, basic_auth=(JIRA_USER, JIRA_TOKEN))



def _approximate_jira_issue_count(jql: str) -> Dict:
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


def get_clean_comments_from_issue(jira, issue) -> list[dict]:
    """
    Returns raw comments from a Jira issue, including author and created timestamp.
    No filtering or cleaning is applied to avoid losing useful content.
    """
    try:
        comments = jira.comments(issue)
        return [
            {
                "author": c.author.displayName,
                "created": c.created,
                "text": c.body  # raw, full comment content
            }
            for c in comments
        ]
    except Exception as e:
        return [{"error": str(e)}]


def _extract_issue_fields(issue, include_comments=False, jira_client=None):
    data = {
        "key": issue.key,
        "summary": issue.fields.summary,
        "status": issue.fields.status.name,
        "priority": issue.fields.priority.name if issue.fields.priority else None,
        "assignee": issue.fields.assignee.displayName if issue.fields.assignee else None,
        "reporter": issue.fields.reporter.displayName if issue.fields.reporter else None,
        "created": issue.fields.created,
        "updated": issue.fields.updated,
        "task_type": issue.fields.issuetype.name if issue.fields.issuetype else None,
    }

    if include_comments and jira_client is not None:
        data["comments"] = get_clean_comments_from_issue(jira_client, issue)

    return data


def _parse_jira_date(input_str: str) -> str:
    """
    Parses flexible date inputs into Jira-compatible YYYY-MM-DD format.

    Supports:
    - Relative keywords: today, yesterday, last week, last month, this year, etc.
    - Shorthands: -1w, -3d, -2m, -1y
    - Date strings: 2025-07-01, 07/01/2025, 1 Jul 2025, July 1, 2025, etc.
    """
    input_str = input_str.strip().lower()
    now = datetime.now(timezone.utc)

    # Handle natural keywords
    if input_str in ["today", "now"]:
        return now.strftime("%Y-%m-%d")
    if input_str == "yesterday":
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")

    # Handle relative phrases
    if input_str == "last week":
        last_week_start = now - timedelta(days=now.weekday() + 7)
        return last_week_start.strftime("%Y-%m-%d")
    if input_str == "this week":
        this_week_start = now - timedelta(days=now.weekday())
        return this_week_start.strftime("%Y-%m-%d")

    if input_str == "last month":
        year = now.year
        month = now.month - 1
        if month == 0:
            month = 12
            year -= 1
        return datetime(year, month, 1).strftime("%Y-%m-%d")

    if input_str == "this month":
        return datetime(now.year, now.month, 1).strftime("%Y-%m-%d")

    if input_str == "last year":
        return datetime(now.year - 1, 1, 1).strftime("%Y-%m-%d")

    if input_str == "this year":
        return datetime(now.year, 1, 1).strftime("%Y-%m-%d")

    # Handle shorthands like -3d, -2w, etc.
    match = re.match(r"^-(\d+)([dwmy])$", input_str)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        delta = timedelta(days={
            "d": amount,
            "w": 7 * amount,
            "m": 30 * amount,
            "y": 365 * amount
        }[unit])
        return (now - delta).strftime("%Y-%m-%d")

    # Try parsing flexible date formats
    known_formats = [
        "%Y-%m-%d",         # 2025-07-01
        "%d/%m/%Y",         # 01/07/2025 (EU)
        "%m/%d/%Y",         # 07/01/2025 (US)
        "%d-%m-%Y",         # 01-07-2025
        "%m-%d-%Y",         # 07-01-2025
        "%d %b %Y",         # 1 Jul 2025
        "%d %B %Y",         # 1 July 2025
        "%B %d, %Y",        # July 1, 2025
        "%b %d, %Y",        # Jul 1, 2025
    ]

    for fmt in known_formats:
        try:
            parsed = datetime.strptime(input_str, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    raise ValueError(f"Unrecognized date format: '{input_str}'")


def find_existing_issue(jira: JIRA, project_key: str) -> Optional[str]:
    """
    Tries to find an existing issue in the form PROJECT_KEY-1 through PROJECT_KEY-5.

    Parameters:
    - jira: An instance of the authenticated JIRA client.
    - project_key: The Jira project key (e.g., 'DELPROJ').

    Returns:
    - The first valid issue key found (e.g., 'DELPROJ-2'), or None if none exist.
    """
    for i in range(1, 6):
        issue_key = f"{project_key}-{i}"
        try:
            jira.issue(issue_key)
            return issue_key
        except Exception:
            time.sleep(0.1)  # 100ms pause

    return None


def get_all_jira_statuses() -> List[str]:
    """
    Fetches all available Jira statuses.

    Returns:
        A list of status names (e.g. ['Open', 'In Progress', 'Resolved', 'Closed'])
    """
    try:
        statuses = jira.statuses()
        return [s.name for s in statuses]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Jira statuses: {e}")


def get_all_jira_priorities() -> list[str]:
    """
    Fetches all available Jira priorities.
    Returns:
        A list of priority names (e.g. ['Highest', 'High', 'Medium', 'Low', 'Lowest'])
    """
    try:
        priorities = jira.priorities()
        return [p.name for p in priorities]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Jira priorities: {e}")
    
def _list_projects() -> list[dict]:
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


def _generate_jql_from_input(
    user_input: str,
    category_filter: Optional[str] = os.getenv("DEFAULT_PROJECT_CATEGORY"),
    exclude_projects: Optional[List[str]] = os.getenv("EXCLUDED_PROJECT_KEYS", "").split(",")
) -> dict:
    """
    Converts natural language input into JQL using Claude and estimates the result size.

    Args:
        user_input: Free-form user query like "all high priority tickets for Erste".
        category_filter: Optional Jira project category to include only certain projects.
        exclude_projects: Optional list of project keys to exclude.

    Returns:
        dict with 'jql' and 'approx_query_results'.
    """
    all_projects = _list_projects()

    # Filter by category if provided
    if category_filter:
        all_projects = [p for p in all_projects if p.get("category", "").lower() == category_filter.lower()]

    # Filter out excluded projects
    exclude_projects = [p.strip() for p in exclude_projects if p.strip()]
    allowed_projects = [p for p in all_projects if p["key"] not in exclude_projects]

    if not allowed_projects:
        raise ValueError("No allowed projects after applying filters.")

    allowed_project_keys = [p["key"] for p in allowed_projects]
    allowed_priorities = get_all_jira_priorities()

    project_map_str = "\n".join([f"{p['key']}: {p['name']}" for p in allowed_projects])

    system_prompt = (
        "You are a Jira assistant that converts natural language requests into structured JSON "
        "for querying Jira issues.\n\n"
        "RULES:\n"
        "- A list of projects is provided in the format '<KEY>: <NAME>'.\n"
        "- The user may refer to a project by either its key or name. You must resolve it to a key.\n"
        "- Only use the provided project keys and priorities.\n"
        "- For issue status, DO NOT use raw status names like 'In Progress'.\n"
        "- Instead, infer resolution as follows:\n"
        "   - If the user explicitly asks for **open**, **unresolved**, or **incomplete** issues, use:\n"
        "       resolution in (Unresolved, EMPTY)\n"
        "   - If the user explicitly asks for **closed**, **completed**, or **resolved** issues, use:\n"
        "       resolution not in (Unresolved, EMPTY)\n"
        "   - If the user refers to **all issues** or only filters by project, priority, or date, DO NOT include a resolution clause.\n"
        "- If a priority is mentioned, include it. Otherwise, omit it.\n"
        "- For date filters:\n"
        "   - Use **updated >=** only if the user explicitly mentions 'recently updated', 'changed', or 'modified'.\n"
        "   - Otherwise, default to **created >=**.\n"
        "   - Only use durations ending in 'd' (days) or 'w' (weeks). If the user mentions:\n"
        "       - months: convert to days using 30 days per month\n"
        "       - quarters: convert to days using 90 days per quarter\n"
        "       - years: convert to days using 365 days per year\n"
        "- Return ONLY this JSON structure: { \"jql\": \"...\" }\n\n"
        "Examples:\n"
        "- 'open issues from last week' → created >= -1w AND resolution in (Unresolved, EMPTY)\n"
        "- 'closed high priority bugs for UCB' → project = UCB AND priority = High AND resolution not in (Unresolved, EMPTY)\n"
        "- 'all PostFinance tickets' → project = ASPFI\n"
        "- 'tickets from past 3 months' → created >= -90d\n"
        "- 'recently updated issues from last 2 weeks' → updated >= -2w\n"
    )

    user_prompt = f"""User Query:
    {user_input}

    Available Projects:
    {project_map_str}

    Allowed Priorities:
    {', '.join(allowed_priorities)}"""

    response = call_nova_lite(system_prompt + "\n" + user_prompt)

    match = re.search(r'\{.*\}', response, re.DOTALL)
    result = json.loads(match.group(0)) if match else json.loads(response)

    generated_jql = result["jql"]

    project_in_clause = f"project IN ({', '.join(f'\'{k}\'' for k in allowed_project_keys)})"
    project_not_in_clause = f" AND project NOT IN ({', '.join(f'\'{p}\'' for p in exclude_projects)})" if exclude_projects else ""

    if "project" not in generated_jql.lower():
        full_jql = f"{project_in_clause}{project_not_in_clause} AND ({generated_jql})"
    else:
        full_jql = generated_jql

    approx = _approximate_jira_issue_count(full_jql)
    approx_count = approx.get("approximate_count", -1)

    return {
        "jql": full_jql,
        "approx_query_results": approx_count
    }


def _summarize_jira_issues(jql: str) -> Dict:
    """
    Executes a JQL query using enhanced search and returns a detailed summary:
    - total issue count
    - total unresolved issue count
    - global unresolved grouped by priority/status/assignee
    - per-project:
        - total issues
        - unresolved by priority
        - unresolved by status
        - unresolved by assignee
        - unresolved ratio
    - top 5 projects with most unresolved
    """
    try:
        all_issues = []
        next_page_token = None
        max_results = 100

        while True:
            issues = jira.enhanced_search_issues(
                jql_str=jql,
                nextPageToken=next_page_token,
                maxResults=max_results,
                fields=["project", "status", "priority", "assignee", "resolution"],
                use_post=True
            )
            if not issues:
                break

            all_issues.extend(issues)
            next_page_token = getattr(issues, "nextPageToken", None)
            if not next_page_token:
                break

        total_issues = len(all_issues)
        unresolved_issues = [i for i in all_issues if not getattr(i.fields, "resolution", None)]

        global_unresolved_by_priority = Counter()
        global_unresolved_by_status = Counter()
        global_unresolved_by_assignee = Counter()

        per_project_data = defaultdict(lambda: {
            "total": 0,
            "unresolved_by_priority": Counter(),
            "unresolved_by_status": Counter(),
            "unresolved_by_assignee": Counter()
        })

        for issue in all_issues:
            project_key = getattr(issue.fields.project, "key", "UNKNOWN")
            per_project_data[project_key]["total"] += 1

        for issue in unresolved_issues:
            fields = issue.fields
            project_key = getattr(fields.project, "key", "UNKNOWN")
            priority = getattr(fields.priority, "name", "None")
            status = getattr(fields.status, "name", "Unknown")
            assignee = getattr(fields.assignee, "displayName", "Unassigned")

            per_project_data[project_key]["unresolved_by_priority"][priority] += 1
            per_project_data[project_key]["unresolved_by_status"][status] += 1
            per_project_data[project_key]["unresolved_by_assignee"][assignee] += 1

            global_unresolved_by_priority[priority] += 1
            global_unresolved_by_status[status] += 1
            global_unresolved_by_assignee[assignee] += 1

        # Add unresolved ratio and convert counters to dicts
        for project_key, data in per_project_data.items():
            total = data["total"]
            unresolved_total = sum(data["unresolved_by_priority"].values())
            data["unresolved_ratio"] = round(unresolved_total / total, 2) if total else 0
            data["unresolved_by_priority"] = dict(data["unresolved_by_priority"])
            data["unresolved_by_status"] = dict(data["unresolved_by_status"])
            data["unresolved_by_assignee"] = dict(data["unresolved_by_assignee"])

        return {
            "total_issues": total_issues,
            "total_unresolved_issues": len(unresolved_issues),
            "unresolved_global_by_priority": dict(global_unresolved_by_priority),
            "unresolved_global_by_status": dict(global_unresolved_by_status),
            "unresolved_global_by_assignee": dict(global_unresolved_by_assignee),
            "per_project": per_project_data,
            "generated_at": datetime.utcnow().replace(tzinfo=pytz.UTC).isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to summarize JQL: {e}")





def _execute_jql_query(jql: str) -> List[Dict]:
    """
    Executes a JQL query and returns all matching issues using Jira Cloud's enhanced search with pagination.

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
        all_issues = []
        next_page_token = None

        while True:
            response = jira.enhanced_search_issues(
                jql_str=jql,
                nextPageToken=next_page_token,
                maxResults=100,
                fields=[
                    "summary", "issuetype", "status", "assignee",
                    "created", "updated", "project", "resolution", "priority"
                ],
                use_post=True,
                json_result=False
            )

            for issue in response:
                fields = issue.fields
                all_issues.append({
                    "key": issue.key,
                    "summary": getattr(fields, "summary", None),
                    "issue_type": getattr(getattr(fields, "issuetype", None), "name", None),
                    "status": getattr(getattr(fields, "status", None), "name", None),
                    "assignee": getattr(getattr(fields, "assignee", None), "displayName", None),
                    "created": getattr(fields, "created", None),
                    "updated": getattr(fields, "updated", None),
                    "project": getattr(getattr(fields, "project", None), "key", None),
                    "resolution": getattr(getattr(fields, "resolution", None), "name", None),
                    "priority": getattr(getattr(fields, "priority", None), "name", None),
                })

            # Pagination check
            next_page_token = getattr(response, "nextPageToken", None)
            if not next_page_token:
                break

        return all_issues

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute JQL: {e}")




def _resolve_project_name(human_input: str, category_filter: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Resolves the most relevant Jira projects from human input.

    Args:
        human_input: Free-form user input like "akbank" or "tipsport"
        category_filter: Optional category name (e.g., "Application Support")

    Returns:
        A list of up to 5 dicts like [{'key': 'UCB', 'name': 'Unicredit Italy'}, ...]
    """
    try:
        projects = jira.projects()
        all_projects = []

        for p in projects:
            category = getattr(p, 'projectCategory', None)
            all_projects.append({
                "key": p.key,
                "name": p.name,
                "category": getattr(category, 'name', '') if category else ''
            })
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Jira projects: {e}")

    # Apply optional category filtering
    if category_filter:
        filtered_projects = [
            p for p in all_projects
            if p["category"].lower() == category_filter.lower()
        ]
    else:
        filtered_projects = all_projects

    if not filtered_projects:
        raise ValueError(f"No projects found in category '{category_filter}'.")

    system_prompt = (
        "You are a Jira assistant helping users match human-friendly descriptions to existing Jira project names.\n\n"
        "RULES:\n"
        "- Select the TOP 5 most relevant project names from the provided list.\n"
        "- Output only a JSON array of the selected project names, ordered by relevance.\n"
        "- Do NOT return any explanations or markdown.\n"
    )

    formatted_projects = "\n".join(f"- {p['name']}" for p in filtered_projects)

    user_message = f"""
    User input:
    {human_input}

    Available Jira projects:
    {formatted_projects}

    Expected output format:
    ["Unicredit Italy", "Core Banking", "Security Improvements"]
    """

    response = call_nova_lite(system_prompt + "\n\n" + user_message).strip()

    try:
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        json_str = match.group(0) if match else response
        selected_names = json.loads(json_str)
    except Exception as e:
        raise ValueError(f"❌ Failed to parse Nova's response: {e}\n\nRaw response:\n{response}")

    if not isinstance(selected_names, list) or not all(isinstance(n, str) for n in selected_names):
        raise ValueError(f"❌ Unexpected format returned from Nova: {selected_names}")

    # Match names back to project dicts
    name_to_project = {p['name']: p for p in filtered_projects}
    selected_projects = [name_to_project[name] for name in selected_names if name in name_to_project]

    return selected_projects[:5]




def _advanced_search_issues(
    projects: list[str] = [],
    priorities: list[str] = [],
    resolved: Optional[bool] = None,
    created_after: str = "",
    updated_after: str = "",
    sort_by: str = "created",
    sort_order: str = "DESC"
) -> list[dict]:
    """
    Search Jira issues using enhanced_search_issues and simplified filters.
    """
    jql_parts = []

    if projects:
        jql_parts.append(f'project IN ({", ".join(f'"{p}"' for p in projects)})')

    if priorities:
        jql_parts.append(f'priority IN ({", ".join(f'"{p}"' for p in priorities)})')

    if resolved is True:
        jql_parts.append('resolution NOT IN (EMPTY, Unresolved)')
    elif resolved is False:
        jql_parts.append('resolution IN (EMPTY, Unresolved)')

    if created_after:
        jql_parts.append(f'created >= "{created_after}"')

    if updated_after:
        jql_parts.append(f'updated >= "{updated_after}"')

    sort_order = sort_order.upper()
    if sort_order not in ("ASC", "DESC"):
        sort_order = "DESC"

    jql = " AND ".join(jql_parts)
    if sort_by:
        jql += f" ORDER BY {sort_by} {sort_order}"

    try:
        response = jira.enhanced_search_issues(
            jql_str=jql,
            fields=[
                "summary", "issuetype", "status", "assignee", "reporter",
                "created", "updated", "project", "resolution", "priority"
            ],
            use_post=True,
            json_result=False
        )
        return [extract_issue_fields(issue) for issue in response]

    except Exception as e:
        return [{"error": str(e), "jql": jql}]


def _get_tickets_insights(ticket_keys: List[str]) -> Dict:
    import textwrap
    import json
    import re

    summaries = {}
    extracted_data = {}

    # Step 1: Extract fields from all issues first
    for key in ticket_keys:
        try:
            extracted_data[key] = _extract_issue_fields(
                jira.issue(key), include_comments=True, jira_client=jira
            )
        except Exception as e:
            summaries[key] = f"❌ Error fetching ticket data: {e}"

    # Step 2: Build full user input for all tickets
    all_ticket_inputs = []
    for key, data in extracted_data.items():
        comment_text = "\n".join(f"{c['author']}: {c['body']}" for c in data.get("comments", []))

        ticket_input = textwrap.dedent(f"""
            Ticket {key}:
            Summary: {data.get('summary')}
            Status: {data.get('status')}
            Priority: {data.get('priority')}
            Assignee: {data.get('assignee')}
            Created: {data.get('created')}
            Updated: {data.get('updated')}

            Description:
            {data.get('description', '')}

            Comments:
            {comment_text}
        """).strip()

        all_ticket_inputs.append(ticket_input)

    full_input = "\n\n".join(all_ticket_inputs)

    system_prompt = (
        "You are a senior Jira analyst. The user will give you raw ticket data for multiple tickets.\n\n"
        "Your job is to create a summary for each ticket, using this structure:\n\n"
        "1. Start with a structured header that includes:\n"
        "   - Summary\n"
        "   - Status\n"
        "   - Priority\n"
        "   - Assignee\n"
        "   - Created\n"
        "   - Updated\n"
        "2. Then provide:\n"
        "   - A short summary of the ticket’s purpose or issue.\n"
        "   - The most recent news based on comments or updates.\n"
        "   - The suggested next step for the ticket.\n\n"
        "FORMAT:\n"
        "{\n"
        "  \"TICKET-123\": \"Summary: ...\\nStatus: ...\\nPriority: ...\\nAssignee: ...\\nCreated: ...\\nUpdated: ...\\n\\nTicket summary... Latest update... Suggested next step...\",\n"
        "  \"TICKET-456\": \"...\"\n"
        "}\n\n"
        "- Output MUST be valid JSON. No markdown or extra commentary."
    )

    user_input = f"Here is the data for the following tickets:\n\n{full_input}"

    try:
        response = call_nova_lite(system_prompt + "\n\nUser Input:\n" + user_input)
        print(f"\n🔍 LLM raw response:\n{response}\n")

        try:
            summaries = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{(?:[^{}]|(?R))*\}", response, re.DOTALL)
            if match:
                partial = match.group(0)
                summaries = json.loads(partial)
            else:
                for key in extracted_data.keys():
                    summaries[key] = f"❌ Failed to parse response:\n{response}"
    except Exception as e:
        for key in extracted_data.keys():
            summaries[key] = f"❌ Error summarizing ticket: {e}"

    return summaries





    

def _summarize_and_analyze_jql(jql: str) -> Dict:
    """
    Simplified summary of Jira issues per project:
    - Total ticket count
    - Unresolved issue count
    - Split by priority
    - Split by status
    - Split by assignee
    - For Incident SLA only:
        - Count by priority
        - Average resolution time by priority
    """
    try:
        from collections import Counter, defaultdict
        from datetime import datetime
        import pytz

        all_issues = []
        next_page_token = None
        max_results = 100

        per_project_data = defaultdict(lambda: {
            "project_name": "",
            "total": 0,
            "unresolved": 0,
            "by_priority": Counter(),
            "by_status": Counter(),
            "by_assignee": Counter(),
            "incident_sla_count_by_priority": Counter(),
            "incident_sla_resolution_by_priority": defaultdict(list)
        })

        while True:
            issues = jira.enhanced_search_issues(
                jql_str=jql,
                nextPageToken=next_page_token,
                maxResults=max_results,
                fields=["project", "priority", "issuetype", "created", "resolutiondate", "status", "assignee", "resolution"],
                use_post=True
            )

            if not issues:
                break

            all_issues.extend(issues)
            next_page_token = getattr(issues, "nextPageToken", None)
            if not next_page_token:
                break

        for issue in all_issues:
            fields = issue.fields
            project_key = getattr(fields.project, "key", "UNKNOWN")
            project_name = getattr(fields.project, "name", "Unknown Project")
            priority = getattr(fields.priority, "name", "None")
            issue_type = getattr(fields.issuetype, "name", "Unknown")
            status = getattr(fields.status, "name", "Unknown")
            assignee = getattr(fields.assignee, "displayName", "Unassigned")
            created = fields.created
            resolved = getattr(fields, "resolutiondate", None)
            resolution = getattr(fields, "resolution", None)

            data = per_project_data[project_key]
            data["project_name"] = project_name
            data["total"] += 1
            data["by_priority"][priority] += 1
            data["by_status"][status] += 1
            data["by_assignee"][assignee] += 1

            if not resolution:
                data["unresolved"] += 1

            if issue_type == "Incident SLA":
                data["incident_sla_count_by_priority"][priority] += 1
                if created and resolved:
                    try:
                        created_dt = datetime.strptime(created[:19], "%Y-%m-%dT%H:%M:%S")
                        resolved_dt = datetime.strptime(resolved[:19], "%Y-%m-%dT%H:%M:%S")
                        days_to_resolve = (resolved_dt - created_dt).total_seconds() / 86400
                        data["incident_sla_resolution_by_priority"][priority].append(days_to_resolve)
                    except Exception:
                        pass

        simplified_output = {}
        for project_key, data in per_project_data.items():
            simplified_output[project_key] = {
                "project_name": data["project_name"],
                "total_issues": data["total"],
                "unresolved_issues": data["unresolved"],
                "by_priority": dict(data["by_priority"]),
                "by_status": dict(data["by_status"]),
                "by_assignee": dict(data["by_assignee"]),
                "incident_sla_count_by_priority": dict(data["incident_sla_count_by_priority"]),
                "incident_sla_avg_resolution_by_priority": {
                    prio: round(sum(times) / len(times), 2)
                    for prio, times in data["incident_sla_resolution_by_priority"].items()
                }
            }

        return simplified_output

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze JQL: {e}")


def _get_issue_keys(jql: str) -> List[str]:
    """
    Fetches issue keys for all issues matching the given JQL.

    Args:
        jql (str): Jira Query Language string.

    Returns:
        List[str]: List of issue keys.
    """
    try:
        issue_keys = []
        next_page_token = None
        max_results = 100

        while True:
            issues = jira.enhanced_search_issues(
                jql_str=jql,
                nextPageToken=next_page_token,
                maxResults=max_results,
                fields=["key"],
                use_post=True
            )

            if not issues:
                break

            issue_keys.extend([issue.key for issue in issues])
            next_page_token = getattr(issues, "nextPageToken", None)

            if not next_page_token:
                break

        return issue_keys

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch issue keys: {e}")