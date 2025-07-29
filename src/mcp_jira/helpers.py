from dataclasses import dataclass
import json
import logging
import time
from typing import Dict, List, Optional
from difflib import get_close_matches
from fastapi import HTTPException
from datetime import datetime, timedelta
import re
from jira import JIRA  # Atlassian Python client
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import re
from mcp_common.utils.bedrock_wrapper import call_claude, call_nova_lite
# from mcp_jira.helpers import get_clean_comments_from_issue


load_dotenv(override=True)


JIRA_URL = os.getenv("JIRA_BASE_URL")
JIRA_USER = os.getenv("JIRA_EMAIL")
JIRA_TOKEN = os.getenv("JIRA_API_TOKEN")

DEFAULT_CATEGORY = os.getenv("DEFAULT_PROJECT_CATEGORY", "")
EXCLUDED_KEYS = [k.strip() for k in os.getenv("EXCLUDED_PROJECT_KEYS", "").split(",") if k.strip()]

jira = JIRA(server=JIRA_URL, basic_auth=(JIRA_USER, JIRA_TOKEN))


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


def extract_issue_fields(issue, include_comments=False, jira_client=None):
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
    now = datetime.utcnow()

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
    exclude_projects: Optional[List[str]] = os.getenv("EXCLUDED_PROJECT_KEYS", "")
) -> dict:
    """
    Converts natural language input into JQL using Claude.

    Args:
        user_input: Free-form user query like "all high priority tickets for Erste".
        category_filter: Optional Jira project category to include only certain projects.
        exclude_projects: Optional list of project keys to exclude.

    Returns:
        dict with 'jql' and 'max_results' keys.
    """
    all_projects = _list_projects()  # [{'key': 'UCB', 'name': 'Unicredit Italy', 'category': 'AppSupport'}, ...]

    # Optional: Filter by category
    if category_filter:
        all_projects = [
            p for p in all_projects
            if p.get("category", "").lower() == category_filter.lower()
        ]

    # Optional: Exclude project keys
    if exclude_projects:
        all_projects = [
            p for p in all_projects
            if p["key"] not in exclude_projects
        ]

    if not all_projects:
        raise ValueError("No allowed projects after applying filters.")

    allowed_project_keys = [p["key"] for p in all_projects]
    allowed_priorities = get_all_jira_priorities()  # e.g., ["High", "Medium", "Low"]

    system_prompt = (
        "You are a Jira assistant that converts natural language requests into structured JSON "
        "for querying Jira issues.\n\n"
        "RULES:\n"
        "- Only use the provided project keys and priorities.\n"
        "- For issue status, DO NOT use raw status names (like 'In Progress', 'To Do', etc).\n"
        "- Instead, determine the resolution type from the user input:\n"
        "   - Use 'resolution = Unresolved' for open/incomplete issues\n"
        "   - Use 'resolution != Unresolved' for closed/completed issues\n"
        "   - Omit resolution condition if the user meant 'all' issues\n"
        "- If a priority is mentioned, use it in a priority clause.\n"
        "- If no priority is mentioned, omit it.\n"
        "- DO NOT include max_results or any limit fields.\n"
        "- Return a JSON object ONLY with this field:\n"
        "   - jql: string\n"
        "- DO NOT include explanations or markdown, just return the JSON.\n"
    )


    user_message = f"""
    User Input:
    {user_input}

    Allowed project keys:
    {', '.join(allowed_project_keys)}

    Allowed priorities:
    {', '.join(allowed_priorities)}

    Expected output format:

    {{
      "jql": "<VALID_JQL_STRING>"
    }}
    """

    response = call_claude(system_prompt, user_message).strip()

    try:
        fenced = re.search(r"\{.*\}", response, re.DOTALL)
        response_json = fenced.group(0) if fenced else response
        result = json.loads(response_json)
    except Exception as e:
        raise ValueError(f"Failed to parse Claude's JSON output: {e}\n\nRaw response:\n{response}")

    if not isinstance(result, dict) or "jql" not in result:
        raise ValueError(f"Claude did not return a valid structure: {result}")


    return result




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




