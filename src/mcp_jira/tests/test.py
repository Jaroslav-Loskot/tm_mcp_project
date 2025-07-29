# $env:PYTHONPATH = ".\src"

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from jira import JIRA
from dotenv import load_dotenv


from mcp_common.utils.bedrock_wrapper import call_claude
from mcp_jira.helpers import _execute_jql_query, _generate_jql_from_input, _list_projects, _parse_jira_date, _resolve_project_name, _summarize_jql_query, extract_issue_fields


load_dotenv(override=True)

JIRA_URL = os.getenv("JIRA_BASE_URL")
JIRA_USER = os.getenv("JIRA_EMAIL")
JIRA_TOKEN = os.getenv("JIRA_API_TOKEN")

DEFAULT_CATEGORY = os.getenv("DEFAULT_PROJECT_CATEGORY", "")
EXCLUDED_KEYS = [k.strip() for k in os.getenv("EXCLUDED_PROJECT_KEYS", "").split(",") if k.strip()]

jira = JIRA(server=JIRA_URL, basic_auth=(JIRA_USER, JIRA_TOKEN))




# print(list_appsupport_projects())
# print(_generate_jql_from_input("project = ASEAT"))
print(len(_execute_jql_query("project = ASEAT")))

