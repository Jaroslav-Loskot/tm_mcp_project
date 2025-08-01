# $env:PYTHONPATH = ".\src"

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from langchain_core.messages import HumanMessage
import mcp_jira.helpers as helpers
from mcp_jira.agent_generate_jql import graph, pretty_print_messages, call_agent_generate_jql, list_issue_type_statuses_tool


# print(helpers._resolve_project_name("UniCredit bank Italy"))

result = call_agent_generate_jql("UniCredit Italy AND UniCredit Austria all opened issues with the type SLA Incident and Incident non SLA? Only for the last 3 months?")
pretty_print_messages(result)
