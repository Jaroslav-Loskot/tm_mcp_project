# $env:PYTHONPATH = ".\src"

import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from langchain_core.messages import HumanMessage
import mcp_jira.helpers as helpers
from mcp_jira.agent_generate_jql_supervisor import ask_agent_to_generate_jql
import mcp_jira

# print(helpers._resolve_project_name("UniCredit bank Italy"))

# result = call_agent_generate_jql("UniCredit Italy AND UniCredit Austria all opened issues with the type SLA Incident and Incident non SLA? Only for the last 3 months?")

# pretty_print_messages(result)

# result = ask_agent_to_generate_jql("all active tickets with the type SLA incidents created within the past year") 
# print("\n✅ Final structured JQL output:")
# print(json.dumps(result, indent=2))


print(mcp_jira.main.generate_jql_from_input("all SLA incidents created within the past year in Potbank PL"))

# print(helpers._resolve_types_and_statuses(""))

