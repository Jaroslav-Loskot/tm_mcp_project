import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


from mcp_jira.helpers import _generate_jql_from_input


# print(list_appsupport_projects())

print(_generate_jql_from_input("project = ASEAT"))

