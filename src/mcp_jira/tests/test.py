import sys
import os

from mcp_jira.main import list_projects
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mcp_jira.helpers import _generate_jql_from_input, _resolve_project_name


# print(list_appsupport_projects())

print(list_projects())

