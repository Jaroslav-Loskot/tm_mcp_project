import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


from mcp_jira.main import execute_jql_query


# print(list_appsupport_projects())

print(execute_jql_query("category in (AppSupport) AND project in (ASUCIT)"))

