# $env:PYTHONPATH = ".\src"

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from mcp_jira.helpers import _analyze_jira_issues_from_jql, _approximate_jira_issue_count, _generate_jql_from_input, _advanced_search_issues, _summarize_jira_tickets



# print(_generate_jql_from_input("Erste Austria unresolved issues ordered by priority"))

# print(_advanced_search_issues(
#     projects=["ASUCIT"],
#     resolved=False,
#     priorities=["High", "Highest"],
#     created_after="2025-07-01",
#     updated_after="2025-07-30",
#     sort_by="updated",
#     sort_order="ASC"
# ))

list = [
"ASEAT-351",
"ASEAT-349",
"ASEAT-344",
"ASEAT-339",
"ASEAT-328",
"ASEAT-39",
"ASEAT-16",
"ASEAT-13",
"ASEAT-12",
"ASEAT-352",
"ASEAT-350",
"ASEAT-340",
"ASEAT-338",
"ASEAT-317",
"ASEAT-283",
"ASEAT-11",
"ASEAT-10"
]



# print(_summarize_jira_tickets(list))

jql = _generate_jql_from_input("Erste Austria critical prirority")
print(jql)
# print(_approximate_jira_issue_count(jql['jql']))
# print(_analyze_jira_issues_from_jql(jql['jql']))