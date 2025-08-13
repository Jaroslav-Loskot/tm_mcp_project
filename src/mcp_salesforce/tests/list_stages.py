from simple_salesforce import Salesforce
import os
from dotenv import load_dotenv
import mcp_salesforce.agent_generate_SOQL as agent
import mcp_salesforce.core_schema as core_schema
import mcp_salesforce.helpers as helpers

load_dotenv()


# print(agent.list_core_attribute_names_tool.invoke({}))
# print(agent.get_salesforce_field_schema.invoke({"full_api_path": "Account.Country__c"}))

# print(agent.llm_pick_best_name_matches_tool.invoke({"query": "CAF Bank", "k": 5}))

# print(core_schema.list_allowed_attribute_names())
# print(core_schema.get_schema_for_attribute("Account.Industry"))
# print(core_schema.detail_select_list("Opportunity"))

print(helpers._best_name_match("Alika Bank", 79))