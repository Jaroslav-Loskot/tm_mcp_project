from simple_salesforce import Salesforce
from thefuzz import fuzz
from typing import List, Dict, Any
import mcp_salesforce.helpers as helpers


print(helpers.find_best_name_matches("CAF Bank"))