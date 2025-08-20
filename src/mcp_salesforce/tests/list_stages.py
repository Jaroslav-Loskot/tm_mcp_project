from simple_salesforce.api import Salesforce
from thefuzz import fuzz
from typing import List, Dict, Any
import mcp_salesforce.helpers as helpers


print(helpers._find_best_name_matches("CAF Bank"))