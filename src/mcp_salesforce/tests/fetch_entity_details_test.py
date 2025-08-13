# src/mcp_salesforce/tests/fetch_entity_details_test.py
import os
import json
from dotenv import load_dotenv
import mcp_salesforce.helpers as helpers



if __name__ == "__main__":
    load_dotenv(override=True)

    # You can override these via environment variables for your org
    ACCOUNT_ID = "001MI00000f0XysYAE"
    OPP_ID     = "0062p00001LIisfAAD"          # e.g., "006xxxxxxxxxxxx"
    ACCOUNT_NM = "Butterfield Bank"
    OPP_NM     = "+ 1 channel integration"

    print(helpers.fetch_entity_details_tool(OPP_NM))



