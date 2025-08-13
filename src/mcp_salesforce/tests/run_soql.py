from simple_salesforce import SalesforceMalformedRequest
import mcp_salesforce.helpers as helpers 

sf = helpers.get_sf_connection()

soql = """
SELECT Id, Name, StageName, Amount, CloseDate FROM Opportunity WHERE StageName = 'Advanced Qualification' AND Opportunity.CreatedDate >= LAST_QUARTER ORDER BY CloseDate ASC
""".strip()

try:
    res = sf.query(soql)
    print("totalSize:", res.get("totalSize"))
    for rec in res.get("records", []):
        # strip Salesforce metadata
        clean = {k: v for k, v in rec.items() if k != "attributes"}
        print(clean)
except SalesforceMalformedRequest as e:
    print("MALFORMED_QUERY:", getattr(e, "content", e))
except Exception as e:
    print("Error:", e)
