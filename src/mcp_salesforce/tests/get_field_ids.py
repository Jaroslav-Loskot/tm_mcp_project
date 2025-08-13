import mcp_salesforce.helpers as helpers 

import json

sf = helpers.get_sf_connection()

FIELDS = [
    "Type",                 # Account Type (picklist)
    "RecordTypeId",         # Record Type ID (reference)
    "ParentId",             # Parent Account ID (reference)
    "BillingCountry",       # Billing Country (string)
    "ShippingCountry",      # Shipping Country (string)
    "Phone",                # Account Phone (phone)
    "Industry",             # Industry (picklist)
    "CurrencyIsoCode",      # Account Currency (picklist)
    "OwnerId",              # Owner ID (reference)
    "Territory__c",         # Territory (picklist)
    "Country__c",           # Country (picklist)
    "Partner_Type__c",      # Partner Type (picklist)
    "RecordTypeId__c",      # (custom string per your list)
    "Customer_Type__c",     # Customer Type (picklist)
]

def get_field_schema(object_api: str, field_api_names: list[str]) -> dict:
    meta = getattr(sf, object_api).describe()
    by_name = {f["name"]: f for f in meta["fields"]}

    def shape(f):
        out = {
            "api": f["name"],
            "label": f.get("label"),
            "type": f["type"],
            "filterable": f.get("filterable", False),
            "sortable": f.get("sortable", False),
        }
        # data shape details
        if "length" in f: out["length"] = f["length"]
        if "precision" in f: out["precision"] = f["precision"]
        if "scale" in f: out["scale"] = f["scale"]

        # picklist values
        if f["type"] in ("picklist", "multipicklist"):
            out["values"] = [pv["value"] for pv in f.get("picklistValues", []) if pv.get("active", True)]

        # references
        if f["type"] == "reference":
            out["references"] = f.get("referenceTo", [])
            if f.get("relationshipName"):
                out["relationshipName"] = f["relationshipName"]

        return out

    result = {"object": object_api, "fields": {}}
    for name in field_api_names:
        f = by_name.get(name)
        if f:
            result["fields"][name] = shape(f)
        else:
            result["fields"][name] = {"error": "Field not found in describe()"}
    return result

schema = get_field_schema("Account", FIELDS)
print(json.dumps(schema, indent=2, ensure_ascii=False))
