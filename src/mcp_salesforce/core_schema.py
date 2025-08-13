# mcp_salesforce/core_schema.py
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from mcp_salesforce import helpers  # uses helpers.get_sf_connection()

# -------------------------------------------------------------------
# Friendly → candidate API names
# (Only fields that exist in your org will be included at runtime.)
# -------------------------------------------------------------------

FIELD_CANDIDATES: Dict[str, Dict[str, List[str]]] = {
    "Opportunity": {
        "Opportunity.Id":            ["Id"],
        "Opportunity.Name":          ["Name"],
        "Opportunity.StageName":     ["StageName"],
        "Opportunity.Amount":        ["Amount"],
        "Opportunity.CloseDate":     ["CloseDate"],
        "Opportunity.NextStep":      ["NextStep"],
        "Opportunity.Currency":      ["CurrencyIsoCode"],  # if multi-currency
        "Opportunity.OwnerId":       ["OwnerId"],
        "Opportunity.LastActivity":  ["LastActivityDate"],
        "Opportunity.RenewalDate":   ["Renewal_Date__c", "RenewalDate__c"],
        "Opportunity.ACV":           ["ACV__c", "Annual_Contract_Value__c"],
        "Opportunity.ARR":           ["ARR__c", "Annual_Recurring_Revenue__c"],
        "Opportunity.Territory":     ["Territory2Id", "Territory__c", "Territory"],
        "Opportunity.Description":   ["Description"],
    },
    "Account": {
        "Account.Id":                ["Id"],
        "Account.Name":              ["Name"],

        # — Your “Account attributes” list —
        "Account.Type":              ["Type"],                # picklist
        "Account.RecordTypeId":      ["RecordTypeId"],        # reference
        "Account.ParentId":          ["ParentId"],            # reference
        "Account.BillingCountry":    ["BillingCountry"],      # string
        "Account.ShippingCountry":   ["ShippingCountry"],     # string
        "Account.Phone":             ["Phone"],               # phone
        "Account.Industry":          ["Industry"],            # picklist
        "Account.CurrencyIsoCode":   ["CurrencyIsoCode"],     # picklist (multi-currency)
        "Account.OwnerId":           ["OwnerId"],             # reference
        "Account.Territory__c":      ["Territory__c"],        # picklist (custom)
        "Account.Country__c":        ["Country__c"],          # picklist (custom)
        "Account.Partner_Type__c":   ["Partner_Type__c"],     # picklist (custom)
        "Account.RecordTypeId__c":   ["RecordTypeId__c"],     # custom string
        "Account.Customer_Type__c":  ["Customer_Type__c"],    # picklist (custom)
    },
}


# -------------------------------------------------------------------
# Helpers to normalize describe() field → compact schema
# -------------------------------------------------------------------

def _find(desc: Dict[str, Any], candidates: List[str]) -> Optional[Dict[str, Any]]:
    """Return the first field from `desc` whose API name matches any candidate."""
    by_name = {f["name"]: f for f in desc.get("fields", []) or []}
    for c in candidates:
        if c in by_name:
            return by_name[c]
    # case-insensitive fallback
    lowered = {k.lower(): v for k, v in by_name.items()}
    for c in candidates:
        f = lowered.get(c.lower())
        if f:
            return f
    return None


def _schema_entry(f: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, token-light schema snapshot for one field."""
    out: Dict[str, Any] = {
        "api":         f["name"],
        "label":       f.get("label"),
        "type":        f.get("type"),
        "filterable":  bool(f.get("filterable")),
        "sortable":    bool(f.get("sortable")),
    }
    if f.get("type") in ("picklist", "multipicklist"):
        out["values"] = [pv["value"] for pv in f.get("picklistValues", []) if pv.get("active", True)]
    if f.get("referenceTo"):
        out["references"] = list(f.get("referenceTo") or [])
    if f.get("relationshipName"):
        out["relationshipName"] = f["relationshipName"]
    return out


# -------------------------------------------------------------------
# Build dynamic allow-list index using describe()
# -------------------------------------------------------------------

@lru_cache(maxsize=1)
def build_core_field_index_cached() -> Dict[str, Dict[str, Any]]:
    """
    Resolve FIELD_CANDIDATES against your org’s schema and return:
      { "<FriendlyName>": {api, label, type, filterable, sortable, [values], [references], [relationshipName] }, ... }
    Only fields that actually exist in your org are included.
    """
    sf = helpers.get_sf_connection()
    opp_desc = sf.Opportunity.describe()
    acc_desc = sf.Account.describe()

    idx: Dict[str, Dict[str, Any]] = {}

    def add(object_desc: Dict[str, Any], friendly: str, candidates: List[str]) -> None:
        f = _find(object_desc, candidates)
        if f:
            idx[friendly] = _schema_entry(f)

    # Opportunity
    for friendly, candidates in FIELD_CANDIDATES["Opportunity"].items():
        add(opp_desc, friendly, candidates)

    # Account
    for friendly, candidates in FIELD_CANDIDATES["Account"].items():
        add(acc_desc, friendly, candidates)

    return idx


def refresh_core_field_cache() -> None:
    """Clear the LRU cache. Call after metadata changes."""
    build_core_field_index_cached.cache_clear()  # type: ignore[attr-defined]


# -------------------------------------------------------------------
# Public convenience APIs
# -------------------------------------------------------------------

def list_allowed_attribute_names() -> List[str]:
    """Stable, sorted list of friendly attribute names allowed for SOQL & details."""
    return sorted(build_core_field_index_cached().keys())


def get_schema_for_attribute(friendly_attr: str) -> Optional[Dict[str, Any]]:
    """
    Return the compact schema dict for a friendly attr like 'Account.Type' or 'Opportunity.StageName'.
    Returns None if not available in this org.
    """
    return build_core_field_index_cached().get(friendly_attr)


def resolve_object_and_api(friendly_attr: str) -> Optional[Tuple[str, str]]:
    """
    Convert 'Account.Type' → ('Account', 'Type') using the resolved API from describe().
    Returns None if the friendly name is unknown.
    """
    schema = get_schema_for_attribute(friendly_attr)
    if not schema:
        return None
    obj = friendly_attr.split(".", 1)[0]
    return obj, schema["api"]


def detail_select_list(object_api: str) -> List[str]:
    """
    For a given object ('Account' or 'Opportunity'), return the *actual* API fields to SELECT for details,
    filtered to those that exist in this org.
    """
    idx = build_core_field_index_cached()
    prefix = f"{object_api}."
    out: List[str] = []
    for friendly, schema in idx.items():
        if friendly.startswith(prefix):
            out.append(schema["api"])  # actual field API
    # de-dup while preserving order
    seen = set()
    uniq = []
    for a in out:
        if a in seen:
            continue
        seen.add(a)
        uniq.append(a)
    return uniq


if __name__ == "__main__":
    pass