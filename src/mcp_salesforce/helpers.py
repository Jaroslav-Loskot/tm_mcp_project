import json
import os
import re
from dotenv import load_dotenv
from simple_salesforce import Salesforce
from typing import Any, Dict, List, Optional, Iterable, Tuple, Union
import mcp_salesforce.core_schema as core_schema
import mcp_common.utils.bedrock_wrapper as bedrock_wrapper
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage


load_dotenv()

def get_sf_connection() -> Salesforce:
    """Create and return a Salesforce connection from environment variables."""
    username = os.getenv("SALESFORCE_USERNAME")
    password = os.getenv("SALESFORCE_PASSWORD")
    security_token = os.getenv("SALESFORCE_SECURITY_TOKEN")
    client_id = os.getenv("SALESFORCE_CLIENT_ID") or "simple-salesforce-client"
    domain = os.getenv("SALESFORCE_DOMAIN")  # e.g., 'login', 'test', or 'yourdomain.my.salesforce.com'

    return Salesforce(
        username=username,
        password=password,
        security_token=security_token,
        client_id=client_id,
        domain=domain
    )


# --------------------------------------------------------------------------------------
# SMALL UTILS
# --------------------------------------------------------------------------------------

_SF_ID_RE = re.compile(r"^[a-zA-Z0-9]{15,18}$")
_PREFIX_TO_OBJ = {"001": "Account", "006": "Opportunity"}  # extend if needed

def _escape_soql_literal(s: str) -> str:
    # minimal escaping for SOQL string literal
    return s.replace("\\", "\\\\").replace("'", "\\'")

def _flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten relationship sub-objects into dot-keys (Owner.Name → 'Owner.Name')."""
    out: Dict[str, Any] = {}
    for k, v in rec.items():
        if k == "attributes":
            continue
        if isinstance(v, dict) and "attributes" in v:
            sub = _flatten_record(v)
            for sk, sv in sub.items():
                out[f"{k}.{sk}"] = sv
        else:
            out[k] = v
    return out




def _strip_attributes(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in r.items() if k != "attributes"} for r in records]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _pre_score(q: str, name: str) -> float:
    """0..100 similarity; simple & dependency-free."""
    from difflib import SequenceMatcher
    qn, nn = _norm(q), _norm(name)
    if not qn or not nn:
        return 0.0
    r = SequenceMatcher(None, qn, nn).ratio() * 100.0
    if qn in nn or nn in qn:
        r = min(100.0, r + 5.0)  # containment bonus
    return round(r, 1)


def _sosl_escape(term: str) -> str:
    """Very light sanitize for SOSL curly braces."""
    return term.replace("{", " ").replace("}", " ").strip()


def _extract_json_array(s: str) -> List[Dict[str, Any]]:
    """Extract the first JSON array from a model response (handles fenced blocks)."""
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        j = json.loads(s)
        if isinstance(j, list):
            return j
    except Exception:
        pass
    start = s.find("[")
    if start == -1:
        return []
    depth = 0
    end = None
    for i, ch in enumerate(s[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return []
    return json.loads(s[start:end])


def _chunk(seq: List[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _pp_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def _maybe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return s



def _strip_attributes(records):
    """Remove the 'attributes' key that SF adds to each record."""
    return [{k: v for k, v in r.items() if k != "attributes"} for r in records]

def fetch_accounts_and_opportunities(
    sf: Salesforce,
    account_fields=None,
    opportunity_fields=None
):
    """
    Fetch all Accounts and Opportunities and return them as lists of dicts.

    Parameters
    ----------
    sf : Salesforce
        An authenticated simple_salesforce Salesforce instance.
    account_fields : list[str] | None
        Fields to pull for Account. Defaults to a useful set.
    opportunity_fields : list[str] | None
        Fields to pull for Opportunity. Defaults to a useful set.

    Returns
    -------
    tuple[list[dict], list[dict]]
        (accounts, opportunities)
    """
    if account_fields is None:
        account_fields = ["Id", "Name", "Type", "Industry", "BillingCountry", "OwnerId", "CreatedDate", "LastModifiedDate"]

    if opportunity_fields is None:
        opportunity_fields = ["Id", "Name", "AccountId", "StageName", "Amount", "CloseDate", "OwnerId", "CreatedDate", "LastModifiedDate"]

    account_query = f"SELECT {', '.join(account_fields)} FROM Account"
    opp_query = f"SELECT {', '.join(opportunity_fields)} FROM Opportunity"

    # query_all will follow nextRecordsUrl under the hood and return the full set
    accounts_result = sf.query_all(account_query)
    opportunities_result = sf.query_all(opp_query)

    accounts = _strip_attributes(accounts_result.get("records", []))
    opportunities = _strip_attributes(opportunities_result.get("records", []))

    return accounts, opportunities



from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from difflib import SequenceMatcher

# assumes you have:
# - _require_sf() -> Salesforce
# - _sosl_escape(s: str) -> str
# - _pre_score(q: str, name: str) -> float (0..100)
# - core_schema.get_schema_for_attribute(attr: str) -> dict | None
# - core_schema.list_core_attribute_names() or similar (not required here)
# - detail_select_list(object_api: str) -> List[str]  # returns friendly attrs like "Opportunity.Id", "Account.BillingCountry", ...

ALLOWED_OBJECTS = {"Account": "001", "Opportunity": "006"}  # prefixes

def _looks_like_id(s: str) -> bool:
    s = (s or "").strip()
    return len(s) in (15, 18) and s.isalnum()

def _id_prefix(s: str) -> str:
    return (s or "")[:3]

def _flatten_record(rec: Dict[str, Any], *, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested SF relationship dicts into dot-keys (e.g., 'Account.Name')."""
    out: Dict[str, Any] = {}
    for k, v in rec.items():
        if k == "attributes":
            continue
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict) and v:
            out.update(_flatten_record(v, prefix=key))
        else:
            out[key] = v
    return out

def _friendly_to_select_expr(object_api: str, friendly_attr: str) -> Optional[str]:
    """
    Map a friendly attribute like 'Opportunity.Amount' or 'Account.BillingCountry'
    to a proper SOQL select expression for the requested object (Account/Opportunity).
    """
    from mcp_salesforce import core_schema  # import here to avoid circulars
    sch = core_schema.get_schema_for_attribute(friendly_attr)
    if not sch:
        return None
    # friendly_attr is "<Obj>.<Suffix>"
    obj, _ = friendly_attr.split(".", 1)
    api = sch.get("api")
    if not api:
        return None

    # Selecting fields ON the requested object:
    if obj == object_api:
        return api

    # Allow selecting parent Account fields from Opportunity via relationship
    if object_api == "Opportunity" and obj == "Account":
        return f"Account.{api}"

    # Otherwise skip (keep simple + predictable)
    return None

def llm_pick_best_name_matches_tool(
    query: str,
    k: int = 5,
    max_per_type: int = 400,
    shortlist_cap: int = 80,
    force: bool = True,
) -> List[Dict[str, Any]]:
    """
    Let the LLM pick top-K best matches across Account + Opportunity.
    Always returns both IDs (non-applicable one is None).

    Output rows:
    {
      "account_id": "001... or None",
      "opportunity_id": "006... or None",
      "name": "<Name>",
      "type": "Account" | "Opportunity",
      "match_score": 0..100
    }
    """
    sf = get_sf_connection()
    term = _sosl_escape(query)
    candidates: List[Dict[str, Any]] = []

    # SOSL (prefix then fuzzy) over both
    returning = [
        f"Account(Id, Name ORDER BY Name LIMIT {int(max_per_type)})",
        f"Opportunity(Id, Name, AccountId ORDER BY Name LIMIT {int(max_per_type)})",
    ]

    def _add_sosl(records):
        for r in records or []:
            t = r.get("attributes", {}).get("type")
            if t == "Account":
                n = r.get("Name") or ""
                if n:
                    candidates.append({"account_id": r.get("Id"), "opportunity_id": None, "name": n, "type": "Account"})
            elif t == "Opportunity":
                n = r.get("Name") or ""
                if n:
                    candidates.append({"account_id": r.get("AccountId"), "opportunity_id": r.get("Id"), "name": n, "type": "Opportunity"})

    if term:
        try:
            sosl = f"FIND {{{term}*}} IN NAME FIELDS RETURNING {', '.join(returning)}"
            _add_sosl(sf.search(sosl))
        except Exception:
            pass
        if not candidates:
            try:
                sosl_fuzzy = f"FIND {{{term}~}} IN NAME FIELDS RETURNING {', '.join(returning)}"
                _add_sosl(sf.search(sosl_fuzzy))
            except Exception:
                pass

    # SOQL LIKE fallback
    if not candidates and term:
        like = "%" + term.replace("'", "\\'") + "%"
        try:
            recs = sf.query(
                f"SELECT Id, Name FROM Account WHERE Name LIKE '{like}' ORDER BY Name LIMIT {int(max_per_type)}"
            ).get("records", [])
            for r in recs:
                n = r.get("Name")
                if n:
                    candidates.append({"account_id": r.get("Id"), "opportunity_id": None, "name": n, "type": "Account"})
        except Exception:
            pass
        try:
            recs = sf.query(
                f"SELECT Id, Name, AccountId FROM Opportunity WHERE Name LIKE '{like}' ORDER BY Name LIMIT {int(max_per_type)}"
            ).get("records", [])
            for r in recs:
                n = r.get("Name")
                if n:
                    candidates.append({"account_id": r.get("AccountId"), "opportunity_id": r.get("Id"), "name": n, "type": "Opportunity"})
        except Exception:
            pass

    # Force: sample recents
    if not candidates and force:
        try:
            recs = sf.query(
                f"SELECT Id, Name FROM Account WHERE Name != NULL ORDER BY LastModifiedDate DESC LIMIT {int(max_per_type)}"
            ).get("records", [])
            for r in recs:
                n = r.get("Name")
                if n:
                    candidates.append({"account_id": r.get("Id"), "opportunity_id": None, "name": n, "type": "Account"})
        except Exception:
            pass
        try:
            recs = sf.query(
                f"SELECT Id, Name, AccountId FROM Opportunity WHERE Name != NULL ORDER BY LastModifiedDate DESC LIMIT {int(max_per_type)}"
            ).get("records", [])
            for r in recs:
                n = r.get("Name")
                if n:
                    candidates.append({"account_id": r.get("AccountId"), "opportunity_id": r.get("Id"), "name": n, "type": "Opportunity"})
        except Exception:
            pass

    if not candidates:
        return []

    # De-dup + pre-score + shortlist
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for c in candidates:
        key = (c.get("account_id"), c.get("opportunity_id"), c["type"])
        if key in seen:
            continue
        seen.add(key)
        c["pre_score"] = _pre_score(query, c["name"])
        uniq.append(c)

    uniq.sort(key=lambda x: (-x["pre_score"], x["type"], x["name"]))
    shortlist = uniq[: max(1, shortlist_cap)]

    # LLM rank
    out: List[Dict[str, Any]] = []
    try:
        llm = bedrock_wrapper.init_chat_model("NOVA_LITE_MODEL_ID")
        system = (
            "Rank candidate names by similarity to the user query. "
            "Return ONLY a JSON array with objects: {account_id, opportunity_id, name, type, match_score}. "
            "match_score is 0-100. Sort by match_score desc. No extra text."
        )
        payload = {
            "query": query,
            "k": k,
            "candidates": [
                {
                    "account_id": c["account_id"],
                    "opportunity_id": c["opportunity_id"],
                    "name": c["name"],
                    "type": c["type"],
                    "pre_score": c["pre_score"],
                }
                for c in shortlist
            ],
        }
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=json.dumps(payload, ensure_ascii=False))]).content
        ranked = _extract_json_array(resp)

        seen2 = set()
        for r in ranked:
            aid, oid, name, typ = r.get("account_id"), r.get("opportunity_id"), r.get("name"), r.get("type")
            if not (name and typ):
                continue
            key = (aid, oid, typ)
            if key in seen2:
                continue
            seen2.add(key)
            try:
                ms = float(r.get("match_score", 0))
            except Exception:
                ms = 0.0
            out.append(
                {
                    "account_id": aid or None,
                    "opportunity_id": oid or None,
                    "name": name,
                    "type": typ,
                    "match_score": max(0.0, min(100.0, round(ms, 1))),
                }
            )
            if len(out) >= max(1, k):
                break
    except Exception:
        out = []

    # Local fallback
    if not out:
        out = [
            {
                "account_id": c["account_id"],
                "opportunity_id": c["opportunity_id"],
                "name": c["name"],
                "type": c["type"],
                "match_score": float(c["pre_score"]),
            }
            for c in shortlist[: max(1, k)]
        ]

    # normalize None
    for r in out:
        r["account_id"] = r.get("account_id") or None
        r["opportunity_id"] = r.get("opportunity_id") or None

    return out


from typing import Dict, Any, List, Optional
from langchain_core.tools import tool



def _lookup_owner_name(sf, owner_id: str) -> Optional[str]:
    """Try User then Group; return Name or None."""
    if not owner_id:
        return None
    for obj in ("User", "Group"):
        try:
            q = f"SELECT Id, Name FROM {obj} WHERE Id = '{_escape_soql_literal(owner_id)}' LIMIT 1"
            recs = sf.query(q).get("records", [])
            if recs:
                return recs[0].get("Name")
        except Exception:
            continue
    return None


def fetch_entity_details_tool(id_or_name: str) -> Dict[str, Any]:
    """
    Fetch a SINGLE Account or Opportunity by Salesforce Id ONLY (no names).

    Input:
      - id_or_name: 15/18-char Salesforce Id.
        Supported prefixes:
          • 001… → Account
          • 006… → Opportunity

    Behavior:
      - Determines object from Id prefix; rejects non-001/006 prefixes.
      - SELECT fields come from core_schema.detail_select_list(<object>) + (Id, Name).
      - If Opportunity: enrich Owner.Name when missing, and return its Account {id, name}.
      - If Account: return ALL linked Opportunities [{id, name}, ...] (paged), with a safety cap.

    Output:
      {
        "object": "Account|Opportunity",
        "input": "<original>",
        "resolved": { "id": "<Id>", "name": "<Name>", "match_score": null, "matched_via": "ID" },
        "select": ["Id","Name","..."],   # actual SOQL expressions used
        "soql": "SELECT ... FROM ... WHERE Id='...' LIMIT 1",
        "record": { "Id": "...", "Name": "...", "Owner.Name": "...", ... },  # flattened
        "related": {
          "account": { "id": "...", "name": "..." }                # for Opportunity
          # OR
          "opportunities": { "count": <int>, "items": [{ "id": "...", "name": "..." }, ...] }  # for Account
        }
      }
      or { "error": "...", "input": "<...>" }
    """
    try:
        # Use your module-level connection + helpers
        sf = get_sf_connection()
        rid = (id_or_name or "").strip()
        if not rid:
            return {"error": "Empty input.", "input": id_or_name}

        # Only accept valid SFIDs
        if not _SF_ID_RE.match(rid):
            return {
                "error": "Only Salesforce Ids are accepted (15/18 chars). Provide Account (001…) or Opportunity (006…) Id.",
                "input": id_or_name,
            }

        # Enforce supported prefixes
        prefix = rid[:3]
        if prefix == "001":
            candidate_objs = ["Account"]
        elif prefix == "006":
            candidate_objs = ["Opportunity"]
        else:
            return {
                "error": f"Unsupported Id prefix '{prefix}'. Only Account (001) and Opportunity (006) are supported.",
                "input": id_or_name,
            }

        # lazy import to avoid import cycles
        import mcp_salesforce.core_schema as core_schema  # provides detail_select_list

        for obj in candidate_objs:
            # Build select list from curated schema
            select_exprs: List[str] = core_schema.detail_select_list(obj) or []
            base = {s.lower() for s in select_exprs}
            if "id" not in base:
                select_exprs.insert(0, "Id")
                base.add("id")
            if "name" not in base:
                select_exprs.insert(1, "Name")
                base.add("name")

            # Ensure fields needed for cross-linking:
            if obj == "Opportunity":
                # We want to expose the linked Account
                need = {"accountid", "account.name"}
                for f in ("AccountId", "Account.Name"):
                    if f.lower() not in base:
                        select_exprs.append(f)
                        base.add(f.lower())
                # Optionally ensure Owner.Name for convenience (enriched later if missing)
                if "owner.name" not in base:
                    select_exprs.append("Owner.Name")
                    base.add("owner.name")

            soql = f"SELECT {', '.join(select_exprs)} FROM {obj} WHERE Id = '{_escape_soql_literal(rid)}' LIMIT 1"
            try:
                res = sf.query(soql)
                recs = res.get("records") or []
            except Exception as e:
                return {"error": str(e), "input": id_or_name, "soql": soql}

            if not recs:
                continue  # try next (shouldn't happen if prefix is correct)

            flat = _flatten_record(recs[0])

            # Enrich Opportunity with Owner.Name if missing
            if obj == "Opportunity" and not flat.get("Owner.Name"):
                owner_id = flat.get("OwnerId") or flat.get("Owner.Id")
                if owner_id:
                    owner_name = _lookup_owner_name(sf, owner_id)
                    if owner_name:
                        flat["Owner.Name"] = owner_name

            out: Dict[str, Any] = {
                "object": obj,
                "input": id_or_name,
                "resolved": {
                    "id": flat.get("Id"),
                    "name": flat.get("Name"),
                    "match_score": None,
                    "matched_via": "ID",
                },
                "select": select_exprs,
                "soql": soql,
                "record": flat,
                "related": {},
            }

            # Cross-links
            if obj == "Opportunity":
                acct_id = flat.get("AccountId") or flat.get("Account.Id")
                acct_name = flat.get("Account.Name")
                # If name not present but id is, fetch it
                if acct_id and not acct_name:
                    try:
                        a = sf.query(
                            f"SELECT Name FROM Account WHERE Id = '{_escape_soql_literal(acct_id)}' LIMIT 1"
                        ).get("records", [])
                        if a:
                            acct_name = a[0].get("Name")
                    except Exception:
                        pass
                out["related"]["account"] = {
                    "id": acct_id,
                    "name": acct_name,
                }

            else:  # Account → list all related Opportunities (paged)
                acct_id = flat.get("Id")
                items: List[Dict[str, Any]] = []
                total = 0
                MAX_ROWS = 5000  # safety cap

                try:
                    q = f"SELECT Id, Name FROM Opportunity WHERE AccountId = '{_escape_soql_literal(acct_id)}' ORDER BY LastModifiedDate DESC"
                    res = sf.query(q)
                    total = int(res.get("totalSize", 0))
                    recs = res.get("records") or []
                    for r in recs:
                        items.append({"id": r.get("Id"), "name": r.get("Name")})
                        if len(items) >= MAX_ROWS:
                            break
                    # paginate if needed and under cap
                    next_url = res.get("nextRecordsUrl")
                    while next_url and len(items) < MAX_ROWS:
                        res = sf.query_more(next_url, identifier_is_url=True)
                        recs = res.get("records") or []
                        for r in recs:
                            items.append({"id": r.get("Id"), "name": r.get("Name")})
                            if len(items) >= MAX_ROWS:
                                break
                        next_url = res.get("nextRecordsUrl")
                except Exception:
                    # keep whatever we gathered
                    pass

                out["related"]["opportunities"] = {
                    "count": total if total else len(items),
                    "items": items,
                    "truncated": len(items) < total  # true if we hit cap or server has more
                }

            return out

        # If we got here, the Id didn't return a record for its expected object.
        return {"error": f"No {candidate_objs[0]} found for given Id.", "input": id_or_name}

    except Exception as e:
        return {"error": str(e), "input": id_or_name}




def resolve_owner_names_tool(owner_ids: List[str]) -> List[Dict[str, Optional[str]]]:
    """
    Translate a list of OwnerId values into [{id, name}] pairs.
    Supports Users (005...) and Groups/Queues (00G...). Missing/unknown -> name=None.
    Preserves input order and de-duplicates internally.
    """
    sf = get_sf_connection()
    ids = [i for i in (owner_ids or []) if i]
    if not ids:
        return []
    uniq = list(dict.fromkeys(ids))

    user_ids = [i for i in uniq if i.startswith("005")]
    group_ids = [i for i in uniq if i.startswith("00G")]
    other_ids = [i for i in uniq if i not in user_ids and i not in group_ids]

    id_to_name: Dict[str, Optional[str]] = {}

    def _query(obj: str, ids_subset: List[str], select: str = "Id, Name"):
        if not ids_subset:
            return
        for chunk in _chunk(ids_subset, 200):
            ids_escaped = ["'" + str(x).replace("'", "\\'") + "'" for x in chunk]
            in_list = ",".join(ids_escaped)
            soql = f"SELECT {select} FROM {obj} WHERE Id IN ({in_list})"
            try:
                for r in sf.query(soql).get("records", []):
                    _id = r.get("Id")
                    _nm = r.get("Name")
                    if _id:
                        id_to_name[_id] = _nm
            except Exception:
                pass

    _query("User", user_ids)
    _query("Group", group_ids + other_ids)

    out: List[Dict[str, Optional[str]]] = []
    for orig in ids:
        out.append({"id": orig, "name": id_to_name.get(orig)})
    return out



if __name__ == "__main__":
    pass 