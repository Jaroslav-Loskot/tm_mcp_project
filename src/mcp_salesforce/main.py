from fastmcp import FastMCP
import mcp_salesforce.agent_generate_SOQL as sf_agent
import mcp_salesforce.helpers as helpers
from typing import Any, Dict, List, Optional, Union
from simple_salesforce import SalesforceMalformedRequest

mcp = FastMCP("Salesforce MCP Server", auth=None)


@mcp.tool
def generate_soql_from_input(user_input: str) -> Union[str, Dict[str, Any]]:
    """
    Generate a SOQL string from natural language using your agent and
    return a compact result dict.

    Returns:
      {
        "soql": "<final SOQL or empty string>",
        "approx_row_count": <int or null>,
        "comment": "<agent comment/notes or empty>"
      }
    """

    return sf_agent.call_agent_generate_soql(user_input)
 

def _strip_attrs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in r.items() if k != "attributes"} for r in rows or []]

# Use your existing connection helper
# def _require_sf() -> Salesforce: ...  # assumed available in your module

@mcp.tool  # replace with @mcp.tool if that's your decorator
def execute_soql_tool(
    soql: str,
    limit: int = 2000,
    paginate: bool = False,
    max_rows: int = 5000,
    strip_attributes: bool = True
) -> Dict[str, Any]:
    """
    Execute a SOQL query and return results.

    Args:
      soql: Full SOQL string (e.g., "SELECT Id, Name FROM Account WHERE Name LIKE 'Acme%'")
      limit: If the SOQL has no LIMIT and paginate=False, append this LIMIT client-side.
      paginate: If True, follow nextRecordsUrl to fetch additional batches up to max_rows.
      max_rows: Max rows to fetch when paginate=True (safety cap).
      strip_attributes: Remove the Salesforce 'attributes' key from each record.

    Returns:
      {
        "soql": "<final SOQL used>",
        "totalSize": <int>,          # server-side count for the full query
        "returned": <int>,           # number of records included in 'records'
        "done": <bool>,              # True if server reported done OR we paginated to completion
        "records": [ {...}, ... ]    # list of result rows
      }
      or on error:
      {
        "soql": "<original soql>",
        "error": "message"
      }
    """
    try:
        sf = helpers.get_sf_connection()

        # Append LIMIT if not present and we're not paginating
        final_soql = soql.strip()
        if (not paginate) and (" limit " not in final_soql.lower()):
            final_soql = f"{final_soql} LIMIT {int(limit)}"

        first = sf.query(final_soql)
        total_size = first.get("totalSize", 0)
        records = list(first.get("records", []) or [])
        done = bool(first.get("done", True))

        # Paginate if asked
        if paginate and not done and records is not None:
            while not done and len(records) < int(max_rows):
                next_url = first.get("nextRecordsUrl")
                if not next_url:
                    break
                nxt = sf.query_more(next_url, True)
                recs = nxt.get("records", []) or []
                if not recs:
                    done = True
                    break
                records.extend(recs)
                first = nxt
                done = bool(nxt.get("done", True))
                if len(records) >= int(max_rows):
                    done = False  # we stopped early due to cap

        if strip_attributes:
            records = _strip_attrs(records)

        return {
            "soql": final_soql,
            "totalSize": total_size,
            "returned": len(records),
            "done": done,
            "records": records,
        }

    except SalesforceMalformedRequest as e:
        return {
            "soql": soql,
            "error": f"MALFORMED_QUERY: {getattr(e, 'content', e)}"
        }
    except Exception as e:
        return {
            "soql": soql,
            "error": str(e)
        }




@mcp.tool
def fetch_salesforce_entity_details(id_or_name: str) -> dict:
    """
    Fetch a SINGLE Account or Opportunity by Salesforce Id. If you do not know the id do not call this tool.

    Input:
      - id: 15/18-char Salesforce Id.
        Supported prefixes:
          • 001… → Account
          • 006… → Opportunity

    Behavior:
      - Determines object from Id prefix; rejects non-001/006 prefixes.
      - SELECT fields come from core_schema.detail_select_list(<object>) + (Id, Name).
      - If object is Opportunity and Owner.Name isn't in the returned record, the tool
        fetches Owner.Name via OwnerId and injects it into the output.

    Output:
      {
        "object": "Account|Opportunity",
        "input": "<original>",
        "resolved": { "id": "<Id>", "name": "<Name>", "match_score": null, "matched_via": "ID" },
        "select": ["Id","Name","..."],   # actual SOQL expressions
        "soql": "SELECT ... FROM ... WHERE Id='...' LIMIT 1",
        "record": { "Id": "...", "Name": "...", "Owner.Name": "...", ... }  # flattened
      }
      or { "error": "...", "input": "<...>" }
    """
    
    return helpers.fetch_entity_details_tool(id_or_name)


@mcp.tool
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

    return helpers.llm_pick_best_name_matches_tool(query, k, max_per_type, shortlist_cap, force)



if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8200)  # run 'fastmcp run main.py --transport sse --port 8100'