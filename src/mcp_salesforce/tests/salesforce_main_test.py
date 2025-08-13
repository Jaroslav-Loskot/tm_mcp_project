import os
import re
import json
from typing import Any, Dict
from dotenv import load_dotenv
from simple_salesforce import Salesforce, SalesforceMalformedRequest

import mcp_salesforce.agent_generate_SOQL as sf_agent
import mcp_salesforce.helpers as helpers 

# -------------------- Parsing helpers --------------------
def _extract_json_from_agent(last_content: Any) -> Dict[str, Any]:
    """
    Normalize the agent output to a dict. Handles:
    - dict already
    - JSON string (with/without fences)
    - Mixed text like <thinking>...</thinking> then JSON
    - list-of-blocks with a JSON 'text' block
    """
    if isinstance(last_content, dict):
        return last_content

    if isinstance(last_content, list):
        for block in last_content:
            if isinstance(block, dict) and ("soql" in block or "comment" in block):
                return block
            if isinstance(block, dict) and block.get("type") == "text":
                try:
                    return json.loads(block["text"])
                except Exception:
                    continue
        raise ValueError(f"Could not parse JSON from list blocks: {last_content}")

    if isinstance(last_content, str):
        s = last_content.strip()
        # Remove <thinking>...</thinking> or any simple tags
        s = re.sub(r"<[^>]+>.*?</[^>]+>\s*", "", s, flags=re.DOTALL)
        # Peel off fenced code
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        # Try straight JSON
        try:
            return json.loads(s)
        except Exception:
            pass
        # Extract first balanced {...}
        start = s.find("{")
        if start == -1:
            raise ValueError("No JSON object start '{' found in agent output.")
        depth, end = 0, None
        for i, ch in enumerate(s[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            raise ValueError("Unbalanced JSON braces in agent output.")
        return json.loads(s[start:end])

    raise ValueError(f"Unsupported content type from agent: {type(last_content)}")


def _try_count_query(sf: Salesforce, soql: str):
    """
    If the query is a simple SELECT (no GROUP BY), try to run a COUNT() version.
    Returns an integer or None.
    """
    lower = soql.lower()
    if " group by " in lower:
        return None

    m = re.match(
        r"^\s*select\s+.+?\s+from\s+([a-zA-Z0-9_]+)\s*(where\s+.+?)?(order\s+by\s+.+?)?$",
        soql, flags=re.IGNORECASE | re.DOTALL
    )
    if not m:
        return None

    obj = m.group(1)
    where = m.group(2) or ""
    count_query = f"SELECT COUNT() FROM {obj} {where}"
    res = sf.query(count_query)
    return res.get("totalSize")


def _strip_attributes(rows):
    return [{k: v for k, v in r.items() if k != "attributes"} for r in rows or []]


# -------------------- Main test runner --------------------
if __name__ == "__main__":
    # ---------- Test A: default mode (returns only final LLM output as string) ----------
    nl_input_a = "List top 10 Opportunities updated last quarter with Id, Name, Amount order by Amount desc"
    print(f"\n🧪 Test A: default (no trace) for:\n  {nl_input_a}")
    out_a = sf_agent.call_agent_generate_soql(nl_input_a)  # returns string by default
    print("\n🔎 Raw LLM output:")
    print(out_a)

    parsed_a = _extract_json_from_agent(out_a)
    soql_a = parsed_a.get("soql", "") or ""
    comment_a = parsed_a.get("comment", "") or parsed_a.get("notes", "")

    print("\n✅ Parsed JSON:")
    print(json.dumps(parsed_a, indent=2, ensure_ascii=False))

    if not soql_a:
        print("\nℹ️ No SOQL returned.")
        if comment_a:
            print(f"Comment: {comment_a}")
    else:
        # Smoke test against Salesforce
        try:
            sf = helpers.get_sf_connection()

            # Ensure runnable (append LIMIT 1 if missing)
            test_query = soql_a if re.search(r"\blimit\s+\d+\b", soql_a, re.IGNORECASE) else f"{soql_a.rstrip()} LIMIT 1"
            print("\n▶️ Smoke test query:")
            print(test_query)
            sample = sf.query(test_query)

            rec = None
            if sample.get("records"):
                rec = _strip_attributes(sample["records"][:1])[0]
            print("\n📌 Sample record:")
            print(rec)

            total = _try_count_query(sf, soql_a)
            if total is not None:
                print(f"\n📈 COUNT() estimate: {total}")
            else:
                print("\n📈 COUNT() estimate skipped (query not compatible).")

            print("\n▶️ Executing as-is (first 5 rows):")
            as_is = sf.query(soql_a)
            print(f"totalSize: {as_is.get('totalSize')}")
            rows = _strip_attributes(as_is.get("records", [])[:5])
            for r in rows:
                print(r)

        except SalesforceMalformedRequest as e:
            print("\n❌ MALFORMED_QUERY")
            print(getattr(e, "content", e))
        except Exception as e:
            print("\n❌ Error executing SOQL:")
            print(e)

    # ---------- Test B: with trace + state bundle ----------
    nl_input_b = "Show me Account Id, Name for accounts created last half a year in US, order by Name asc, limit 10"
    print(f"\n\n🧪 Test B: with_trace + return_state for:\n  {nl_input_b}")
    bundle = sf_agent.call_agent_generate_soql(
        nl_input_b,
        stream=False,
        with_trace=True,
        return_state=True
    )
    print("\n🛰️ Trace:")
    print(bundle["trace"])

    print("\n🔎 Raw LLM output (bundle['output']):")
    print(bundle["output"])

    parsed_b = _extract_json_from_agent(bundle["output"])
    print("\n✅ Parsed JSON:")
    print(json.dumps(parsed_b, indent=2, ensure_ascii=False))

    soql_b = parsed_b.get("soql", "") or ""
    comment_b = parsed_b.get("comment", "") or parsed_b.get("notes", "")

    if not soql_b:
        print("\nℹ️ No SOQL returned.")
        if comment_b:
            print(f"Comment: {comment_b}")
    else:
        try:
            sf = helpers.get_sf_connection()
            test_query = soql_b if re.search(r"\blimit\s+\d+\b", soql_b, re.IGNORECASE) else f"{soql_b.rstrip()} LIMIT 1"
            print("\n▶️ Smoke test query:")
            print(test_query)
            sample = sf.query(test_query)
            rec = None
            if sample.get("records"):
                rec = _strip_attributes(sample["records"][:1])[0]
            print("\n📌 Sample record:")
            print(rec)
        except SalesforceMalformedRequest as e:
            print("\n❌ MALFORMED_QUERY")
            print(getattr(e, "content", e))
        except Exception as e:
            print("\n❌ Error executing SOQL:")
            print(e)

    print("\n✅ Tests completed.")