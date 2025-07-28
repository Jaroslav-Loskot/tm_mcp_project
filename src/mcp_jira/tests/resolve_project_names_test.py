import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mcp_jira.helpers import _resolve_project_name, get_all_jira_projects

test_cases = [
    {"input": "AKB", "expected_name": "AK Bank", "expected_key": "AKB"},
    {"input": "BWG", "expected_name": "BAWAG - AppSupport", "expected_key": "ASBWG"},
    {"input": "Admin", "expected_name": "Administrative Tasks", "expected_key": "ADM"},
    {"input": "Discovery desk", "expected_name": "Product Discovery", "expected_key": "PRD"},
    {"input": "unicredit italy", "expected_name": "UniCredit Italy - AppSupport"},
    {"input": "croatia service desk", "expected_name": "UniCredit Croatia - AppSupport"},
    {"input": "germany uncredit", "expected_name": "UniCredit Germany - AppSupport"},
    {"input": "austria support", "expected_name": "UniCredit Austria - AppSupport"},
    {"input": "german desk", "expected_name": "UniCredit Germany - AppSupport"},
    {"input": "croatia bank", "expected_name": "UniCredit Croatia - AppSupport"},
    {"input": "support italy", "expected_name": "UniCredit Italy - AppSupport"},
]


def test_resolve_project_names():
    from mcp_jira.helpers import _resolve_project_name  # adjust if path differs

    passed = 0
    failed = 0

    for case in test_cases:
        user_input = case["input"]
        expected = case["expected_name"]
        try:
            result = _resolve_project_name(user_input)
            if not result:
                print(f"❌ NO RESULT for input: '{user_input}' (expected: {expected})")
                failed += 1
            elif expected in [r["name"] for r in result]:
                print(f"✅ PASS: '{user_input}' → matched '{expected}'")
                passed += 1
            else:
                print(f"❌ FAIL: '{user_input}' → got {[r['name'] for r in result]}, expected '{expected}'")
                failed += 1
        except Exception as e:
            print(f"❌ EXCEPTION for '{user_input}': {e}")
            failed += 1

    print(f"\nSummary: ✅ {passed} passed, ❌ {failed} failed")



test_resolve_project_names()
