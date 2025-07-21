import json
import logging
import re
from datetime import date
from fastapi import HTTPException

from mcp_common.utils.bedrock_wrapper import call_claude



# Optional utility function (so it's reusable/testable)
def get_today() -> date:
    return date.today()

def parse_time_range_to_bounds(input_str: str) -> dict:
    """
    Uses LLM to convert a time range expression into structured time_from and time_to values (YYYY-MM-DD format).
    """

    today_str = str(get_today())
    logging.info(f"[Time Range Parsing] Today is: {today_str}")

    system_prompt = f"""You are a helpful assistant converting human-readable time range expressions into structured date ranges.

Today's date is: {today_str}

Your task is to return a JSON object with the following fields:
- "time_from": the start date of the range in YYYY-MM-DD format, or null if not determinable
- "time_to": the end date of the range in YYYY-MM-DD format, or null if not determinable

You must infer the correct values from expressions like:
- "last two weeks"
- "-1w"
- "2025-01-12 to 2025-01-20"
- "newer than 2025 Jun"
- "before 2024-01-01"

Use today's date if needed for relative expressions. Return ONLY the JSON object.
"""

    response_text = call_claude(system_prompt=system_prompt, user_input=input_str)

    # Strip markdown-style fences if present
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if fenced_match:
        response_text = fenced_match.group(1)

    try:
        result = json.loads(response_text)

        # Validate keys
        time_from = result.get("time_from")
        time_to = result.get("time_to")

        def is_valid_date(date_str):
            try:
                if date_str is None:
                    return True
                date.fromisoformat(date_str)
                return True
            except Exception:
                return False

        if not (is_valid_date(time_from) and is_valid_date(time_to)):
            raise ValueError("Invalid date format in Claude output")

        return {"time_from": time_from, "time_to": time_to}

    except Exception as e:
        logging.error(f"Failed to parse Claude time range: {response_text}")
        raise HTTPException(status_code=500, detail=f"Failed to interpret time range: {str(e)}")
