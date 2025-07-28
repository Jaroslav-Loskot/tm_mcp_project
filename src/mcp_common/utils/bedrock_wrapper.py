import json
import logging
import os
from typing import Dict, List

import boto3
from dotenv import load_dotenv
from fastapi import HTTPException

from dotenv import load_dotenv
from pathlib import Path



load_dotenv(Path(__file__).resolve().parents[3] / ".env")


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

CLAUDE_MODEL_ID = os.getenv("CLAUDE_MODEL_ID")
NOVA_LITE_MODEL_ID = os.getenv("NOVA_LITE_MODEL_ID")

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

def call_claude(system_prompt: str, user_input: str) -> str:
    result = call_llm(CLAUDE_MODEL_ID, system_prompt, user_input)
    if result is None:
        raise ValueError("Claude LLM returned None")
    return result



def call_nova_lite(user_prompt: str) -> str:
    try:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": user_prompt
                        }
                    ]
                }
            ]
        }

        response = bedrock_client.invoke_model(
            modelId=NOVA_LITE_MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )

        raw = response["body"].read()
        response_body = json.loads(raw)

        # Extract assistant message
        message = response_body.get("output", {}).get("message", {})
        content_blocks = message.get("content", [])

        for block in content_blocks:
            text = block.get("text", "")
            if text:
                # Strip triple backticks and optional 'json' language tag
                cleaned = (
                    text.strip()
                    .removeprefix("```json")
                    .removeprefix("```")
                    .removesuffix("```")
                    .strip()
                )
                return cleaned

        raise ValueError("No assistant text found in Nova response.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nova Lite request failed: {str(e)}")





# --- Claude Generation via signed HTTP request ---
def call_llm(modelId: str, system_prompt: str, user_input: str) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.7,
        "system": system_prompt,  
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_input}]}
        ],
    }

    try:
        response = bedrock_client.invoke_model(
            modelId=modelId,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        raw = response["body"].read().decode()
        parsed = json.loads(raw)
        return parsed["content"][0]["text"].strip()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude request failed: {str(e)}")


# --- Titan Embedding ---
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def fetch_embedding(text: str) -> list[float]:
    """
    Fetch embedding using Amazon Titan model.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty.")

    try:
        payload = {"inputText": text}
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        body = response["body"].read().decode()
        logging.info(f"Bedrock response body: {body}")
        result = json.loads(body)

        embedding = result.get("embedding")
        if not embedding or not isinstance(embedding, list):
            logging.error(f"Invalid embedding structure: {result}")
            raise HTTPException(
                status_code=500, detail="Embedding response invalid or missing."
            )

        return embedding

    except Exception as e:
        logging.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Embedding generation failed: {str(e)}"
        )
