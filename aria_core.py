import json
import os
import tiktoken
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 1. Output Schema Definition
class AriaResponse(BaseModel):
    intent: str = Field(description="Primary intent: query, action_request, clarification, or chit_chat")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    reply: str = Field(description="Direct response to the user")

# 2. Dynamic Context Budgeting
def count_tokens(text: str) -> int:
    # Use cl100k_base as a reliable local approximation for token length
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def build_context(system_prompt: str, history: list, new_user_msg: str, max_tokens: int = 2000) -> list:
    """
    Guarantees system prompt + latest query are preserved.
    Prunes the oldest dialogue turns when over token budget.
    """
    sys_turn = {"role": "system", "content": system_prompt}
    new_turn = {"role": "user", "content": new_user_msg}
    
    current_tokens = count_tokens(system_prompt) + count_tokens(new_user_msg)
    selected_history = []

    # Traverse from most recent turns backwards
    for turn in reversed(history):
        turn_tokens = count_tokens(turn["content"])
        if current_tokens + turn_tokens <= max_tokens:
            selected_history.insert(0, turn)
            current_tokens += turn_tokens
        else:
            break

    return [sys_turn] + selected_history + [new_turn]

# 3. Model Invocation with JSON Schema Validation
def call_aria(messages: list) -> AriaResponse:
    # Groq uses json_object mode; we request strict JSON conforming to the Pydantic schema
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    
    usage = completion.usage
    print(f"\n[Tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out | Total: {usage.total_tokens}]")
    
    raw_text = completion.choices[0].message.content
    data = json.loads(raw_text)
    return AriaResponse(**data)