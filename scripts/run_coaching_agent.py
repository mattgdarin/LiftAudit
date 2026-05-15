#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage
from liftaudit.coaching.agent import State, get_coaching_agent

agent = get_coaching_agent()

print("LiftAudit Coaching Agent — type 'quit' to exit\n")

messages = []
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        break
    if not user_input:
        continue

    messages.append(HumanMessage(content=user_input))
    state = agent.invoke(State(
        messages=messages,
        preferences="",
        strengths=[],
        weaknesses=[],
        active_lifts=[],
        plan=None,
    ))
    messages = state["messages"]
    content = messages[-1].content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    print(f"\nCoach: {content}\n")
