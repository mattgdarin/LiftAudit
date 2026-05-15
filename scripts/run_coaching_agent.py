#!/usr/bin/env python3
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
    ))
    messages = state["messages"]
    print(f"\nCoach: {messages[-1].content}\n")
