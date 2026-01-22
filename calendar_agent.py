from langchain_ollama import OllamaLLM
from datetime import datetime
import uuid
import json


def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


class CalendarStore:
    def __init__(self):
        self.events = []

    def add_event(self, title, start, end):
        event = {
            "id": str(uuid.uuid4()),
            "title": title,
            "start": start,
            "end": end
        }
        self.events.append(event)
        return event

    def list_events(self):
        return self.events

    def is_available(self, start, end):
        for event in self.events:
            if start < event["end"] and end > event["start"]:
                return False, event
        return True, None


class BaseAgent:
    def __init__(self, llm, calendar):
        self.llm = llm
        self.calendar = calendar


class IntentAgent(BaseAgent):
    def run(self, user_input):
        prompt = f"""
        Identify the intent.
        Choose one:
        - schedule_event
        - check_availability
        - list_events

        Input:
        {user_input}

        Respond with ONLY the intent name.
        """
        response = self.llm.invoke(prompt)
        return response.strip().lower()


class TimeAgent(BaseAgent):
    def run(self, user_input):
        prompt = f"""
        Extract event title, start time, and end time.

        Rules:
        - Output ONLY valid JSON
        - Use ISO format: YYYY-MM-DD HH:MM
        - If unsure, return {{}}

        Input:
        {user_input}

        Output:
        {{
        "title": "...",
        "start": "...",
        "end": "..."
        }}
        """
        response = self.llm.invoke(prompt)
        return response.strip()


class SchedulerAgent(BaseAgent):
    def run(self, title, start, end):
        available, conflict = self.calendar.is_available(start, end)
        if not available:
            return f"❌ Conflict with '{conflict['title']}' from {conflict['start']} to {conflict['end']}"
        event = self.calendar.add_event(title, start, end)
        return f"✅ Event scheduled: {event['title']} ({start} → {end})"


class AvailabilityAgent(BaseAgent):
    def run(self, start, end):
        available, conflict = self.calendar.is_available(start, end)
        if available:
            return "✅ You are available during this time."
        return f"❌ Not available. Conflict with '{conflict['title']}'."


class ResponseAgent(BaseAgent):
    def run(self, text):
        prompt = f"""
        Rewrite this response:

        {text}
        """
        response = self.llm.invoke(prompt)
        return response.strip()


class CalendarAgent:
    def __init__(self, llm):
        self.calendar = CalendarStore()
        self.intent_agent = IntentAgent(llm, self.calendar)
        self.time_agent = TimeAgent(llm, self.calendar)
        self.scheduler_agent = SchedulerAgent(llm, self.calendar)
        self.availability_agent = AvailabilityAgent(llm, self.calendar)
        self.response_agent = ResponseAgent(llm, self.calendar)

    def handle(self, user_input):
        intent = self.intent_agent.run(user_input)

        if intent == "list_events":
            events = self.calendar.list_events()
            if not events:
                return "📭 No events scheduled."
            return "\n".join(
                f"{e['title']} ({e['start']} → {e['end']})"
                for e in events
            )

        if intent in ["schedule_event", "check_availability"]:
            raw = self.time_agent.run(user_input)
            parsed = extract_json(raw)

            if not parsed:
                return "❌ I couldn't understand the date and time."

            try:
                start = datetime.fromisoformat(parsed["start"])
                end = datetime.fromisoformat(parsed["end"])
            except Exception:
                return "❌ Invalid date format."

            if intent == "check_availability":
                return self.response_agent.run(
                    self.availability_agent.run(start, end)
                )

            if intent == "schedule_event":
                return self.response_agent.run(
                    self.scheduler_agent.run(
                        parsed.get("title", "Event"), start, end
                    )
                )

        return "❓ I couldn't understand your request."


def main():
    llm = OllamaLLM(model="gemma:2b", temperature=0)
    agent = CalendarAgent(llm)

    print("\n📅 Calendar Agent Ready (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = agent.handle(user_input)
        print("Agent:", response)


if __name__ == "__main__":
    main()





# # OUTPUT
# 📅 Calendar Agent Ready (type 'exit' to quit)

# You:  Schedule for an event on 23 January at 4 PM
# Agent: Sure, here is the rewritten response:

# ✅ Event scheduled: 2023-01-23 16:00:00 - 2023-01-23 17:00:00
# You: list events
# Agent: ... (2023-01-23 16:00:00 → 2023-01-23 17:00:00)
# You: check avaibality on 24 january
# Agent: Sure, here is the rewritten response:

# ✅ You are available during this time. Please let me know if you have any questions or need assistance.
# You: check availability at 23 january
# Agent: Sure, here is the rewritten response:

# ✅ You are available during this time. Please let me know if you have any questions or need assistance.
# You: book another slot on 23 january at 8 PM
# Agent: Sure, here is the rewritten response:

# ✅ Event scheduled: 2023-01-23 20:00:00 - 2023-01-23 20:30:00
# You: list events
# Agent: ... (2023-01-23 16:00:00 → 2023-01-23 17:00:00)
# ... (2023-01-23 20:00:00 → 2023-01-23 20:30:00)
# You: exit