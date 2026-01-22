import json
import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

llm = OllamaLLM(model="gemma:2b", temperature=0)

def safe_parse(text, key, allowed=None):
    try:
        match = re.search(rf'"{key}"\s*:\s*"([^"]+)"', text)
        if match:
            value = match.group(1)
            if allowed:
                return value if value in allowed else allowed[0]
            return value
    except:
        pass
    return "" if not allowed else allowed[0]

def run(prompt, data, key, allowed=None):
    chain = prompt | llm
    response = chain.invoke(data)
    return {key: safe_parse(response, key, allowed)}

cleaner_prompt = ChatPromptTemplate.from_template("""
Return JSON with key "cleaned_text" only. No extra text.

{{"cleaned_text":"..."}}

TEXT:
{text}
""")

intent_prompt = ChatPromptTemplate.from_template("""
Return JSON with key "intent" only. No extra text.

{{"intent":"..."}}

TEXT:
{text}
""")

category_prompt = ChatPromptTemplate.from_template("""
Return JSON with key "category" only. Choose ONE of:
Billing, Technical, Sales, HR, General.

{{"category":"..."}}

TEXT:
{text}
INTENT:
{intent}
""")

priority_prompt = ChatPromptTemplate.from_template("""
Return JSON with key "priority" only. Choose ONE of:
High, Medium, Low.

{{"priority":"..."}}

TEXT:
{text}
CATEGORY:
{category}
""")

def multi_agent_pipeline(text):
    cleaned = run(cleaner_prompt, {"text": text}, "cleaned_text")
    intent = run(intent_prompt, {"text": cleaned["cleaned_text"]}, "intent")
    category = run(category_prompt, {"text": cleaned["cleaned_text"], "intent": intent["intent"]}, "category",
                   ["Billing", "Technical", "Sales", "HR", "General"])
    priority = run(priority_prompt, {"text": cleaned["cleaned_text"], "category": category["category"]}, "priority",
                   ["High", "Medium", "Low"])
    return {
        "cleaned_text": cleaned["cleaned_text"],
        "intent": intent["intent"],
        "category": category["category"],
        "priority": priority["priority"]
    }

if __name__ == "__main__":
    sample = """
    Hello Support Team,

    I have some issue in my computer system,
    please help me to figure this out.

    Thanks,
    Rahul
    """
    print(json.dumps(multi_agent_pipeline(sample), indent=2))




# OUTPUT: 1
# {
#   "cleaned_text": "Hello Support Team, My payment failed twice and my account is locked. Please resolve urgently. Thanks, Rahul",
#   "intent": "failed_payment_attempts",
#   "category": "Billing",
#   "priority": "High"
# }


# OUTPUT: 2
# {
#   "cleaned_text": "Hello Support Team, I have some issue in my computer system, please help me to figure this out. Thanks, Rahul",
#   "intent": "help",
#   "category": "Technical",
#   "priority": "Medium"
# }
