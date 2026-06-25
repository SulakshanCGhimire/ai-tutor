from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
import math

load_dotenv()

def create_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )

# --- Tools ---

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    Use this for any math calculations, arithmetic, or numerical problems.
    Example input: '2 + 2', '15 * 8', 'sqrt(144)', '(10 + 5) / 3'
    """
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed.update({"abs": abs, "round": round, "pow": pow})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool
def study_planner(topic_and_days: str) -> str:
    """
    Creates a study plan for a given topic and number of days.
    Input format: 'topic|days' — Example: 'Python Programming|5'
    Use this when a student asks for a study plan, schedule, or how to prepare for an exam.
    """
    try:
        parts = topic_and_days.split("|")
        topic = parts[0].strip()
        days = int(parts[1].strip()) if len(parts) > 1 else 3

        plan = [f"📅 {days}-Day Study Plan for: {topic}\n"]
        sections = [
            "Introduction & Core Concepts",
            "Deep Dive & Key Theories",
            "Practice Problems & Examples",
            "Review & Weak Areas",
            "Mock Tests & Revision",
            "Final Review & Rest",
            "Exam Preparation Tips"
        ]

        for day in range(1, days + 1):
            section = sections[(day - 1) % len(sections)]
            plan.append(f"Day {day}: {section}")

        plan.append("\n✅ Tip: Review previous day's notes before starting each session.")
        return "\n".join(plan)
    except Exception as e:
        return f"Error generating study plan: {str(e)}"

@tool
def summarizer(text: str) -> str:
    """
    Summarizes a long piece of text into key points.
    Use this when a student provides a passage, paragraph, or notes and wants a summary.
    Input: any text to summarize.
    """
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    total = len(sentences)

    if total <= 3:
        return f"Summary:\n" + "\n".join(f"• {s}." for s in sentences)

    step = max(1, total // 4)
    key = [sentences[i] for i in range(0, total, step)][:4]
    return "Summary (Key Points):\n" + "\n".join(f"• {s}." for s in key)

# --- Agent ---

def build_agent(llm):
    tools = [calculator, study_planner, summarizer]

    system_prompt = (
        "You are an intelligent academic tutor agent. "
        "You have access to three tools:\n"
        "1. calculator — for any math or numerical calculations\n"
        "2. study_planner — for creating study schedules (input format: 'topic|days')\n"
        "3. summarizer — for summarizing long text or notes\n\n"
        "Always use the appropriate tool when the task matches. "
        "After using a tool, explain the result to the student clearly."
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )
    return agent

def run():
    llm = create_llm()
    agent = build_agent(llm)

    print("=== AI Tutor Agent ===")
    print("I can help with:")
    print("  🔢 Math calculations")
    print("  📅 Study planning")
    print("  📝 Text summarization")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        try:
            response = agent.invoke({
                "messages": [HumanMessage(content=user_input)]
            })
            final = response["messages"][-1].content
            print(f"\nAgent: {final}\n")
        except Exception as e:
            print(f"\nError: {str(e)}\n")

if __name__ == "__main__":
    run()