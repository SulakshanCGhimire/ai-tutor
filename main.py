from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import math

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────

def create_llm(temperature=0.7):
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)

# ── MEMORY ───────────────────────────────────────────────────────────────────

class TutorMemory:
    def __init__(self, user_name: str, subject: str):
        self.user_name = user_name
        self.subject = subject
        self.topics_studied = []
        self.chat_history = []

    def add_interaction(self, user_msg: str, ai_msg: str):
        self.chat_history.append(HumanMessage(content=user_msg))
        self.chat_history.append(AIMessage(content=ai_msg))

    def add_topic(self, topic: str):
        if topic not in self.topics_studied:
            self.topics_studied.append(topic)

    def get_system_prompt(self) -> str:
        topics = ", ".join(self.topics_studied) if self.topics_studied else "none yet"
        return (
            f"You are a personalized academic tutor. "
            f"Student name: {self.user_name}. Subject: {self.subject}. "
            f"Topics already covered: {topics}. "
            f"Always address student by name. Build on previous topics when relevant."
        )

    def get_full_history(self) -> list:
        return [SystemMessage(content=self.get_system_prompt())] + self.chat_history

    def show_summary(self):
        print(f"\n--- Session Summary for {self.user_name} ---")
        print(f"Subject : {self.subject}")
        print(f"Topics  : {', '.join(self.topics_studied) if self.topics_studied else 'none'}")
        print(f"Exchanges: {len(self.chat_history) // 2}\n")

def extract_topic(text: str) -> str | None:
    keywords = ["explain", "what is", "tell me about", "teach me", "study", "learn about"]
    lower = text.lower()
    for kw in keywords:
        if kw in lower:
            topic = lower.replace(kw, "").strip(" ?.")
            return topic.title() if topic else None
    if len(text.split()) <= 5:
        return text.title()
    return None

# ── TOOLS ─────────────────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression. Input: '2+2', 'sqrt(144)', '(10+5)/3'"""
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed.update({"abs": abs, "round": round, "pow": pow})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def study_planner(topic_and_days: str) -> str:
    """Creates a study plan. Input format: 'topic|days' e.g. 'Python|5'"""
    try:
        parts = topic_and_days.split("|")
        topic = parts[0].strip()
        days = int(parts[1].strip()) if len(parts) > 1 else 3
        sections = [
            "Introduction & Core Concepts",
            "Deep Dive & Key Theories",
            "Practice Problems & Examples",
            "Review & Weak Areas",
            "Mock Tests & Revision",
            "Final Review & Rest",
            "Exam Preparation Tips"
        ]
        plan = [f"📅 {days}-Day Study Plan: {topic}\n"]
        for day in range(1, days + 1):
            plan.append(f"Day {day}: {sections[(day - 1) % len(sections)]}")
        plan.append("\n✅ Tip: Review previous day's notes before each session.")
        return "\n".join(plan)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def summarizer(text: str) -> str:
    """Summarizes a long text into key bullet points."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    total = len(sentences)
    if total <= 3:
        return "Summary:\n" + "\n".join(f"• {s}." for s in sentences)
    step = max(1, total // 4)
    key = [sentences[i] for i in range(0, total, step)][:4]
    return "Summary:\n" + "\n".join(f"• {s}." for s in key)

# ── PROMPTS & CHAINS ──────────────────────────────────────────────────────────

ROLES = {
    "1": ("Teacher",      "You are a patient teacher. Explain clearly with examples. Address student encouragingly."),
    "2": ("Examiner",     "You are a strict examiner. Ask tough questions one at a time. Score answers out of 10."),
    "3": ("Study Coach",  "You are an energetic study coach. Give motivation, strategies, and study tips."),
    "4": ("Subject Expert","You are a subject expert. Give precise, technical, advanced answers.")
}

explanation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful academic tutor."),
    ("human", "Explain '{topic}' for a {level} student in a clear and structured way.")
])

quiz_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an academic tutor that generates quizzes."),
    ("human", "Generate a {num_questions}-question MCQ quiz on '{topic}'. Include answers at the end.")
])

notes_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an academic tutor that creates revision notes."),
    ("human", "Create concise revision notes for '{topic}'. Include key points, definitions, and formulas.")
])

cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an academic tutor. Always solve problems step-by-step before giving a final answer."),
    ("human", "Solve step-by-step:\nProblem: {problem}")
])

# ── FEATURE MODULES ───────────────────────────────────────────────────────────

def module_chat(llm, memory):
    print("\n[Chat] Type 'back' to return to menu\n")
    while True:
        user_input = input(f"{memory.user_name}: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "back":
            break
        topic = extract_topic(user_input)
        if topic:
            memory.add_topic(topic)
        messages = memory.get_full_history()
        messages.append(HumanMessage(content=user_input))
        response = llm.invoke(messages)
        memory.add_interaction(user_input, response.content)
        print(f"\nTutor: {response.content}\n")

def module_zero_shot(llm):
    print("\n[Zero-Shot] 1)Explain  2)Summarize  3)Simplify  4)Back")
    parser = StrOutputParser()
    while True:
        choice = input("Mode: ").strip()
        if choice == "4":
            break
        topic = input("Topic: ").strip()
        if not topic:
            continue
        if choice == "1":
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an academic tutor."),
                ("human", "Explain this topic in detail: {topic}")
            ])
        elif choice == "2":
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an academic tutor."),
                ("human", "Summarize this topic in 5 bullet points: {topic}")
            ])
        elif choice == "3":
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an academic tutor."),
                ("human", "Explain this topic in very simple language a 10-year-old can understand: {topic}")
            ])
        else:
            print("Invalid.")
            continue
        print("\n" + (prompt | llm | parser).invoke({"topic": topic}) + "\n")

def module_few_shot(llm):
    print("\n[Few-Shot] 1)Quiz  2)Structured Answer  3)Back")
    parser = StrOutputParser()
    while True:
        choice = input("Mode: ").strip()
        if choice == "3":
            break
        if choice == "1":
            topic = input("Quiz topic: ").strip()
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You generate quizzes in strict MCQ format."),
                ("human", f"""Follow this format exactly:
Q1: <question>
A) ... B) ... C) ... D) ...
Answer: <letter>

Generate a 3-question quiz on: {topic}""")
            ])
            print("\n" + (prompt | llm | parser).invoke({}) + "\n")
        elif choice == "2":
            question = input("Your question: ").strip()
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You answer questions in structured format."),
                ("human", f"""Answer using this format:
Definition: ...
Example: ...
Key Term: ...

Question: {question}""")
            ])
            print("\n" + (prompt | llm | parser).invoke({}) + "\n")

def module_cot(llm):
    print("\n[Chain-of-Thought] Type 'back' to return\n")
    parser = StrOutputParser()
    while True:
        problem = input("Problem: ").strip()
        if problem.lower() == "back":
            break
        print("\n" + (cot_prompt | llm | parser).invoke({"problem": problem}) + "\n")

def module_roles(llm):
    print("\n[Role Prompting]")
    for k, (name, _) in ROLES.items():
        print(f"  {k}) {name}")
    print("  5) Back\n")
    while True:
        choice = input("Select role: ").strip()
        if choice == "5":
            break
        if choice not in ROLES:
            print("Invalid.")
            continue
        name, system = ROLES[choice]
        print(f"\n--- {name} activated --- (type 'switch' or 'quit')\n")
        history = [SystemMessage(content=system)]
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "quit":
                return
            if user_input.lower() == "switch":
                break
            if not user_input:
                continue
            history.append(HumanMessage(content=user_input))
            response = llm.invoke(history)
            history.append(AIMessage(content=response.content))
            print(f"\n{name}: {response.content}\n")

def module_templates(llm):
    print("\n[Templates] 1)Explanation  2)Quiz  3)Notes  4)Study Plan  5)Back")
    parser = StrOutputParser()
    while True:
        choice = input("Mode: ").strip()
        if choice == "5":
            break
        topic = input("Topic: ").strip()
        if not topic:
            continue
        if choice == "1":
            level = input("Level (beginner/intermediate/advanced): ").strip()
            result = (explanation_prompt | llm | parser).invoke({"topic": topic, "level": level})
        elif choice == "2":
            num = input("Number of questions: ").strip()
            result = (quiz_prompt | llm | parser).invoke({"topic": topic, "num_questions": num})
        elif choice == "3":
            result = (notes_prompt | llm | parser).invoke({"topic": topic})
        elif choice == "4":
            days = input("Duration (days): ").strip()
            plan_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a study coach."),
                ("human", "Create a {duration}-day study plan for '{topic}'. Break into daily goals.")
            ])
            result = (plan_prompt | llm | parser).invoke({"topic": topic, "duration": days})
        else:
            print("Invalid.")
            continue
        print(f"\n{result}\n")

def module_chains(llm):
    print("\n[Chains] 1)Explain only  2)Explain→Notes  3)Full Pipeline  4)Back")
    parser = StrOutputParser()
    exp_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful academic tutor."),
        ("human", "Explain '{topic}' clearly for a student.")
    ])
    n_prompt = ChatPromptTemplate.from_messages([
        ("system", "You create revision notes."),
        ("human", "Create revision notes from this explanation:\n\n{explanation}")
    ])
    q_prompt = ChatPromptTemplate.from_messages([
        ("system", "You generate quizzes."),
        ("human", "Generate a 3-question MCQ quiz from these notes:\n\n{notes}")
    ])
    while True:
        choice = input("Mode: ").strip()
        if choice == "4":
            break
        topic = input("Topic: ").strip()
        if not topic:
            continue
        print("\n⏳ Generating explanation...")
        explanation = (exp_prompt | llm | parser).invoke({"topic": topic})
        print(f"\n--- Explanation ---\n{explanation}\n")
        if choice in ["2", "3"]:
            print("⏳ Generating notes...")
            notes = (n_prompt | llm | parser).invoke({"explanation": explanation})
            print(f"\n--- Notes ---\n{notes}\n")
            if choice == "3":
                print("⏳ Generating quiz...")
                quiz = (q_prompt | llm | parser).invoke({"notes": notes})
                print(f"\n--- Quiz ---\n{quiz}\n")

def module_agent(llm):
    print("\n[Agent] Math | Study Plan | Summarize — type 'back' to return\n")
    tools = [calculator, study_planner, summarizer]
    system = (
        "You are an academic tutor agent with tools: "
        "calculator (math), study_planner (input: 'topic|days'), summarizer (text). "
        "Use the right tool automatically."
    )
    agent = create_react_agent(model=llm, tools=tools, prompt=system)
    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() == "back":
            break
        try:
            response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
            print(f"\nAgent: {response['messages'][-1].content}\n")
        except Exception as e:
            print(f"Error: {e}\n")

# ── MAIN MENU ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("   AI-Powered Academic Tutor")
    print("=" * 50)

    user_name = input("\nYour name: ").strip() or "Student"
    subject = input("Subject: ").strip() or "General"
    llm = create_llm()
    memory = TutorMemory(user_name, subject)

    print(f"\nWelcome, {user_name}! Subject: {subject}\n")

    menu = {
        "1": ("💬 Personal Chat (with memory)",    lambda: module_chat(llm, memory)),
        "2": ("🎯 Zero-Shot Prompting",             lambda: module_zero_shot(llm)),
        "3": ("📋 Few-Shot Prompting",              lambda: module_few_shot(llm)),
        "4": ("🧠 Chain-of-Thought Solving",        lambda: module_cot(llm)),
        "5": ("🎭 Role-Based Tutor",                lambda: module_roles(llm)),
        "6": ("📄 Prompt Templates",                lambda: module_templates(llm)),
        "7": ("🔗 LangChain Chains",                lambda: module_chains(llm)),
        "8": ("🤖 AI Agent & Tools",                lambda: module_agent(llm)),
        "9": ("📊 Session Summary",                 lambda: memory.show_summary()),
        "0": ("🚪 Exit",                            None)
    }

    while True:
        print("\n--- Main Menu ---")
        for k, (label, _) in menu.items():
            print(f"  {k}) {label}")

        choice = input("\nSelect: ").strip()

        if choice == "0":
            memory.show_summary()
            print(f"\nGoodbye, {user_name}! Keep learning! 🚀\n")
            break
        elif choice in menu:
            _, action = menu[choice]
            action()
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()