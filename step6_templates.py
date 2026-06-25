from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def create_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )

# --- Templates ---

explanation_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful academic tutor."),
    ("human", "Explain the topic '{topic}' for a {level} student in a clear and structured way.")
])

quiz_template = ChatPromptTemplate.from_messages([
    ("system", "You are an academic tutor that generates quizzes."),
    ("human", "Generate a {num_questions}-question multiple choice quiz on '{topic}'. Include answers at the end.")
])

revision_template = ChatPromptTemplate.from_messages([
    ("system", "You are an academic tutor that creates revision notes."),
    ("human", "Create concise revision notes for '{topic}'. Include key points, definitions, and important formulas or facts.")
])

study_plan_template = ChatPromptTemplate.from_messages([
    ("system", "You are a study coach that creates personalized study plans."),
    ("human", "Create a {duration}-day study plan for a student preparing for an exam on '{topic}'. Break it into daily goals.")
])

# --- Runners ---

def run_explanation(llm):
    topic = input("Enter topic: ").strip()
    level = input("Enter level (beginner / intermediate / advanced): ").strip()
    chain = explanation_template | llm
    response = chain.invoke({"topic": topic, "level": level})
    print(f"\n--- Explanation ---\n{response.content}\n")

def run_quiz(llm):
    topic = input("Enter topic: ").strip()
    num = input("Number of questions (e.g. 3): ").strip()
    chain = quiz_template | llm
    response = chain.invoke({"topic": topic, "num_questions": num})
    print(f"\n--- Quiz ---\n{response.content}\n")

def run_revision(llm):
    topic = input("Enter topic: ").strip()
    chain = revision_template | llm
    response = chain.invoke({"topic": topic})
    print(f"\n--- Revision Notes ---\n{response.content}\n")

def run_study_plan(llm):
    topic = input("Enter topic: ").strip()
    duration = input("Study duration in days (e.g. 7): ").strip()
    chain = study_plan_template | llm
    response = chain.invoke({"topic": topic, "duration": duration})
    print(f"\n--- Study Plan ---\n{response.content}\n")

def run():
    llm = create_llm()

    print("=== Prompt Templates - Academic Tutor ===")
    print("Modes: 1) Explanation  2) Quiz  3) Revision Notes  4) Study Plan  5) Quit\n")

    actions = {
        "1": run_explanation,
        "2": run_quiz,
        "3": run_revision,
        "4": run_study_plan
    }

    while True:
        choice = input("Select mode (1/2/3/4/5): ").strip()
        if choice == "5":
            break
        if choice not in actions:
            print("Invalid choice.\n")
            continue
        actions[choice](llm)

if __name__ == "__main__":
    run()