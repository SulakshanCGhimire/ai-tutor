from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def create_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )

# --- Prompt Templates ---

explanation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful academic tutor."),
    ("human", "Explain the topic '{topic}' clearly and in detail for a student.")
])

notes_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an academic tutor that creates revision notes."),
    ("human", "Based on this explanation, create concise revision notes with key points and definitions:\n\n{explanation}")
])

quiz_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an academic tutor that generates quizzes."),
    ("human", "Based on these revision notes, generate a 3-question multiple choice quiz with answers:\n\n{notes}")
])

# --- Individual Chains ---

def build_chains(llm):
    parser = StrOutputParser()

    explanation_chain = explanation_prompt | llm | parser
    notes_chain = notes_prompt | llm | parser
    quiz_chain = quiz_prompt | llm | parser

    return explanation_chain, notes_chain, quiz_chain

# --- Sequential Runner ---

def run_full_pipeline(llm, topic):
    explanation_chain, notes_chain, quiz_chain = build_chains(llm)

    print("\n⏳ Generating explanation...")
    explanation = explanation_chain.invoke({"topic": topic})
    print(f"\n--- Explanation ---\n{explanation}\n")

    print("⏳ Generating revision notes...")
    notes = notes_chain.invoke({"explanation": explanation})
    print(f"\n--- Revision Notes ---\n{notes}\n")

    print("⏳ Generating quiz...")
    quiz = quiz_chain.invoke({"notes": notes})
    print(f"\n--- Quiz ---\n{quiz}\n")

    return explanation, notes, quiz

def run_single_chain(llm, choice, topic):
    explanation_chain, notes_chain, quiz_chain = build_chains(llm)

    if choice == "1":
        print("\n⏳ Generating explanation...")
        result = explanation_chain.invoke({"topic": topic})
        print(f"\n--- Explanation ---\n{result}\n")

    elif choice == "2":
        print("\n⏳ Generating explanation first...")
        explanation = explanation_chain.invoke({"topic": topic})
        print("\n⏳ Generating revision notes...")
        result = notes_chain.invoke({"explanation": explanation})
        print(f"\n--- Revision Notes ---\n{result}\n")

    elif choice == "3":
        print("\n⏳ Running full pipeline...")
        explanation = explanation_chain.invoke({"topic": topic})
        notes = notes_chain.invoke({"explanation": explanation})
        result = quiz_chain.invoke({"notes": notes})
        print(f"\n--- Quiz (from pipeline) ---\n{result}\n")

def run():
    llm = create_llm()

    print("=== LangChain Chains - Academic Tutor ===")
    print("Modes:")
    print("  1) Explanation only")
    print("  2) Explanation → Revision Notes")
    print("  3) Explanation → Revision Notes → Quiz")
    print("  4) Full Pipeline (all steps + output shown at each stage)")
    print("  5) Quit\n")

    while True:
        choice = input("Select mode (1/2/3/4/5): ").strip()

        if choice == "5":
            break
        if choice not in ["1", "2", "3", "4"]:
            print("Invalid choice.\n")
            continue

        topic = input("Enter topic: ").strip()
        if not topic:
            continue

        if choice == "4":
            run_full_pipeline(llm, topic)
        else:
            run_single_chain(llm, choice, topic)

if __name__ == "__main__":
    run()