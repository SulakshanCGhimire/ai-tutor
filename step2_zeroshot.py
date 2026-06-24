from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def create_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )

def explain_topic(llm, topic):
    messages = [
        SystemMessage(content="You are an academic tutor. Explain concepts clearly for students."),
        HumanMessage(content=f"Explain the following topic in detail:\n\n{topic}")
    ]
    return llm.invoke(messages).content

def summarize_topic(llm, topic):
    messages = [
        SystemMessage(content="You are an academic tutor. Summarize study material concisely."),
        HumanMessage(content=f"Summarize the following topic in 5 bullet points:\n\n{topic}")
    ]
    return llm.invoke(messages).content

def simplify_topic(llm, topic):
    messages = [
        SystemMessage(content="You are an academic tutor. Simplify complex concepts for beginners."),
        HumanMessage(content=f"Explain the following topic in very simple language a 10-year-old can understand:\n\n{topic}")
    ]
    return llm.invoke(messages).content

def run():
    llm = create_llm()

    print("=== Zero-Shot Academic Tutor ===")
    print("Modes: 1) Explain  2) Summarize  3) Simplify  4) Quit\n")

    while True:
        choice = input("Select mode (1/2/3/4): ").strip()

        if choice == "4":
            break
        if choice not in ["1", "2", "3"]:
            print("Invalid choice.\n")
            continue

        topic = input("Enter topic: ").strip()
        if not topic:
            continue

        if choice == "1":
            print("\n--- Explanation ---")
            print(explain_topic(llm, topic))
        elif choice == "2":
            print("\n--- Summary ---")
            print(summarize_topic(llm, topic))
        elif choice == "3":
            print("\n--- Simplified ---")
            print(simplify_topic(llm, topic))

        print()

if __name__ == "__main__":
    run()