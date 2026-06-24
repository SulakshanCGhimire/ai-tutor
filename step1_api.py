from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

def create_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )

def chat():
    llm = create_llm()
    history = [
        SystemMessage(content="You are a helpful academic tutor. Help students learn and prepare for exams.")
    ]

    print("=== AI Academic Tutor ===")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))
        response = llm.invoke(history)
        history.append(AIMessage(content=response.content))

        print(f"\nTutor: {response.content}\n")

if __name__ == "__main__":
    chat()