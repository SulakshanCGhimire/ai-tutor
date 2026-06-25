from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

ROLES = {
    "1": {
        "name": "Teacher",
        "system": "You are a patient and friendly teacher. Explain concepts in simple, clear language with relatable examples. Always encourage the student."
    },
    "2": {
        "name": "Examiner",
        "system": "You are a strict academic examiner. When given a topic, ask the student tough exam-style questions one at a time. Evaluate their answers and give a score out of 10 with feedback."
    },
    "3": {
        "name": "Study Coach",
        "system": "You are an energetic study coach. Give motivational advice, study strategies, time management tips, and help students build effective study plans."
    },
    "4": {
        "name": "Subject Expert",
        "system": "You are a subject matter expert with deep technical knowledge. Give precise, detailed, and advanced answers. Use technical terminology and provide in-depth explanations."
    }
}

def run():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )

    print("=== Role-Based Academic Tutor ===\n")
    print("Select a tutor role:")
    for key, role in ROLES.items():
        print(f"  {key}) {role['name']}")
    print("  5) Quit\n")

    while True:
        choice = input("Select role (1/2/3/4/5): ").strip()

        if choice == "5":
            break
        if choice not in ROLES:
            print("Invalid choice.\n")
            continue

        role = ROLES[choice]
        print(f"\n--- {role['name']} mode activated ---")
        print("Type 'switch' to change role, 'quit' to exit\n")

        history = [SystemMessage(content=role["system"])]

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "quit":
                return
            if user_input.lower() == "switch":
                print()
                break
            if not user_input:
                continue

            history.append(HumanMessage(content=user_input))
            response = llm.invoke(history)
            history.append(AIMessage(content=response.content))

            print(f"\n{role['name']}: {response.content}\n")

if __name__ == "__main__":
    run()