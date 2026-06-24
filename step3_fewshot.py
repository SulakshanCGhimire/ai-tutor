from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def create_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )

def generate_quiz(llm, topic):
    messages = [
        SystemMessage(content="You are an academic tutor that generates quizzes in a strict format."),
        HumanMessage(content=f"""Generate a 3-question multiple choice quiz on the topic below.
Follow this exact format:

Q1: What is the powerhouse of the cell?
A) Nucleus
B) Mitochondria
C) Ribosome
D) Golgi body
Answer: B

Q2: What process do plants use to make food?
A) Respiration
B) Digestion
C) Photosynthesis
D) Fermentation
Answer: C

Now generate a quiz on: {topic}""")
    ]
    return llm.invoke(messages).content

def categorized_answer(llm, question):
    messages = [
        SystemMessage(content="You are an academic tutor that answers questions in a structured format."),
        HumanMessage(content=f"""Answer the question below using this exact format:

Question: What is Newton's First Law?
Definition: An object at rest stays at rest, and an object in motion stays in motion unless acted upon by an external force.
Example: A ball rolling on the floor slows down due to friction.
Key Term: Inertia

Question: What causes seasons on Earth?
Definition: Seasons are caused by the tilt of Earth's axis as it orbits the Sun.
Example: When the Northern Hemisphere tilts toward the Sun, it experiences summer.
Key Term: Axial tilt

Now answer this question in the same format:
Question: {question}""")
    ]
    return llm.invoke(messages).content

def run():
    llm = create_llm()

    print("=== Few-Shot Academic Tutor ===")
    print("Modes: 1) Generate Quiz  2) Structured Answer  3) Quit\n")

    while True:
        choice = input("Select mode (1/2/3): ").strip()

        if choice == "3":
            break
        if choice not in ["1", "2"]:
            print("Invalid choice.\n")
            continue

        if choice == "1":
            topic = input("Enter quiz topic: ").strip()
            if not topic:
                continue
            print("\n--- Quiz ---")
            print(generate_quiz(llm, topic))

        elif choice == "2":
            question = input("Enter your question: ").strip()
            if not question:
                continue
            print("\n--- Structured Answer ---")
            print(categorized_answer(llm, question))

        print()

if __name__ == "__main__":
    run()