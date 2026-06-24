from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def create_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )

def solve_math(llm, problem):
    messages = [
        SystemMessage(content="You are an academic tutor. Always solve problems step-by-step before giving a final answer."),
        HumanMessage(content=f"""Solve the following math problem step-by-step.

Example:
Problem: A train travels 60 km/h for 2.5 hours. How far does it travel?
Step 1: Identify the formula — Distance = Speed × Time
Step 2: Substitute values — Distance = 60 × 2.5
Step 3: Calculate — Distance = 150 km
Final Answer: The train travels 150 km.

Now solve this problem step-by-step:
Problem: {problem}""")
    ]
    return llm.invoke(messages).content

def solve_reasoning(llm, problem):
    messages = [
        SystemMessage(content="You are an academic tutor. Break down reasoning problems into clear logical steps."),
        HumanMessage(content=f"""Solve the following reasoning problem step-by-step.

Example:
Problem: All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?
Step 1: Identify the premise — All mammals are warm-blooded.
Step 2: Identify the fact — Whales are mammals.
Step 3: Apply logic — Since whales are mammals, and all mammals are warm-blooded...
Final Answer: Yes, whales are warm-blooded.

Now solve this step-by-step:
Problem: {problem}""")
    ]
    return llm.invoke(messages).content

def solve_science(llm, problem):
    messages = [
        SystemMessage(content="You are an academic tutor. Explain science problems step-by-step with clear reasoning."),
        HumanMessage(content=f"""Solve the following science problem step-by-step.

Example:
Problem: Why does ice float on water?
Step 1: Identify the concept — Density determines if an object floats or sinks.
Step 2: Compare densities — Ice has a density of 0.917 g/cm³, water has 1.0 g/cm³.
Step 3: Apply the rule — Since ice is less dense than water, it floats.
Final Answer: Ice floats on water because it is less dense than liquid water.

Now solve this step-by-step:
Problem: {problem}""")
    ]
    return llm.invoke(messages).content

def run():
    llm = create_llm()

    print("=== Chain-of-Thought Academic Tutor ===")
    print("Modes: 1) Math Problem  2) Reasoning Problem  3) Science Problem  4) Quit\n")

    while True:
        choice = input("Select mode (1/2/3/4): ").strip()

        if choice == "4":
            break
        if choice not in ["1", "2", "3"]:
            print("Invalid choice.\n")
            continue

        problem = input("Enter your problem: ").strip()
        if not problem:
            continue

        if choice == "1":
            print("\n--- Math Solution ---")
            print(solve_math(llm, problem))
        elif choice == "2":
            print("\n--- Reasoning Solution ---")
            print(solve_reasoning(llm, problem))
        elif choice == "3":
            print("\n--- Science Solution ---")
            print(solve_science(llm, problem))

        print()

if __name__ == "__main__":
    run()