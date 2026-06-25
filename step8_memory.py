from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

def create_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )

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
            f"The student's name is {self.user_name}. "
            f"They are studying {self.subject}. "
            f"Topics they have already covered: {topics}. "
            f"Always address the student by name. "
            f"Build on previous topics when relevant. "
            f"Keep responses encouraging and personalized."
        )

    def get_full_history(self) -> list:
        return [SystemMessage(content=self.get_system_prompt())] + self.chat_history

    def show_summary(self):
        print(f"\n--- Session Summary for {self.user_name} ---")
        print(f"Subject: {self.subject}")
        if self.topics_studied:
            print(f"Topics Studied:")
            for i, t in enumerate(self.topics_studied, 1):
                print(f"  {i}. {t}")
        else:
            print("Topics Studied: none")
        print(f"Total Exchanges: {len(self.chat_history) // 2}")
        print("---\n")

def extract_topic(user_input: str) -> str | None:
    keywords = ["explain", "what is", "tell me about", "teach me", "study", "learn about"]
    lower = user_input.lower()
    # Check keyword triggers
    for kw in keywords:
        if kw in lower:
            topic = lower.replace(kw, "").strip(" ?.")
            return topic.title() if topic else None
    # Fallback: treat short inputs as topics directly
    if len(user_input.split()) <= 5:
        return user_input.title()
    return None

def run():
    llm = create_llm()

    print("=== Memory-Based Personalized Tutor ===\n")
    user_name = input("What is your name? ").strip()
    subject = input("What subject are you studying? ").strip()

    memory = TutorMemory(user_name=user_name, subject=subject)

    print(f"\nHello {user_name}! I'm your personal tutor for {subject}.")
    print("Commands: 'summary' to see session summary, 'quit' to exit\n")

    while True:
        user_input = input(f"{user_name}: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            memory.show_summary()
            break

        if user_input.lower() == "summary":
            memory.show_summary()
            continue

        # Auto-detect topic from input
        topic = extract_topic(user_input)
        if topic:
            memory.add_topic(topic)

        messages = memory.get_full_history()
        messages.append(HumanMessage(content=user_input))

        response = llm.invoke(messages)
        ai_reply = response.content

        memory.add_interaction(user_input, ai_reply)

        print(f"\nTutor: {ai_reply}\n")

if __name__ == "__main__":
    run()