from dotenv import load_dotenv
import os
import google.generativeai as genai
from crewai import Agent, Task, Crew 

# Load .env variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Wrapper class for CrewAI
class GeminiLLM:
    def __init__(self, model_name="gemini-pro"):
        self.model = genai.GenerativeModel(model_name)

    def complete(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text

# Create your LLM instance
gemini_llm = GeminiLLM()

# Create a CrewAI agent using Gemini
agent = Agent(
    role="Assistant",
    goal="Help the user with whatever they ask",
    backstory="An advanced AI model trained by Google.",
    llm=gemini_llm
)