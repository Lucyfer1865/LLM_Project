from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini for custom tools
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables!")

# Ensure OpenAI API key is set in environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    print("Successfully configured OpenAI API key")
else:
    print("Warning: OPENAI_API_KEY not found in environment variables!")

# Disable telemetry
os.environ["CREWAI_TELEMETRY"] = "False"

# Wrapper for custom tools using Gemini
class GeminiLLM:
    """
    A wrapper around Gemini API for custom tools.
    Note: CrewAI will use OpenAI directly, while custom tools use this class.
    """
    
    def __init__(self, model_name="gemini-pro", temperature=0.7):
        """Initialize with model name and temperature."""
        self.model_name = model_name
        self.temperature = temperature
        self._model = genai.GenerativeModel(model_name=model_name)
        print(f"Initialized GeminiLLM wrapper for custom tools with model: {model_name}")
    
    def complete(self, prompt: str) -> str:
        """Simple completion method for custom tools."""
        try:
            response = self._model.generate_content(
                contents=prompt,
                generation_config={"temperature": self.temperature}
            )
            return response.text
        except Exception as e:
            print(f"Error in GeminiLLM complete: {str(e)}")
            return f"Error: {str(e)}"