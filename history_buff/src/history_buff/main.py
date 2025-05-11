#!/usr/bin/env python
import warnings
import os
import sys
from dotenv import load_dotenv

# Import the crew class from crew.py
from src.history_buff.crew import HistoryBuff

# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables from .env file (for API keys, etc.)
load_dotenv()

# Check for required API keys
def check_api_keys():
    """Check that all required API keys are present in the environment."""
    missing_keys = []
    
    # Check for OpenAI API key (required for CrewAI)
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    
    # Check for Gemini API key (used for custom tools)
    if not os.getenv("GEMINI_API_KEY"):
        missing_keys.append("GEMINI_API_KEY")
    
    # Check for SerperDev API key (if needed)
    if not os.getenv("SERPER_API_KEY"):
        missing_keys.append("SERPER_API_KEY")
    
    if missing_keys:
        print("❌ Error: The following required API keys are missing from your .env file:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nPlease add these keys to your .env file and try again.")
        return False
    
    return True

def run():
    """
    Run the CrewAI HistoryBuff pipeline.
    """
    # Configure environment for CrewAI
    os.environ["CREWAI_TELEMETRY"] = "False"
    os.environ["LANGCHAIN_TRACING"] = "false"
    
    # First check for required API keys
    if not check_api_keys():
        return
    
    # Get the topic from user input
    topic = input("Enter the topic for the history report: ")
    
    # Define input parameters for the crew
    inputs = {
        'topic': topic,
        'current_year': '2025'
    }
    
    try:
        # Create the history buff instance 
        print("Initializing HistoryBuff...")
        history_buff = HistoryBuff()
        
        # Set up tasks with inputs
        print(f"Creating tasks for topic: {topic}")
        history_buff._create_tasks(inputs)
        
        # Get the crew and kick it off
        print("Creating crew and starting the process...")
        crew = history_buff.crew()
        
        # Execute the crew's tasks
        try:
            print(f"Executing tasks for topic: {topic}...")
            result = crew.kickoff(inputs=inputs)
            
            # Print the final report to the console
            print("\n\nFinal Report:")
            print(result)
            
            # Inform the user where the full report is saved
            print("\n\nThe full report has been saved to 'full_report.md' and the timeline to 'timeline.md'")
            
        except Exception as e:
            print(f"\n\nError during execution: {str(e)}")
            print("\nThis could be due to API limits, connection issues, or other runtime errors.")
            print("\nCheck your API keys and try again, or try with a simpler query.")
            
    except Exception as e:
        # Print any unexpected errors that occur
        print(f"Unexpected error: {str(e)}")
        # Print stack trace for debugging
        import traceback
        traceback.print_exc()

# Entry point for script execution
if __name__ == "__main__":
    run()