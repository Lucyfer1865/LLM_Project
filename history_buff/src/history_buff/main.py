#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from crew import HistoryBuff
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

load_dotenv()
'''
def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'History',
        'current_year': str(datetime.now().year)
    }
    
    try:
        HistoryBuff().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
'''

# Import necessary modules
import os
from dotenv import load_dotenv  # For loading environment variables
from crew import HistoryBuff    # Import the crew class from crew.py

# Load environment variables from .env file (for API keys, etc.)
load_dotenv()

def run():
    """
    Run the CrewAI HistoryBuff pipeline.
    """

    topic = input("Enter the topic for the history report: ")
    # Define input parameters for the crew
    inputs = {
        'topic': topic,  # Example topic
        'current_year': '2025'               # Example current year
    }
    try:
        # Create and run the crew, passing in the inputs
        result = HistoryBuff().crew().kickoff(inputs=inputs)
        # Print the final report to the console
        print("\n\nFinal Report:")
        print(result)
    except Exception as e:
        # Print any errors that occur
        print(f"Error: {str(e)}")

# Entry point for script execution
if __name__ == "__main__":
    run()
