from typing import Type
from pydantic import BaseModel, Field
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Import CrewAI tool base and standard tools
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# Load env variables and configure Gemini
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables!")

# Create a direct Gemini model instance
gemini_model = genai.GenerativeModel("gemini-pro")

# Define input schema for timeline builder tool
class TimelineInput(BaseModel):
    events: list = Field(..., description="List of historical events")

# Tool: Build a markdown timeline from a list of events
class TimelineBuilderTool(BaseTool):
    """Tool for building a historical timeline."""
    
    name: str = "timeline_builder_tool"
    description: str = "Tool for building a historical timeline by analyzing events and their connections."
    args_schema: Type[BaseModel] = TimelineInput
    
    def __init__(self):
        super().__init__()
    
    def _run(self, events: list) -> str:
        # Use Gemini to generate a markdown timeline
        prompt = f"""
            Create a markdown timeline from these events: {events}
            Format: Use '###' for years, and '-' for events under each year.
            Include event dates, titles, and brief descriptions.
        """
        
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error in TimelineBuilderTool: {str(e)}")
            return "Error: Failed to generate timeline."


# Tool: Enhanced Serper search tool for structured web search results
class EnhancedSerperTool(SerperDevTool):
    def __init__(self):
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError("SERPER_API_KEY environment variable is required")
        super().__init__(api_key=api_key)
    
    def _run(self, query: str) -> list:
        try:
            # Call the parent SerperTool's run method
            results = super()._run(query)
            # Structure the results for easier downstream use
            return [{
                'title': r.get('title'),
                'link': r.get('link'),
                'snippet': r.get('snippet'),
                'date': r.get('date')
            } for r in results.get('organic', [])]
        except Exception as e:
            print(f"Error in SerperDev search: {str(e)}")
            return [{"error": str(e)}]

# Tool: Extract temporal context (dates, key events) from a query using Gemini
class ChronoAPITool(BaseTool):
    name: str = "ChronoAPITool"
    description: str = "Extracts temporal context (dates, key events) from a historical query"

    def __init__(self):
        super().__init__()
    
    def _run(self, query: str) -> dict:
        # Use Gemini to extract start/end years and key events
        prompt = f"""
            Extract a timeline JSON from: {query}
            Output format: {{"start_year":, "end_year":, "key_events": []}}
        """
        
        try:
            response = gemini_model.generate_content(prompt)
            result = response.text
            return json.loads(result) if result else {"error": "Failed to extract timeline."}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON from Gemini response."}
        except Exception as e:
            print(f"Gemini API error in ChronoAPITool: {str(e)}")
            return {"error": f"Error processing with Gemini: {str(e)}"}

# Tool: Classify query intent (factual/hypothetical) using Gemini
class IntentClassifierTool(BaseTool):
    name: str = "IntentClassifierTool"
    description: str = "Classifies historical queries as factual or hypothetical"
    
    def __init__(self):
        super().__init__()

    def _run(self, query: str) -> dict:
        # Use Gemini to classify the query
        prompt = f"""
            Classify this historical query: {query}
            Output JSON format: {{"type": "factual/hypothetical", "intent": "academic/casual"}}
        """
        
        try:
            response = gemini_model.generate_content(prompt)
            result = response.text
            return json.loads(result) if result else {"error": "Failed to classify query."}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON from Gemini response."}
        except Exception as e:
            print(f"Gemini API error in IntentClassifierTool: {str(e)}")
            return {"error": f"Error classifying with Gemini: {str(e)}"}

# Tool: Format markdown for reports
class MarkdownFormatterTool(BaseTool):
    name: str = "MarkdownFormatterTool"
    description: str = "Formats historical data into well-structured markdown reports"
    
    def __init__(self):
        super().__init__()

    def _run(self, content: str, format_type: str = "report") -> str:
        # Use Gemini to format the content according to the specified format type
        prompt = f"""
            Format this historical content into a well-structured {format_type} using markdown:
            {content}
            
            Include:
            - Clear headings and subheadings
            - Bullet points for key facts
            - Proper citation formatting
            - Table of contents (if appropriate)
        """
        
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error in MarkdownFormatterTool: {str(e)}")
            return f"Error: Failed to format {format_type}."