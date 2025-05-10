
from typing import Type
from pydantic import BaseModel, Field

'''
class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."
'''

# Import CrewAI tool base and standard tools
from crewai.tools import BaseTool, SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from typing import Type
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize Gemini LLM for tool use
gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Define input schema for timeline builder tool
class TimelineInput(BaseModel):
    events: list = Field(..., description="List of historical events")

# Tool: Build a markdown timeline from a list of events
class TimelineBuilderTool(BaseTool):
    name = "TimelineBuilderTool"
    description = "Creates markdown timelines from a list of historical events"
    args_schema: Type[BaseModel] = TimelineInput

    def _run(self, events: list) -> str:
        # Use Gemini to generate a markdown timeline
        response = gemini_llm.invoke(f"""
            Create a markdown timeline from these events: {events}
            Format: Use '###' for years, and '-' for events under each year.
            Include event dates, titles, and brief descriptions.
        """)
        return response.content

# Tool: Enhanced Serper search tool for structured web search results
class EnhancedSerperTool(SerperDevTool):
    def _run(self, query: str) -> list:
        # Call the parent SerperDevTool's run method
        results = super()._run(query)
        # Structure the results for easier downstream use
        return [{
            'title': r.get('title'),
            'link': r.get('link'),
            'snippet': r.get('snippet'),
            'date': r.get('date')
        } for r in results.get('organic', [])]

# Tool: Extract temporal context (dates, key events) from a query using Gemini
class ChronoAPITool(BaseTool):
    name = "ChronoAPITool"
    description = "Extracts temporal context (dates, key events) from a historical query"
    
    def _run(self, query: str) -> dict:
        # Use Gemini to extract start/end years and key events
        response = gemini_llm.invoke(f"""
            Extract a timeline JSON from: {query}
            Output format: {{"start_year":, "end_year":, "key_events": []}}
        """)
        return response.content

# Tool: Classify query intent (factual/hypothetical) using Gemini
class IntentClassifierTool(BaseTool):
    name = "IntentClassifierTool"
    description = "Classifies historical queries as factual or hypothetical"
    args_schema: Type[BaseModel] = None  # No schema needed for simple string input

    def _run(self, query: str) -> dict:
        # Use Gemini to classify the query
        response = gemini_llm.invoke(f"""
            Classify this historical query: {query}
            Output JSON format: {{"type": "factual/hypothetical", "intent": "academic/casual"}}
        """)
        return response.content
