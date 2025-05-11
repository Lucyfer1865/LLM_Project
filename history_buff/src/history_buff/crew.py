import os
import yaml
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv

# Import custom tools
from src.history_buff.tools.custom_tool import (
    TimelineBuilderTool, 
    EnhancedSerperTool, 
    ChronoAPITool, 
    IntentClassifierTool, 
    MarkdownFormatterTool
)
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# Load environment variables
load_dotenv()

# Disable telemetry and tracing to minimize dependencies
os.environ["CREWAI_TELEMETRY"] = "False"
os.environ["LANGCHAIN_TRACING"] = "false"

class HistoryBuff:
    """
    HistoryBuff crew for historical Q&A with timeline generation.
    Uses OpenAI for CrewAI and Gemini for custom tools.
    """
    
    def __init__(self):
        # Fix paths using proper absolute path resolution
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load configuration files
        agents_config_path = os.path.join(base_dir, 'config', 'agents.yaml')
        tasks_config_path = os.path.join(base_dir, 'config', 'tasks.yaml')
        
        print(f"Loading agents config from: {agents_config_path}")
        print(f"Loading tasks config from: {tasks_config_path}")
        
        with open(agents_config_path, 'r') as f:
            self.agents_config = yaml.safe_load(f)
        with open(tasks_config_path, 'r') as f:
            self.tasks_config = yaml.safe_load(f)
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Create agents
        self.agents = self._create_agents()
        
        # Tasks will be created later
        self.tasks = {}
    
    def _initialize_tools(self):
        """Initialize all tools with proper error handling."""
        tools = {}
        
        # Initialize standard tools
        try:
            tools["SerperDevTool"] = SerperDevTool()
            print("Successfully initialized SerperDevTool")
        except Exception as e:
            print(f"Warning: Failed to initialize SerperDevTool: {str(e)}")
            
        try:
            tools["ScrapeWebsiteTool"] = ScrapeWebsiteTool()
            print("Successfully initialized ScrapeWebsiteTool")
        except Exception as e:
            print(f"Warning: Failed to initialize ScrapeWebsiteTool: {str(e)}")
            
        # Initialize custom tools
        try:
            tools["TimelineBuilderTool"] = TimelineBuilderTool()
            tools["EnhancedSerperTool"] = EnhancedSerperTool()
            tools["ChronoAPITool"] = ChronoAPITool()
            tools["IntentClassifierTool"] = IntentClassifierTool()
            tools["MarkdownFormatterTool"] = MarkdownFormatterTool()
            print("Successfully initialized all custom tools")
        except Exception as e:
            print(f"Error initializing custom tools: {str(e)}")
            raise
        
        return tools
    
    def _create_agents(self):
        """Create all the agents for the crew."""
        agents = {}
        
        # Create each agent
        for agent_name in ['query_decipherer', 'temporal_specialist', 'researcher', 
                           'timeline_agent', 'reporting_analyst']:
            agents[agent_name] = self._create_agent(agent_name)
            
        return agents
    
    def _create_agent(self, agent_name):
        """Create a single agent with its tools."""
        config = self.agents_config[agent_name]
        
        # Get tools for this agent
        agent_tools = []
        for tool_name in config.get('tools', []):
            if tool_name in self.tools:
                agent_tools.append(self.tools[tool_name])
                print(f"Added tool {tool_name} to agent {agent_name}")
            else:
                print(f"Warning: Tool {tool_name} not found for agent {agent_name}")
        
        # Create the agent using OpenAI model
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            tools=agent_tools,
            llm="gpt-3.5-turbo",  # Using OpenAI's GPT-3.5 Turbo
            verbose=True
        )
    
    def _create_tasks(self, inputs=None):
        """Create all the tasks for the crew with proper format string substitution."""
        tasks = {}
        
        # Default inputs if none provided
        if inputs is None:
            inputs = {'topic': 'the historical topic', 'current_year': '2025'}
            
        try:
            # Create query analysis task
            tasks['query_analysis'] = Task(
                description=self.tasks_config['query_analysis']['description'].format(**inputs),
                expected_output=self.tasks_config['query_analysis']['expected_output'],
                agent=self.agents['query_decipherer']
            )
            
            # Create temporal context task
            tasks['temporal_context'] = Task(
                description=self.tasks_config['temporal_context']['description'].format(**inputs),
                expected_output=self.tasks_config['temporal_context']['expected_output'],
                agent=self.agents['temporal_specialist'],
                context=[tasks['query_analysis']]
            )
            
            # Create research task
            tasks['research'] = Task(
                description=self.tasks_config['research']['description'].format(**inputs),
                expected_output=self.tasks_config['research']['expected_output'],
                agent=self.agents['researcher'],
                context=[tasks['temporal_context']]
            )
            
            # Create timeline creation task
            tasks['timeline_creation'] = Task(
                description=self.tasks_config['timeline_creation']['description'].format(**inputs),
                expected_output=self.tasks_config['timeline_creation']['expected_output'],
                agent=self.agents['timeline_agent'],
                context=[tasks['research']],
                output_file='timeline.md'
            )
            
            # Create reporting task
            tasks['reporting'] = Task(
                description=self.tasks_config['reporting']['description'].format(**inputs),
                expected_output=self.tasks_config['reporting']['expected_output'],
                agent=self.agents['reporting_analyst'],
                context=[tasks['research'], tasks['timeline_creation']],
                output_file='full_report.md'
            )
        except KeyError as e:
            print(f"Error formatting task description: {e}")
            print("Using default task descriptions instead")
            
            # Create simplified tasks with fixed descriptions if there's an error
            tasks['query_analysis'] = Task(
                description=f"Analyze the historical query: '{inputs.get('topic', 'historical topic')}'",
                expected_output="JSON with query type and key entities",
                agent=self.agents['query_decipherer']
            )
            
            tasks['temporal_context'] = Task(
                description=f"Establish historical timeframe for: '{inputs.get('topic', 'historical topic')}'",
                expected_output="JSON timeline with dates",
                agent=self.agents['temporal_specialist'],
                context=[tasks['query_analysis']]
            )
            
            tasks['research'] = Task(
                description=f"Research: '{inputs.get('topic', 'historical topic')}'",
                expected_output="Historical data with sources",
                agent=self.agents['researcher'],
                context=[tasks['temporal_context']]
            )
            
            tasks['timeline_creation'] = Task(
                description=f"Create timeline for: '{inputs.get('topic', 'historical topic')}'",
                expected_output="Markdown timeline",
                agent=self.agents['timeline_agent'],
                context=[tasks['research']],
                output_file='timeline.md'
            )
            
            tasks['reporting'] = Task(
                description=f"Create report about: '{inputs.get('topic', 'historical topic')}'",
                expected_output="Markdown report",
                agent=self.agents['reporting_analyst'],
                context=[tasks['research'], tasks['timeline_creation']],
                output_file='full_report.md'
            )
            
        # Store tasks for use in crew
        self.tasks = tasks
        
        return tasks
    
    def crew(self):
        """Create and return the crew instance."""
        # Ensure we have tasks created
        if not self.tasks:
            self._create_tasks()
            
        try:
            # Create the crew with OpenAI
            return Crew(
                agents=list(self.agents.values()),
                tasks=list(self.tasks.values()),
                process=Process.hierarchical,
                verbose=True,
                manager_llm="gpt-3.5-turbo"  # Using OpenAI's GPT-3.5 Turbo
            )
        except Exception as e:
            print(f"Error creating crew: {str(e)}")
            # Try again with a simpler setup (no manager_llm, no process)
            print("Attempting to create crew with simplified configuration...")
            return Crew(
                agents=list(self.agents.values()),
                tasks=list(self.tasks.values()),
                verbose=True
            )