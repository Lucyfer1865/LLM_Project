from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

'''
@CrewBase
class HistoryBuff():
    """HistoryBuff crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the HistoryBuff crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
'''


# Import Gemini LLM integration
from langchain_google_genai import ChatGoogleGenerativeAI

# Define the main CrewAI class using the CrewBase decorator
@CrewBase
class HistoryBuff:
    """
    HistoryBuff crew for historical Q&A with timeline generation.
    """

    # YAML config file paths for agents and tasks
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    def __init__(self):
        # Initialize Gemini LLM (Google Generative AI)
        self.gemini = ChatGoogleGenerativeAI(model="gemini-pro")

    # Define each agent using the @agent decorator

    @agent
    def query_decipherer(self) -> Agent:
        # Agent for analyzing and classifying queries
        return Agent(config=self.agents_config['query_decipherer'], llm=self.gemini)

    @agent 
    def temporal_specialist(self) -> Agent:
        # Agent for extracting temporal/geographical context
        return Agent(config=self.agents_config['temporal_specialist'], llm=self.gemini)

    @agent
    def researcher(self) -> Agent:
        # Agent for deep research and data gathering
        return Agent(config=self.agents_config['researcher'], llm=self.gemini)

    @agent
    def timeline_agent(self) -> Agent:
        # Agent for creating timelines from events
        return Agent(config=self.agents_config['timeline_agent'], llm=self.gemini)

    @agent
    def reporting_analyst(self) -> Agent:
        # Agent for synthesizing the final report
        return Agent(config=self.agents_config['reporting_analyst'], llm=self.gemini)

    # Define each task using the @task decorator

    @task
    def query_analysis(self) -> Task:
        # Task: Analyze the query
        return Task(config=self.tasks_config['query_analysis'])

    @task
    def temporal_context(self) -> Task:
        # Task: Extract temporal context, depends on query analysis
        return Task(config=self.tasks_config['temporal_context'], context=[self.query_analysis])

    @task
    def research(self) -> Task:
        # Task: Conduct research, depends on temporal context
        return Task(config=self.tasks_config['research'], context=[self.temporal_context])

    @task
    def timeline_creation(self) -> Task:
        # Task: Create timeline, depends on research, outputs to file
        return Task(
            config=self.tasks_config['timeline_creation'],
            context=[self.research],
            output_file='timeline.md'
        )

    @task
    def reporting(self) -> Task:
        # Task: Create final report, depends on research and timeline, outputs to file
        return Task(
            config=self.tasks_config['reporting'],
            context=[self.research, self.timeline_creation],
            output_file='full_report.md'
        )

    # Define the crew using the @crew decorator
    @crew
    def crew(self) -> Crew:
        # Assemble the crew with all agents and tasks
        return Crew(
            agents=self.agents,      # All agents defined above
            tasks=self.tasks,        # All tasks defined above
            process=Process.hierarchical,  # Use hierarchical process for dependencies
            verbose=2,               # Verbose logging for debugging
            manager_llm=self.gemini  # Use Gemini as the manager LLM
        )
