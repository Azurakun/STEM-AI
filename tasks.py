from crewai import Task
from agents import Agents
from tools import search_tool

class Tasks:
    def __init__(self, event):
        self.stem_info = event

    def research_task(self):
        agent = Agents().stem_agent()
        search_task = Task (
            description = """Research and provide detailed information about {}, or STEM.
                            use search_tool to search for the answers.
                            Use reliable sources to ensure accuracy and include relevant STEM context, causes, and consequences. 
                            If needed, connect the event or figure to broader STEM trends or themes.""".format(self.stem_info), 
            expected_output="""A comprehensive and well-researched report that includes:
                            - An overview of the topics related to Science Technology Enggineering & Mathematics.
                            - Key dates, locations, and individuals involved in the topics. 
                            - The causes and consequences of the event or significance of the figure. 
                            - Other relevant context and connections to broader themes or trends in the same topic. 
                            - Citations from reliable sources to support the information provided.
                            make the information provided in a third person perspective.""", 
            agent=agent, 
            tools=[search_tool]
        )
        return search_task
