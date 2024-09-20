from langgraph.graph import END, StateGraph
from covernexus.state import State
from covernexus.test_lead_agent import Test_Lead_Agent
from covernexus.test_generator_agent import Test_Generator_Agent
from covernexus.tools import Editing_Tool, Executing_Tool
from covernexus.utils import logger, FINISH, GENERATOR

def decision_maker(state: State):
    """This function is responsible for extracting the next agent based on the current state of the system."""
    next_agent = GENERATOR if GENERATOR in state["instruction"] or state["current_iteration"] == 0 and state["config"].get("current_testbase_coverage", 0) < state["config"]["desired_coverage"] else END # or state["current_iteration"] == 0
    logger.info(f"Next agent: {next_agent}")
    # If the current iteration exceeds the maximum iteration, the next agent will be the finishing agent.
    if state["current_iteration"] >= state["config"]["max_iteration"]:
        return END
    return next_agent

class CoverNexus:
    """This class is responsible for managing the flow of multiple agents and tools to achieve the desired goal."""
    def __init__(self, llm1, llm2):
        self.llm1 = llm1
        self.llm2 = llm2
        
        # Create a state graph to manage the flow of agents
        graph = StateGraph(State)
        
        # Initialize the agents and tools
        test_lead_agent = Test_Lead_Agent(self.llm1)
        test_generator_agent = Test_Generator_Agent(self.llm2)
        editing_tool = Editing_Tool()
        executing_tool = Executing_Tool()
        
        # Add the agents and tools to the state graph
        graph.add_node(test_lead_agent.get_name(), test_lead_agent)
        graph.add_node(test_generator_agent.get_name(), test_generator_agent)
        graph.add_node(editing_tool.get_name(), editing_tool)
        graph.add_node(executing_tool.get_name(), executing_tool)
        
        # Define the flow of the agents and tools
        graph.add_edge(test_generator_agent.get_name(), editing_tool.get_name())
        graph.add_edge(editing_tool.get_name(), executing_tool.get_name())
        graph.add_edge(executing_tool.get_name(), test_lead_agent.get_name())
        graph.add_conditional_edges(test_lead_agent.get_name(), decision_maker)
        

        # Set the entry point of the state graph
        graph.set_entry_point(test_lead_agent.get_name())
        self.graph = graph.compile() 

    def __call__(self, state: State):
        return self.graph.invoke(state)