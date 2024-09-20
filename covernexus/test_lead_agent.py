from covernexus.state import State
from covernexus.utils import logger, GENERATOR, FINISH, PLANNER, CODEBASE
import time

class Test_Lead_Agent():
    """This agent is responsible for determining the next steps based on the coverage score of a specific codebase file. 
    If the coverage score meets the required threshold specified in the configuration, the next agent will be the finishing agent. 
    Otherwise, the next agent will be assigned based on the lower coverage score or errors encountered."""
    
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: State):
        if state["use_planner"]:
            if state["current_iteration"] >= state["config"]["max_iteration"]:
                state["instruction"] = FINISH
                return state
            elif state["config"].get("current_testbase_coverage", 0) >=  state["config"]["desired_coverage"]:
                state["instruction"] = FINISH
                state["best_score"] = state["config"].get("current_testbase_coverage", 0)
                state["first_score"] = state["config"].get("current_testbase_coverage", 0)
                state["not_error_best_score"] = state["config"].get("current_testbase_coverage", 0)
                return state
            
            if state["codebase_script"]:
                codebase_script = "\n".join([f"Line {i+1} "+ code for i, code in enumerate(state["codebase_script"].split("\n"))])
            else:
                codebase_script = state["codebase_script"]   
                 
            config = f"Desired coverage: {state['config']['desired_coverage']}%"
            
            if state["current_iteration"] != 0:
                self.system_prompt = """You are responsible for determining next steps for it based on the coverage score of a specific codebase file named """ + CODEBASE + """. 
    If the coverage score meets the required threshold specified in """ + config + """ and does not exist any errors or failure, the next agent will be """ + FINISH + """. Otherwise, assign the next agent as """ + GENERATOR + """ due to the lower coverage score and errors/failure encountered.
    You should guide next agent to generate a test script with high coverage score step by step if there is any error then modify error testcases; require next agent to provide use subtest

    Use the following information to provide the instruction:
    - Codebase script: """ + codebase_script + """
    - Generated Test Script: """ + state["raw_generated_test_script"] + """
    - Coverage Report Execution Output: """ + state["executed_output"] + """
    Here are name of next agent and instruction for it to generate test script with high coverage score:
    """
            else:
                self.system_prompt = """You are responsible for determining next steps for it based on the coverage score of a specific codebase file named """ + CODEBASE + """. 
    If the coverage score meets the required threshold specified in """ + config + """ and does not exist any errors or failure, the next agent will be """ + FINISH + """. Otherwise, assign the next agent as """ + GENERATOR + """ due to the lower coverage score and errors/failure encountered.
    You should guide next agent to generate a test script with high coverage score step by step if there is any error then modify error testcases; require next agent to provide use subtest in format of given testbase

    Use the following information to provide the instruction:
    - Codebase script: """ + codebase_script + """
    - Testbase script: """ + state["testbase_script"] + """
    - Coverage Report Execution Output: """ + state["executed_output"] + """
    Here are name of next agent and instruction for it to generate test script with high coverage score:
    """
            if state["stream"]: # can only use for close-source model, it mean stream = true -> use close-source model
                self.runnable = self.llm 
                output = ""
                for chunk in self.runnable.stream(self.system_prompt):
                    if hasattr(chunk, "content"):
                        print(chunk.content, end="", flush=True)
                        output+=chunk.content
                    else:
                        print(chunk, end="", flush=True)
                        output+=chunk
                
            else: # can only use for open-source model, it mean stream = false -> use open-source model
                
                output = self.llm.invoke(self.system_prompt)
                if hasattr(output, "content"):
                    output = output.content
                print(output)
            
            state["instruction"] = str(output.strip())
            
        else:
            if state["current_iteration"] == 0:
                state["instruction"] = GENERATOR
            else:
                state["instruction"] = FINISH
        
        return state
    
    def get_name(self):
        return PLANNER 