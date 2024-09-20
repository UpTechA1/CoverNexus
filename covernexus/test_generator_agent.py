from covernexus.state import State
import time
from covernexus.utils import logger, GENERATOR, CODEBASE

class Test_Generator_Agent():
    def __init__(self, llm):
        self.llm = llm
        
    def __call__(self, state: State):
        # time.sleep(5)
        # Increment the current iteration
        state["current_iteration"] += 1
        logger.info(f"Current iteration: {state['current_iteration']}")

        # Load the codebase script
        if state["current_iteration"] == 1 or state["codebase_script"] == "": 
            with open(state["codebase_path"]) as f:
                self.code_base = f.read()
                state["codebase_script"] = self.code_base
                logger.info(f"Reading codebase script successfully")
                
        # Load the testbase script
        if state['testbase_path'] != '' and state['current_iteration'] == 1:
            with open(state["testbase_path"]) as f:
                self.test_base = f.read()
                state["testbase_script"] = self.test_base
                logger.info(f"Reading testbase script successfully")
        
        des_cov = str(state["config"]["desired_coverage"])
        self.system_prompt = """You are tasked with generating a high-coverage test script excluding codebase script, targeting a coverage score of """+ des_cov +"""%. The script should comprehensively cover all logic, branches, and statements in the codebase.
You should provide only the test script without explanations, inclusion of the codebase script inside, and duplicative or verbose or too many subtests, follow the instructions below
Codebase scripts """ + CODEBASE + """: """ + state["codebase_script"] + """
"""
        
        if state['testbase_script'] != '' and state['current_iteration']==1:
            self.system_prompt += "Testbase script: " + state['testbase_script'] + "\n"
        else:    
            self.system_prompt += "Generated testbase script: " + state['raw_generated_test_script'] + "\n"
            
        self.system_prompt += "Executed output of test script: "+ state['executed_output'] + "\n"
        self.system_prompt += "Instruction based on executed output: " + str(state["instruction"]) + "\n"
        self.system_prompt += "Coverage test script:\n"
        
        if state["stream"]:
            self.runnable = self.llm  
            output = ""
            for chunk in self.runnable.stream(self.system_prompt):
                if hasattr(chunk, "content"):
                    print(chunk.content, end="", flush=True)
                    output+=chunk.content
                else:
                    print(chunk, end="", flush=True)
                    output+=chunk
        else:
            
            output = self.llm.invoke(self.system_prompt)
            if hasattr(output, "content"):
                output = output.content
            print(output)
            
        state["raw_generated_test_script"] = output
        
        logger.info(f"Generated test script based on codebase script successfully\n {state['raw_generated_test_script']}")
        return state
    
    def get_name(self):
        return GENERATOR