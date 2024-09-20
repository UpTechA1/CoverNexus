from covernexus.state import State
from covernexus.utils import logger, GENERATOR, FINISH, EDITOR, EXECUTOR, CODEBASE
import re
import time
import os
import ast
import astor
import astunparse
import subprocess
import traceback

class Editing_Tool():
    """For editing and saving the generated test script to a file"""
    def __init__(self):
        pass
    
    def __call__(self, state):
        with open(state["generated_test_path"], "w") as f:
            state["raw_generated_test_script"] = self.postprocess(state["raw_generated_test_script"])
            f.write(state["raw_generated_test_script"])
            
        logger.info(f"Generated test script saved at: {state['generated_test_path']} for {state['raw_generated_test_script']}")
        
        return state
            
    def postprocess(self, output):
        # imp = "from codebase import *\n"
        pattern = r"```python(.*?)```"
        imp = "from " + CODEBASE.split('.')[0] + " import *\n"
        matches = re.findall(pattern, output, re.DOTALL)
        if len(matches) == 0:
            test_script = imp + output
        else:
            test_script = imp + matches[0]
        
        processed_test_script = test_script 
        
        return processed_test_script
    
    def split_tests(self, input_code):
        # Parse the original code into an AST
        tree = ast.parse(input_code)

        class TestSplitter(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # We want to split each function based on any assert statements
                new_functions = []
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        if isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr.startswith("assert"):
                            # Create a new function for each assert statement
                            if f"_{i + 1}" not in node.name[-3:]:
                                new_func_name = f"{node.name}_{i + 1}"
                            else:
                                new_func_name = node.name
                             
                            new_func = ast.FunctionDef(
                                name=new_func_name,
                                args=node.args,
                                body=[stmt],
                                decorator_list=node.decorator_list
                            )
                            new_functions.append(new_func)
                
                # Return the list of new functions to replace the original one
                return new_functions
        
        # Transform the AST to split test functions
        transformer = TestSplitter()
        new_tree = transformer.visit(tree)
        
        # Convert the transformed AST back to source code
        return astor.to_source(new_tree)

    def get_name(self):
        return EDITOR
    
    
    
import subprocess
class Executing_Tool():
    """For running coverage test script"""
    def __init__(self):
        self.edit_tool = Editing_Tool()
        pass
    def __call__(self, state):
        try:
            test_result = subprocess.run(["coverage", "run", state["generated_test_path"]], capture_output=True, text=True, check=False, timeout=5)
            if "FAIL" in test_result.stderr or "ERROR" in test_result.stderr:
                logger.error(f"Exist error while executing the test script")
                state["exist_error"] = True
            
            else:
                state["exist_error"] = False

            # Generate and capture the coverage report
            coverage_result = subprocess.run(["coverage", "report", "-m", state["codebase_path"]], capture_output=True, text=True, check=False)

            # Print the results of the coverage report
            state["executed_output"] = test_result.stderr + "\n"  + coverage_result.stdout if "FAIL" in test_result.stderr or "ERROR" in test_result.stderr else coverage_result.stdout
            
            logger.info(f"Executed test script successfully\n {state['executed_output']}")
            
        except subprocess.TimeoutExpired:
            state["executed_output"] = "The process took too long to complete and was terminated, please use small number for input or check the test script."
            logger.info("The process took too long to complete and was terminated.")
        
        
        new_score = self.extract_score(state["executed_output"])
        
        state["msg_trace"][state["current_iteration"]] = {
            "generated_test_script": state["raw_generated_test_script"],
            "executed_output": state["executed_output"],
            "coverage": new_score if new_score is not None else 0,
            "exist_error": state["exist_error"]
        }
        
        if new_score is not None:
            if state["current_iteration"] == 1:
                state["first_score"] = new_score
                state["first_generated_test_script"] = state["raw_generated_test_script"]

                state["best_score"] = state["first_score"]
                state["best_score_generated_test_script"] = state["raw_generated_test_script"]
                
                if not state["exist_error"]:
                    state["not_error_best_generated_test_script"] = state["raw_generated_test_script"]
                    state["not_error_best_score"] = new_score
                else:
                    state["not_error_best_generated_test_script"] = ""
                    state["not_error_best_score"] = 0
            else: 
                if new_score >= state["best_score"]: #just save script if coverage is higher, not care about error, because we will pass to filter node afterward
                    state["best_score"] = new_score
                    state["best_score_generated_test_script"] = state["raw_generated_test_script"]
                    
                if new_score >= state["not_error_best_score"] and not state["exist_error"]:
                    state["not_error_best_generated_test_script"] = state["raw_generated_test_script"]
                    state["not_error_best_score"] = new_score
                    
        if state["best_score"] is None:
            state["best_score"] = 0
        if state["not_error_best_score"] is None :
            state["not_error_best_score"] = 0
        if state["first_score"] is None:
            state["first_score"] = 0


        logger.info(f"Current coverage at iteration {state['current_iteration']}: {new_score}%")
        return state
    
    def extract_score(self, output):
        """Extract the coverage score from the coverage report"""
        match = re.search(f"{CODEBASE}.* (\d+%)", output)
        if match:
            coverage_percentage = match.group(1)
            coverage_percentage = float(coverage_percentage[:-1])
        else:
            coverage_percentage = None
            logger.info("Could not find coverage percentage.")
        return coverage_percentage
    
    def get_name(self):
        return EXECUTOR
