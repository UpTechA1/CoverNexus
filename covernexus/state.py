from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import AnyMessage, add_messages
from typing import Any
import time

class State(TypedDict):
    thoughts: Annotated[list[AnyMessage], add_messages]
    config: dict[str, Any] = {}
    instruction :str = ""
    current_iteration: int = 0
    msg_trace: dict[str, Any] = {}
    
    codebase_path: str = ""
    testbase_path: str = ""
    generated_test_path: str = "" # if testbase_path is not empty, this generated_test_path will be the path of the testbase_path
    
    codebase_script: str = ""
    testbase_script: str = ""
    executed_output: str = ""
    exist_error: bool = False
    
    raw_generated_test_script: str = ""
    
    not_error_best_generated_test_script: str = ""
    best_score_generated_test_script: str = ""
    first_generated_test_script: str = ""
    
    not_error_best_score: float = 0.0 
    best_score: float = 0.0 
    first_score: float = 0.0
    
    use_planner: bool = True
    stream : bool = True
    
