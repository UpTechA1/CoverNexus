import logging
import sys
import os

GENERATOR = "GENERATING_AGENT"
FINISH = "FINISH"
PLANNER = "PLANNING_AGENT"
EDITOR = "EDITING_TOOL"
EXECUTOR = "EXECUTING_TOOL"
CODEBASE = "codebase.py"

logger = logging.getLogger()
logger.handlers = []

# Set level and add new handlers
logger.setLevel(logging.INFO)

# Handler for console output
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s\n')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Handler for file output
os.makedirs('log', exist_ok=True)
file_handler = logging.FileHandler(os.path.join(os.getcwd(), 'log/multiagents.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

