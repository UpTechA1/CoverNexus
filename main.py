import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv()
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
import subprocess
from covernexus.covernexus import CoverNexus
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import json
import re

class Settings:
    def __init__(self, args):
        self.args = args
        self.config =  {
            "desired_coverage": self.args.desired_coverage,
            "max_iteration": self.args.max_iterations,
            "use_planner": self.args.use_planner,
            "stream": self.args.stream,
        }

        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        anth_api = os.getenv("ANTHROPIC_API_KEY")
        claude_llm = ChatAnthropic(api_key=anth_api, 
                           model="claude-3-5-sonnet-20240620", 
                           temperature=0,
                           max_tokens=4000,
                            max_retries=8)
        
        gpt4_llm = ChatOpenAI(
                    model="gpt-4-turbo-2024-04-09",
                    temperature=0,
                    max_tokens=4000,
                    max_retries=3)

        gpt35_llm = ChatOpenAI(
                    model="gpt-3.5-turbo-0125",
                    temperature=0,
                    max_tokens=4000,
                    max_retries=3)
        
        if self.args.test_lead == "gpt4":
            self.llm1 = gpt4_llm
        elif self.args.test_lead == "gpt35":
            self.llm1 = gpt35_llm
        elif self.args.test_lead == "claude":
            self.llm1 = claude_llm

        if self.args.test_generator == "gpt4":
            self.llm2 = gpt4_llm
        elif self.args.test_generator == "gpt35":
            self.llm2 = gpt35_llm
        elif self.args.test_generator == "claude":
            self.llm2 = claude_llm

        self.graph = CoverNexus(self.llm1, self.llm2)
    
    def run(self):
        if self.args.test_file_path != "":
            with open(self.args.test_file_path, "r") as f:
                test_script = f.read()
            subprocess.run(["coverage", "run", self.args.test_file_path], check=True)
            result = subprocess.run(["coverage", "report", "-m", self.args.source_file_path], capture_output=True, text=True, check=True)
            match = re.search(f'{self.args.source_file_path}.* (\d+%)', result.stdout)
            if match:
                coverage_percentage = match.group(1)
                coverage_percentage = float(coverage_percentage[:-1])
            else:
                coverage_percentage = 0
            print(f"Initial coverage: {coverage_percentage}")
            print(result.stdout)
            self.config["current_testbase_coverage"] = coverage_percentage
            
        state = self.graph({'thoughts':[], 
            "config": self.config, 
            "instruction": "", 
            'current_iteration': 0, 
            
            'codebase_path': self.args.source_file_path, 
            'testbase_path': self.args.test_file_path, 
            'generated_test_path': self.args.test_file_output_path,

            'codebase_script': '',
            'testbase_script': '',
            'exist_error': False,
            
            'raw_generated_test_script': '',
            
            'not_error_best_generated_test_script': '',
            'best_score_generated_test_script': '',
            'first_generated_test_script':'',
    
            'not_error_best_score': 0.0,
            'best_score': 0.0,
            'first_score': 0.0,
            
            'executed_output': 'coverage score: None',
            'use_planner': self.config["use_planner"],
            'stream': self.config["stream"],
            'msg_trace': {},
        })
        

        data = dict()
        data['num_iterations_used'] = state['current_iteration']
    
        data['exist_error'] = state['exist_error']
        data['best_score'] = state['best_score'] #for checking the first generation performance
        data['first_score'] = state['first_score'] # for checking multiagent system performance
        data['not_error_best_score'] = state['not_error_best_score']
        
        data['best_score_generated_test_script'] = state['best_score_generated_test_script']
        data['first_generated_test_script'] = state['first_generated_test_script']
        data['not_error_best_generated_test_script'] = state['not_error_best_generated_test_script']
        
        data['exist_not_error'] = False # have error
        data['msg_trace'] = state['msg_trace']
        for key in data['msg_trace']:
            if not data['msg_trace'][key]["exist_error"]:
                data['exist_not_error'] = True # not error
                break
            
        if data["exist_not_error"]:
            data['output'] = state['not_error_best_generated_test_script']
            data['output_coverage'] = state['not_error_best_score']
        else:
            data['output'] = state['best_score_generated_test_script']
            data['output_coverage'] = state['best_score']
        with open(self.args.test_file_output_path, "w") as f:
            f.write(data['output'])
        with open(self.args.overall_output_path, "w") as f:
            json.dump(data, f, indent=4)
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-file-path", type=str,required=True, help="Path to the source file."
    )
    parser.add_argument(
        "--test-file-path", type=str,required=False, help="Path to the input testbase file.", default=""
    )
    parser.add_argument(
        "--test-file-output-path",
        required=True,
        help="Path to the output test file.",
        type=str,
    )
    parser.add_argument(
        "--overall-output-path",
        required=False,
        type=str,
        default="overall_output.json",
        help="Path to the overall output json file.",
    )
    parser.add_argument(
        "--desired-coverage",
        type=int,
        default=90,
        help="The desired coverage percentage. Default: %(default)s.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="The maximum number of iterations. Default: %(default)s.",
    )
    parser.add_argument(
        "--test-lead",
        type=str,
        required=True,
        help="Which model for planner agent. Default: %(default)s.",
    )
    parser.add_argument(
        "--test-generator",
        type=str,
        required=True,
        help="Which model for generator agent. Default: %(default)s.",
    )
    parser.add_argument(
        "--use-planner",
        required=False,
        type=bool,
        default=True,
        help="Whether to use planner agent or not. Default: %(default)s.",
    )
    parser.add_argument(
        "--stream",
        required=False,
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="Whether to stream the output or not. Close-source like gpt4 or claude should be true while open-source like deepseek should be false. Default: %(default)s.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    agent = Settings(args)
    agent.run()


if __name__ == "__main__":
    main()
    

