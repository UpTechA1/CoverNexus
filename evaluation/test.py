# %%
import os
import sys
import json, time
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import traceback
from dotenv import load_dotenv
import shutil
load_dotenv()
# %% [markdown]
# # Settings

# %%
config = {
    "planner": "claude", # gpt35, gpt4, claude, deepseek, codeqwen
    "generator": "claude", # gpt35, gpt4, claude, deepseek, codeqwen
    "use_testbase": False,
    "desired_coverage": 100,
    "max_iteration": 5,
    "use_planner": True,
    "stream": True, #-> false for opensource 
    "range": range(164)
}

# %% [markdown]
# # Initilize Folder

# %%
# Get the current working directory
current_directory = os.getcwd()
print("Current Directory:", current_directory)

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Print the parent directory
print("Parent Directory:", parent_directory)

sys.path.append(parent_directory) # go to parent dir

# %%
# import glob
# ipynb_file = glob.glob("*.ipynb")
# for i in range(len(ipynb_file)):
#     ipynb_file[i] = ipynb_file[i].replace(".ipynb", "")

# %%

# for i in range(len(ipynb_file)):
# if ipynb_file[i].split("_")[-1] == config["planner"]:
temp_path = parent_directory+f"/covernexus/temp_{config['planner']}/"
os.makedirs(temp_path, exist_ok=True)
codebase_path = temp_path + "codebase.py"
generated_test_path = temp_path + "generated_test.py"
if config["use_testbase"]:
    testbase_path = temp_path + "testbase.py"
else:
    testbase_path = ""
print("Testbase Path:",generated_test_path )
# %% [markdown]
# # Intialize model

# %%

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = 'False'
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

anth_api = os.getenv("ANTHROPIC_API_KEY")
claude_llm = ChatAnthropic(api_key=anth_api, 
                           model="claude-3-5-sonnet-20240620", 
                           temperature=0,
                           max_tokens=4000,
                            max_retries=8)


# %%
class LLM:
    def __init__(self, model, tokenizer, name):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        
    def invoke(self, prompt:str):
        if self.name == "codeqwen":
            messages = [
                {"role": "user", "content": prompt + " Using import unitest library"}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=4000
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        
        elif self.name == "deepseek":
            messages=[
                { 'role': 'user', 'content': prompt+ " Using import unitest library"}
            ]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            # tokenizer.eos_token_id is the id of <|EOT|> token
            outputs = model.generate(inputs, max_new_tokens=4000, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


# %%

from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
import torch
if config['planner'] == "codeqwen":
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/CodeQwen1.5-7B-Chat",
        torch_dtype="auto",
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")
    codeqwen = LLM(model, tokenizer, config['planner'])
    
elif config['planner'] == "deepseek":
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map = "cuda:0")
    deepseek = LLM(model, tokenizer, config['planner'])


# %%
if config["planner"] == "gpt4":
    llm1 = gpt4_llm
elif config["planner"] == "gpt35":
    llm1 = gpt35_llm
elif config["planner"] == "claude":
    llm1 = claude_llm

elif config["planner"] == "deepseek":
    llm1 = deepseek
    
elif config["planner"] == "codeqwen":
    llm1 = codeqwen




# %%
if config["generator"] == "gpt4":
    llm2 = gpt4_llm
elif config["generator"] == "gpt35":
    llm2 = gpt35_llm
elif config["generator"] == "claude":
    llm2 = claude_llm
elif config["generator"] == "deepseek":
    llm2 = deepseek
elif config["generator"] == "codeqwen":
    llm2 = codeqwen

# %%
from covernexus.covernexus import CoverNexus
from covernexus.utils import *
graph = CoverNexus(llm1, llm2)

# %%

logger.info(f"Model used for planner: {llm1}, Model used for generator: {llm2}")
for i in config["range"]: 
    if config["use_planner"]:
        if config["use_testbase"]:
            folder_dir = parent_directory + f'/data/HumanEvalCoverageMultiAgent_{config["planner"]}_{config["generator"]}_testbase'
            data_dir = parent_directory +  f'/data/HumanEvalCoverageTestBase'
            os.makedirs(folder_dir, exist_ok=True)

        else:
            folder_dir = parent_directory + f'/data/HumanEvalCoverageMultiAgent_{config["planner"]}_{config["generator"]}'
            data_dir = parent_directory +  f'/data/HumanEvalCoverage'
            os.makedirs(folder_dir, exist_ok=True)

    else:
        if config["use_testbase"]:
            folder_dir = parent_directory + f'/data/HumanEvalCoverageSingle_{config["generator"]}_testbase'
            data_dir = parent_directory +  f'/data/HumanEvalCoverageTestBase'
            os.makedirs(folder_dir, exist_ok=True)

        else:
            folder_dir = parent_directory + f'/data/HumanEvalCoverageSingle_{config["generator"]}'
            data_dir = parent_directory +  f'/data/HumanEvalCoverage'
            os.makedirs(folder_dir, exist_ok=True)
            
    state_config = {"desired_coverage": config["desired_coverage"], "max_iteration": config["max_iteration"]}

    
    with open(data_dir+f'/{i}.json', 'r') as f:
        logger.info(f"==========================={i}======================")
        
        data = json.load(f)
        function = data['prompt'] + data['canonical_solution']
        with open(codebase_path, 'w') as out:
            out.write(function)
        if testbase_path != "":
            with open(testbase_path, 'w') as out:
                out.write(data["testbase"])
                state_config["current_testbase_coverage"] = data["coverage_testbase"]
        
        state = {
                'thoughts':[], 
                "config": state_config, 
                "instruction": "", 
                "current_iteration": 0, 
                "msg_trace": {},
                

                "codebase_path": codebase_path, 
                'testbase_path': testbase_path, 
                'generated_test_path': generated_test_path, 
                
                "codebase_script":"",
                "testbase_script":"",
                "executed_output": "",
                "exist_error": False,
                "raw_generated_test_script":"",
                
                "not_error_best_generated_test_script":"",
                "best_score_generated_test_script": "",
                "first_generated_test_script": "",
                
                "not_error_best_score" : 0.0 ,
                "best_score" : 0.0 ,
                "first_score" : 0.0,
               
                
                'executed_output': 'coverage score: None',
                'use_planner': config["use_planner"],
                
                'stream': config["stream"],
                
            }
                
        try:
            state = graph(state)
            # time.sleep(6)
  
        except Exception as e:
            logger.error(f"Error in generating test script: {e}")
            logger.error(traceback.format_exc())
            data['exception'] = str(e)
            
        data['num_iterations_used'] = state['current_iteration']
        if config["use_testbase"] and config["desired_coverage"] <= data["coverage_testbase"]:
            data["msg_trace"] = {
                    "1":{
                        "generated_test_script": data["testbase"],
                        "coverage": data["coverage_testbase"],
                        "exist_error": False,
                        "executed_output": ""
                    }
                } 
            data['best_score_generated_test_script'] = data["testbase"]
            data['first_generated_test_script'] = data["testbase"]
            data['not_error_best_generated_test_script'] = data["testbase"]

   
        else:
            data['msg_trace'] = state['msg_trace']
            data['best_score_generated_test_script'] = state['best_score_generated_test_script']
            data['first_generated_test_script'] = state['first_generated_test_script']
            data['not_error_best_generated_test_script'] = state['not_error_best_generated_test_script']

   
        
        data['exist_error'] = state['exist_error']
        data['best_score'] = state['best_score'] #for checking the first generation performance
        data['first_score'] = state['first_score'] # for checking multiagent system performance
        data['not_error_best_score'] = state['not_error_best_score']
        
        data['exist_not_error'] = False # have error
        for key in data['msg_trace']:
            if not data['msg_trace'][key]["exist_error"]:
                data['exist_not_error'] = True # not error
                break


    with open(folder_dir + f'/{i}.json', 'w') as out:
        json.dump(data, out, indent=4)

# remove folder at temp path
shutil.rmtree(temp_path)


