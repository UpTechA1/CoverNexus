# CoverNexus: Multi-Agent LLM System for Automated Code Coverage Enhancement

CoverNexus is an innovative multi-agent system that leverages Large Language Models (LLMs) to enhance code coverage through automated unit test generation. Our system features a flexible architecture that combines LLMs with specialized testing components, surpassing current methods in both coverage and correctness. 

CoverNexus has been rigorously evaluated using **CoverBench**, a new benchmark based on **HumanEval**, specifically designed for assessing test generation and coverage improvement. Our experiments show that CoverNexus achieves remarkable results, with **GPT-4** delivering **99.91% coverage** and **77.44% correctness** in multi-agent setups. Notably, closed-source models perform better in multi-agent configurations, while open-source models excel in single-agent scenarios. This research offers significant insights into the trade-offs between coverage and correctness, contributing to AI-assisted software testing and more efficient software development processes.

---

## Project Structure

The repository is organized into the following key folders and files:

1. **covernexus**: Contains the core implementation of the model, along with a demo package for storing related files.
2. **data**: Stores the original **HumanEval** and **CoverBench** datasets, along with CoverNexus evaluation results. This folder also includes `parsing.ipynb` for creating CoverBench, and additional analysis notebooks to compute results.
3. **evaluation**: Contains logs and the script `test.py` for running experiments.
4. **package_demo**: Includes the demo files such as codebase, test base, test generation files, and more.
5. **Others**: Contains environment variables, required libraries, and the main application files.

---

## How to Use

### Step 1: Install Dependencies

Ensure all necessary packages are installed by running:

```bash
conda create -n covernexus python=3.10.10 -y
conda activate covernexus
pip install -r requirements.txt
```

### Step 2: Setup API Keys

Currently, CoverNexus supports interactions with GPT-3.5, GPT-4, and Claude through the demo interface. For other open-source models or specific versions, you will need to run the experiments in the `evaluation` folder.

To add API keys for GPT and Claude, populate the `.env` file with your keys:

```plaintext
OPENAI_API_KEY = "your-openai-api-key"
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

### Step 3: Run the Demo

#### For Claude:

```bash
python main.py \
--source-file-path 'package_demo/codebase.py' \
--test-file-output-path 'package_demo/generated_test.py' \
--overall-output-path 'package_demo/overall_output.json' \
--desired-coverage 90 \
--max-iterations 3 \
--test-lead 'claude' \
--test-generator 'claude' \
--stream True
```

#### For GPT-3.5:

```bash
python main.py \
--source-file-path 'package_demo/codebase.py' \
--test-file-path 'package_demo/testbase.py' \
--test-file-output-path 'package_demo/generated_test.py' \
--overall-output-path 'package_demo/overall_output.json' \
--desired-coverage 90 \
--max-iterations 3 \
--test-lead 'gpt35' \
--test-generator 'gpt35' \
--stream True
```

### Command Parameters:
- **source-file-path**: Path to the file for which you want to generate coverage.
- **test-file-path** *(optional)*: Path to a human-written test file for augmenting coverage.
- **test-file-output-path**: Path where the generated tests will be saved.
- **overall-output-path**: Path for saving results after the test generation process.
- **desired-coverage**: Target coverage percentage (e.g., 90).
- **max-iterations**: Number of iterations to loop and increase coverage.
- **test-lead**: Model used by the test lead agent (e.g., `claude`, `gpt35`, `gpt4`).
- **test-generator**: Model used by the test generator agent.
- **stream**: Set to `True` to enable stream mode for real-time updates.

---

## Citation

If you use CoverNexus in your research, please cite our paper:

**Title**: *CoverNexus: Multi-Agent LLM System for Automated Code Coverage Enhancement*  
**Authors**: Thiem Nguyen Ba, Binh Nguyen Thanh, Trung Tran Viet  
**Affiliation**: Hanoi University of Science and Technology, Vietnam  
**Contact**: [thiem.nb214931@sis.hust.edu.vn](mailto:thiem.nb214931@sis.hust.edu.vn), [binh.nt210106@sis.hust.edu.vn](mailto:binh.nt210106@sis.hust.edu.vn), [trung.tranviet@hust.edu.vn](mailto:trung.tranviet@hust.edu.vn)  
**URL**: [Springer LNCS](http://www.springer.com/gp/computer-science/lncs)
