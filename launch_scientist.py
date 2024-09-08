"""
This script, launch_scientist.py, orchestrates an AI-driven scientific research process. It automates the generation, 
evaluation, and documentation of research ideas in a specified experimental domain. The script performs the following 
key functions:

1. Idea Generation: Uses AI models to generate novel research ideas based on a given experiment template.
2. Novelty Check: Evaluates the novelty of generated ideas to ensure originality.
3. Experiment Simulation: Simulates experiments for each novel idea, modifying existing experiment code.
4. Results Analysis: Analyzes and compares experimental results with baseline data.
5. Paper Writing: Generates a LaTeX-formatted scientific paper documenting each experiment.
6. Peer Review Simulation: Simulates a peer review process for the generated papers.
7. Paper Improvement: Optionally improves the paper based on the simulated peer review.

Input Files:
- Template files in ./templates/{experiment_name}/: Provide the base experiment setup.
- ./templates/{experiment_name}/run_0/final_info.json: Contains baseline results for comparison.
- ./templates/{experiment_name}/experiment.py: The base experiment code to be modified.
- ./templates/{experiment_name}/plot.py: Code for visualizing experimental results.
- ./templates/{experiment_name}/latex/template.tex: LaTeX template for paper generation.

Output Files and Locations:
- ./results/{experiment_name}/{timestamp}_{idea_name}/: Directory for each processed idea, containing:
  - Modified experiment.py and plot.py
  - notes.txt: Detailed notes on the experiment
  - log.txt: Execution log
  - {idea_name}.pdf: Generated scientific paper
  - review.txt: Simulated peer review
  - {idea_name}_improved.pdf: Improved version of the paper (if improvement option is enabled)
  - review_improved.txt: Review of the improved paper
- ./templates/{experiment_name}/ideas.json: Stores all generated ideas, including novelty assessment

The script supports various AI models, parallel processing on multiple GPUs, and command-line arguments for 
customization. It uses libraries such as OpenAI, Anthropic, and custom modules (aider.coders, aider.models, 
ai_scientist) for AI interactions and scientific process simulation.
"""

# This script runs AI scientist experiments. It generates ideas, performs experiments,
# writes up results, and reviews the output.

# First, we import all the necessary libraries and modules
import openai  # For AI model interactions
import os.path as osp  # For handling file paths
import shutil  # For file operations
import json  # For working with JSON data
import argparse  # For parsing command-line arguments
import multiprocessing  # For parallel processing
import torch  # For GPU operations
import os  # For operating system related operations
import time  # For time-related functions
import sys  # For system-specific parameters and functions
import requests  # Add this import for making HTTP requests to OLLAMA

from aider.coders import Coder  # Custom module for code generation
from aider.models import Model  # Custom module for AI models
from aider.io import InputOutput  # Custom module for input/output operations
from datetime import datetime  # For working with dates and times
# Import custom functions from ai_scientist module
from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_writeup import perform_writeup, generate_latex
from ai_scientist.perform_review import perform_review, load_paper, perform_improvement
from colorama import init, Fore, Style  # For colored terminal output

# Initialize colorama for cross-platform colored output
init()

# Set a constant for the number of reflections
NUM_REFLECTIONS = 3

# Define a function to print the current time
def print_time():
    # This prints the current date and time in a specific format
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Define a function to parse command-line arguments
def parse_arguments():
    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    
    # Add various command-line arguments
    # Each argument is defined with a name, type, default value, and help text
    parser.add_argument("--skip-idea-generation", action="store_true", help="Skip idea generation and load existing ideas")
    parser.add_argument("--skip-novelty-check", action="store_true", help="Skip novelty check and use existing ideas")
    parser.add_argument("--experiment", type=str, default="nanoGPT", help="Experiment to run AI Scientist on.")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20240620", choices=[
        "claude-3-5-sonnet-20240620",
        "gpt-4o-2024-05-13",
        "deepseek-coder-v2-0724",
        "llama3.1-405b",
        # Anthropic Claude models via Amazon Bedrock
        "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
        "bedrock/anthropic.claude-3-opus-20240229-v1:0",
        # Anthropic Claude models Vertex AI
        "vertex_ai/claude-3-opus@20240229",
        "vertex_ai/claude-3-5-sonnet@20240620",
        "vertex_ai/claude-3-sonnet@20240229",
        "vertex_ai/claude-3-haiku@20240307",
        # OLLAMA models
        "ollama/deepseek-coder-v2",
        "ollama/phi3.5",
        "ollama/llama3.1",
        "ollama/gemma2",
    ], help="Model to use for AI Scientist.")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="URL for OLLAMA API (default: http://localhost:11434)")
    parser.add_argument("--writeup", type=str, default="latex", choices=["latex"], help="What format to use for writeup")
    parser.add_argument("--parallel", type=int, default=0, help="Number of parallel processes to run. 0 for sequential execution.")
    parser.add_argument("--improvement", action="store_true", help="Improve based on reviews.")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.")
    parser.add_argument("--num-ideas", type=int, default=50, help="Number of ideas to generate")
    # ... (other arguments)
    
    # Parse the arguments and return them
    return parser.parse_args()

# Define a function to get available GPUs
def get_available_gpus(gpu_ids=None):
    # If specific GPU IDs are provided, use those
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    # Otherwise, return all available GPUs
    return list(range(torch.cuda.device_count()))

# Define a worker function for parallel processing
def worker(queue, base_dir, results_dir, model, client, client_model, writeup, improvement, gpu_id):
    # Set the GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Worker {gpu_id} started.")
    
    # Continuously process ideas from the queue
    while True:
        idea = queue.get()
        if idea is None:
            break  # Exit if we receive None (signals end of work)
        # Process the idea
        success = do_idea(base_dir, results_dir, idea, model, client, client_model, writeup, improvement, log_file=True)
        print(f"Completed idea: {idea['Name']}, Success: {success}")
    
    print(f"Worker {gpu_id} finished.")

# Define the main function to process an idea
def do_idea(base_dir, results_dir, idea, model, client, client_model, writeup, improvement, log_file=False):
    # This function handles the entire process for a single idea
    # It creates a project folder, performs experiments, generates a writeup, and reviews the paper
    
    # Create project folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)
    with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")
    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")
    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log
    try:
        print_time()
        print(f"*Starting idea: {idea_name}*")
        
        # Perform experiments
        fnames = [exp_file, vis_file, notes]
        io = InputOutput(yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt")
        if model == "deepseek-coder-v2-0724":
            main_model = Model("deepseek/deepseek-coder")
        elif model == "llama3.1-405b":
            main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        else:
            main_model = Model(model)
        coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

        print_time()
        print(f"*Starting Experiments*")
        try:
            success = perform_experiments(idea, folder_name, coder, baseline_results)
            if not success:
                print_warning(f"Experiments failed for idea {idea_name}")
                return False
        except Exception as e:
            print_error(f"Error during experiments for idea {idea_name}: {e}")
            return False

        print_time()
        print(f"*Starting Writeup*")
        # Perform writeup
        if writeup == "latex":
            writeup_file = osp.join(folder_name, "latex", "template.tex")
            fnames = [exp_file, writeup_file, notes]
            if model == "deepseek-coder-v2-0724":
                main_model = Model("deepseek/deepseek-coder")
            elif model == "llama3.1-405b":
                main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
            else:
                main_model = Model(model)
            coder = Coder.create(
                main_model=main_model,
                fnames=fnames,
                io=io,
                stream=False,
                use_git=False,
                edit_format="diff",
            )
            try:
                perform_writeup(idea, folder_name, coder, client, client_model)
                print_success("Writeup completed successfully")
            except Exception as e:
                print_error(f"Failed to perform writeup: {e}")
                return False
        else:
            print_error(f"Writeup format {writeup} not supported.")
            return False

        print_time()
        print(f"*Starting Review*")
        # Review paper
        if writeup == "latex":
            try:
                paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")
                review = perform_review(
                    paper_text,
                    model="gpt-4o-2024-05-13",
                    client=openai.OpenAI(),
                    num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review.txt"), "w") as f:
                    f.write(json.dumps(review, indent=4))
                print_success("Review completed successfully")
            except Exception as e:
                print_error(f"Failed to perform review: {e}")
                # Continue execution despite review failure

        # Improve writeup if needed
        if writeup == "latex" and improvement:
            print_time()
            print(f"*Starting Improvement*")
            try:
                perform_improvement(review, coder)
                generate_latex(
                    coder, folder_name, f"{folder_name}/{idea['Name']}_improved.pdf"
                )
                paper_text = load_paper(f"{folder_name}/{idea['Name']}_improved.pdf")
                review = perform_review(
                    paper_text,
                    model="gpt-4o-2024-05-13",
                    client=openai.OpenAI(),
                    num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review_improved.txt"), "w") as f:
                    f.write(json.dumps(review))
                print_success("Improvement completed successfully")
            except Exception as e:
                print_error(f"Failed to perform improvement: {e}")
                # Continue execution despite improvement failure

        print_success(f"Successfully processed idea: {idea_name}")
        return True
    except Exception as e:
        print_error(f"Failed to evaluate idea {idea_name}: {str(e)}")
        return False
    finally:
        print("FINISHED IDEA")
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()

# Define a function to print success messages
def print_success(message):
    print(Fore.GREEN + message + Style.RESET_ALL)

# Define a function to print error messages
def print_error(message):
    print(Fore.RED + message + Style.RESET_ALL)

# Define a function to print warning messages
def print_warning(message):
    print(Fore.YELLOW + message + Style.RESET_ALL)

# Main execution block
if __name__ == "__main__":
    # This block runs when the script is executed directly (not imported)
    
    # Parse command-line arguments
    args = parse_arguments()

    # Check available GPUs and adjust parallel processes if necessary
    available_gpus = get_available_gpus(args.gpus)
    if args.parallel > len(available_gpus):
        print(
            f"Warning: Requested {args.parallel} parallel processes, but only {len(available_gpus)} GPUs available. Adjusting to {len(available_gpus)}."
        )
        args.parallel = len(available_gpus)

    print(f"Using GPUs: {available_gpus}")

    # Create client based on the chosen model
    if args.model == "claude-3-5-sonnet-20240620":
        import anthropic

        print(f"Using Anthropic API with model {args.model}.")
        client_model = "claude-3-5-sonnet-20240620"
        client = anthropic.Anthropic()
    elif args.model.startswith("bedrock") and "claude" in args.model:
        import anthropic

        # Expects: bedrock/<MODEL_ID>
        client_model = args.model.split("/")[-1]

        print(f"Using Amazon Bedrock with model {client_model}.")
        client = anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
        )
    elif args.model.startswith("vertex_ai") and "claude" in args.model:
        import anthropic

        # Expects: vertex_ai/<MODEL_ID>
        client_model = args.model.split("/")[-1]

        print(f"Using Vertex AI with model {client_model}.")
        client = anthropic.AnthropicVertex()
    elif args.model == "gpt-4o-2024-05-13":
        import openai

        print(f"Using OpenAI API with model {args.model}.")
        client_model = "gpt-4o-2024-05-13"
        client = openai.OpenAI()
    elif args.model == "deepseek-coder-v2-0724":
        import openai

        print(f"Using OpenAI API with {args.model}.")
        client_model = "deepseek-coder-v2-0724"
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
        )
    elif args.model == "llama3.1-405b":
        import openai

        print(f"Using OpenAI API with {args.model}.")
        client_model = "meta-llama/llama-3.1-405b-instruct"
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    elif args.model.startswith("ollama/"):
        import requests
        ollama_model = args.model.split("/")[1]
        print(f"Using OLLAMA API with model {ollama_model} at {args.ollama_url}")
        client_model = args.model  # Keep the full "ollama/model_name" format
        
        # Test the connection and model availability
        try:
            response = requests.post(f"{args.ollama_url}/api/generate", json={
                "model": ollama_model,
                "prompt": "Hello, are you available?",
                "stream": False
            })
            response.raise_for_status()
            print("Successfully connected to OLLAMA API and model is available.")
            print(f"OLLAMA response: {response.json()['response']}")
        except requests.RequestException as e:
            print_error(f"Failed to connect to OLLAMA API or model not available: {e}")
            print(f"Response content: {e.response.content if hasattr(e, 'response') else 'No response content'}")
            sys.exit(1)
        
        # Create a simple client object to pass around
        class OllamaClient:
            def __init__(self, base_url, model):
                self.base_url = base_url
                self.model = model
            
            def generate(self, prompt):
                response = requests.post(f"{self.base_url}/api/generate", json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                })
                response.raise_for_status()
                return response.json()["response"]
        
        client = OllamaClient(args.ollama_url, ollama_model)

    else:
        raise ValueError(f"Model {args.model} not supported.")

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)
    try:
        ideas = generate_ideas(
            base_dir,
            client=client,
            model=client_model,
            skip_generation=args.skip_idea_generation,
            max_num_generations=args.num_ideas,
            num_reflections=NUM_REFLECTIONS,
        )
        print_success("Ideas generated successfully")
    except Exception as e:
        print_error(f"Failed to generate ideas: {e}")
        ideas = []

    try:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=client,
            model=client_model,
        )
        print_success("Idea novelty checked successfully")
    except Exception as e:
        print_error(f"Failed to check idea novelty: {e}")

    try:
        with open(osp.join(base_dir, "ideas.json"), "w") as f:
            json.dump(ideas, f, indent=4)
        print_success("Ideas saved to JSON file")
    except Exception as e:
        print_error(f"Failed to save ideas to JSON: {e}")

    novel_ideas = [idea for idea in ideas if idea["novel"]]
    # novel_ideas = list(reversed(novel_ideas))

    if args.parallel > 0:
        print(f"Running {args.parallel} parallel processes")
        queue = multiprocessing.Queue()
        for idea in novel_ideas:
            queue.put(idea)

        processes = []
        for i in range(args.parallel):
            gpu_id = available_gpus[i % len(available_gpus)]
            p = multiprocessing.Process(
                target=worker,
                args=(
                    queue,
                    base_dir,
                    results_dir,
                    args.model,
                    client,
                    client_model,
                    args.writeup,
                    args.improvement,
                    gpu_id,
                ),
            )
            p.start()
            time.sleep(150)
            processes.append(p)

        # Signal workers to exit
        for _ in range(args.parallel):
            queue.put(None)

        for p in processes:
            p.join()

        print("All parallel processes completed.")
    else:
        for idea in novel_ideas:
            print(f"Processing idea: {idea['Name']}")
            try:
                success = do_idea(
                    base_dir,
                    results_dir,
                    idea,
                    args.model,
                    client,
                    client_model,
                    args.writeup,
                    args.improvement,
                )
                if success:
                    print_success(f"Completed idea: {idea['Name']}")
                else:
                    print_warning(f"Idea processing incomplete: {idea['Name']}")
            except Exception as e:
                print_error(f"Failed to evaluate idea {idea['Name']}: {str(e)}")

    print_success("All ideas evaluated.")
