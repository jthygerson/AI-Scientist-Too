"""
This script implements an automated experimentation system for AI research. It manages the process of running multiple experiments, analyzing results, and generating plots based on an initial idea.

Key components and functionality:
1. Experiment execution: Runs experiments using 'experiment.py' in the specified folder.
2. Result analysis: Processes experiment results from 'final_info.json' files.
3. Plotting: Generates plots using 'plot.py' in the experiment folder.
4. AI-driven experimentation: Uses an AI coder to plan and modify experiments based on results.
5. Error handling and logging: Manages timeouts, errors, and provides colored console output.

Input files:
- ai_scientist/perform_experiments.py (this file)
- <folder_name>/experiment.py (experiment script, copied to run_X.py for each run)
- <folder_name>/plot.py (plotting script)
- <folder_name>/run_X/final_info.json (results for each experiment run)

Output files and locations:
- <folder_name>/run_X/ (output directory for each experiment run)
- <folder_name>/run_X.py (copy of experiment script for each run)
- <folder_name>/notes.txt (experiment notes and plot descriptions)
- <folder_name>/*.png (generated plot images)

The script interacts with an AI coder (not defined in this file) to plan experiments, analyze results, and generate plots. It uses a maximum of 5 experiment runs and 4 iterations per run to complete the research task.
"""

# Import necessary libraries
import shutil  # For file operations
import os.path as osp  # For handling file paths
import subprocess  # For running external commands
from subprocess import TimeoutExpired  # For handling timeouts
import sys  # For system-specific parameters and functions
import json  # For working with JSON data
from colorama import init, Fore, Style  # For colored console output

# Initialize colorama for cross-platform colored output
init()

# Define constants
MAX_ITERS = 4  # Maximum number of iterations
MAX_RUNS = 5  # Maximum number of experiment runs
MAX_STDERR_OUTPUT = 1500  # Maximum length of error output to display

# Define the prompt for the AI coder
coder_prompt = """Your goal is to implement the following idea: {title}.
The proposed experiment is as follows: {idea}.
You are given a total of up to {max_runs} runs to complete the necessary experiments. You do not need to use all {max_runs}.

First, plan the list of experiments you would like to run. For example, if you are sweeping over a specific hyperparameter, plan each value you would like to test for each run.

Note that we already provide the vanilla baseline results, so you do not need to re-run it.

For reference, the baseline results are as follows:

{baseline_results}

After you complete each change, we will run the command `python experiment.py --out_dir=run_i' where i is the run number and evaluate the results.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
You can then implement the next thing on your list."""

# Function to print error messages in red
def print_error(message):
    print(f"{Fore.RED}{message}{Style.RESET_ALL}", file=sys.stderr)

# Function to print warning messages in yellow
def print_warning(message):
    print(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")

# Function to run an experiment
def run_experiment(folder_name, run_num, timeout=7200):
    try:
        # Get the absolute path of the experiment folder
        cwd = osp.abspath(folder_name)
        
        # Copy the experiment file for record-keeping
        shutil.copy(
            osp.join(folder_name, "experiment.py"),
            osp.join(folder_name, f"run_{run_num}.py"),
        )

        # Prepare the command to run the experiment
        command = [
            "python",
            "experiment.py",
            f"--out_dir=run_{run_num}",
        ]
        
        try:
            # Run the experiment command
            result = subprocess.run(
                command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
            )

            # Check for any error output
            if result.stderr:
                print_warning(f"Run {run_num} produced stderr output:")
                print(result.stderr, file=sys.stderr)

            # Check if the experiment failed
            if result.returncode != 0:
                print_error(f"Run {run_num} failed with return code {result.returncode}")
                # Remove the output directory if it exists
                if osp.exists(osp.join(cwd, f"run_{run_num}")):
                    shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
                print_error(f"Run failed with the following error {result.stderr}")
                # Truncate long error messages
                stderr_output = result.stderr
                if len(stderr_output) > MAX_STDERR_OUTPUT:
                    stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
                next_prompt = f"Run failed with the following error {stderr_output}"
            else:
                # If the experiment succeeded, process the results
                try:
                    # Read the results from the JSON file
                    with open(osp.join(cwd, f"run_{run_num}", "final_info.json"), "r") as f:
                        results = json.load(f)
                    results = {k: v["means"] for k, v in results.items()}

                    # Prepare the next prompt for the AI coder
                    next_prompt = f"""Run {run_num} completed. Here are the results:
    {results}

    Decide if you need to re-plan your experiments given the result (you often will not need to).

    Someone else will be using `notes.txt` to perform a writeup on this in the future.
    Please include *all* relevant information for the writeup on Run {run_num}, including an experiment description and the run number. Be as verbose as necessary.

    Then, implement the next thing on your list.
    We will then run the command `python experiment.py --out_dir=run_{run_num + 1}'.
    YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
    If you are finished with experiments, respond with 'ALL_COMPLETED'."""
                except FileNotFoundError:
                    print_error(f"final_info.json not found for run {run_num}")
                    next_prompt = f"Run {run_num} completed, but final_info.json was not found. Please check the experiment output."
                except json.JSONDecodeError:
                    print_error(f"Error decoding final_info.json for run {run_num}")
                    next_prompt = f"Run {run_num} completed, but there was an error decoding final_info.json. Please check the file contents."
            return result.returncode, next_prompt
        except TimeoutExpired:
            # Handle experiment timeout
            print_error(f"Run {run_num} timed out after {timeout} seconds")
            if osp.exists(osp.join(cwd, f"run_{run_num}")):
                shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
            next_prompt = f"Run timed out after {timeout} seconds"
            return 1, next_prompt
    except Exception as e:
        # Handle unexpected errors
        print_error(f"Unexpected error in run_experiment: {str(e)}")
        return 1, f"Unexpected error: {str(e)}"

# Function to run the plotting script
def run_plotting(folder_name, timeout=600):
    try:
        # Get the absolute path of the experiment folder
        cwd = osp.abspath(folder_name)
        
        # Prepare the command to run the plotting script
        command = [
            "python",
            "plot.py",
        ]
        
        try:
            # Run the plotting command
            result = subprocess.run(
                command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
            )

            # Check for any error output
            if result.stderr:
                print_warning("Plotting produced stderr output:")
                print(result.stderr, file=sys.stderr)

            # Check if plotting failed
            if result.returncode != 0:
                print_error(f"Plotting failed with return code {result.returncode}")
                next_prompt = f"Plotting failed with the following error {result.stderr}"
            else:
                next_prompt = ""
            return result.returncode, next_prompt
        except TimeoutExpired:
            # Handle plotting timeout
            print_error(f"Plotting timed out after {timeout} seconds")
            next_prompt = f"Plotting timed out after {timeout} seconds"
            return 1, next_prompt
    except Exception as e:
        # Handle unexpected errors
        print_error(f"Unexpected error in run_plotting: {str(e)}")
        return 1, f"Unexpected error: {str(e)}"

# Main function to perform experiments
def perform_experiments(idea, folder_name, coder, baseline_results) -> bool:
    try:
        # Run experiments
        current_iter = 0
        run = 1
        next_prompt = coder_prompt.format(
            title=idea["Title"],
            idea=idea["Experiment"],
            max_runs=MAX_RUNS,
            baseline_results=baseline_results,
        )
        
        # Loop through experiment runs
        while run < MAX_RUNS + 1:
            if current_iter >= MAX_ITERS:
                print_warning("Max iterations reached")
                break
            coder_out = coder.run(next_prompt)
            print(coder_out)
            if "ALL_COMPLETED" in coder_out:
                break
            return_code, next_prompt = run_experiment(folder_name, run)
            if return_code == 0:
                run += 1
                current_iter = 0
            current_iter += 1
        
        if current_iter >= MAX_ITERS:
            print_warning("Not all experiments completed.")
            return False

        # Run plotting
        current_iter = 0
        next_prompt = """
    Great job! Please modify `plot.py` to generate the most relevant plots for the final writeup. 

    In particular, be sure to fill in the "labels" dictionary with the correct names for each run that you want to plot.

    Only the runs in the `labels` dictionary will be plotted, so make sure to include all relevant runs.

    We will be running the command `python plot.py` to generate the plots.
    """
        while True:
            coder_out = coder.run(next_prompt)
            return_code, next_prompt = run_plotting(folder_name)
            current_iter += 1
            if return_code == 0 or current_iter >= MAX_ITERS:
                break
        
        # Update notes
        next_prompt = """
    Please modify `notes.txt` with a description of what each plot shows along with the filename of the figure. Please do so in-depth.

    Somebody else will be using `notes.txt` to write a report on this in the future.
    """
        coder.run(next_prompt)

        return True
    except Exception as e:
        # Handle unexpected errors
        print_error(f"Unexpected error in perform_experiments: {str(e)}")
        return False
