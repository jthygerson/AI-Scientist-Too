"""
This script generates and evaluates research ideas for AI experiments using large language models.

Key functionalities:
1. Generates research ideas based on a given experiment and task description
2. Iteratively refines ideas through multiple rounds of reflection
3. Checks the novelty of ideas by searching academic literature
4. Supports various LLM backends (OpenAI, Anthropic, DeepSeek, etc.)

Input files:
- templates/{experiment}/prompt.json: Contains task description and system prompts
- templates/{experiment}/experiment.py: Contains the code for the experiment
- templates/{experiment}/seed_ideas.json: Initial ideas to seed the generation process

Output files:
- results/{experiment}/ideas.json: Stores generated ideas with novelty assessments

Environment variables used:
- S2_API_KEY: Semantic Scholar API key for literature search
- OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY: API keys for various LLM providers

The script uses command-line arguments to control its behavior, including:
- Selecting the experiment and LLM model
- Skipping idea generation
- Enabling novelty checking

Functions:
- generate_ideas: Main function for idea generation
- generate_next_idea: Generates a single new idea
- check_idea_novelty: Assesses the novelty of ideas using literature search
- search_for_papers: Queries the Semantic Scholar API for relevant papers

Note: This script requires several external libraries including openai, anthropic, 
requests, backoff, and colorama.
"""

# Import necessary libraries
import json  # For working with JSON data
import os  # For interacting with the operating system
import os.path as osp  # For handling file paths
import time  # For adding delays
from typing import List, Dict, Union  # For type hinting
from ai_scientist.llm import get_response_from_llm, extract_json_between_markers  # Custom functions for interacting with language models

import requests  # For making HTTP requests
import backoff  # For implementing retry logic
from colorama import Fore, Style, init

# Initialize colorama for colored terminal output
init(autoreset=True)

# Get the Semantic Scholar API key from environment variables
S2_API_KEY = os.getenv("S2_API_KEY")

# Define prompts for idea generation
idea_first_prompt = """{task_description}
<experiment.py>
{code}
</experiment.py>

Here are the ideas that you have already generated:

'''
{prev_ideas_string}
'''

Come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided.
Note that you will not have access to any additional resources or datasets.
Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).

Be cautious and realistic on your ratings.
This JSON will be automatically parsed, so ensure the format is precise.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""

idea_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your idea.
Stick to the spirit of the original idea unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""


# GENERATE IDEAS
def generate_ideas(
    base_dir,
    client,
    model,
    skip_generation=False,
    max_num_generations=20,
    num_reflections=5,
):
    try:
        # If skip_generation is True, try to load existing ideas from a file
        if skip_generation:
            try:
                with open(osp.join(base_dir, "ideas.json"), "r") as f:
                    ideas = json.load(f)
                print("Loaded existing ideas:")
                for idea in ideas:
                    print(idea)
                return ideas
            except FileNotFoundError:
                print("No existing ideas found. Generating new ideas.")
            except json.JSONDecodeError:
                print("Error decoding existing ideas. Generating new ideas.")

        # Initialize an empty list to store idea strings
        idea_str_archive = []
        
        # Load seed ideas from a file
        with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)
        for seed_idea in seed_ideas:
            idea_str_archive.append(json.dumps(seed_idea))

        # Read the experiment code from a file
        with open(osp.join(base_dir, "experiment.py"), "r") as f:
            code = f.read()

        # Load the prompt from a JSON file
        with open(osp.join(base_dir, "prompt.json"), "r") as f:
            prompt = json.load(f)

        idea_system_prompt = prompt["system"]

        # Generate ideas
        for _ in range(max_num_generations):
            print()
            print(f"Generating idea {_ + 1}/{max_num_generations}")
            try:
                # Combine all previous ideas into a single string
                prev_ideas_string = "\n\n".join(idea_str_archive)

                msg_history = []
                print(f"Iteration 1/{num_reflections}")
                
                # Get a response from the language model
                text, msg_history = get_response_from_llm(
                    idea_first_prompt.format(
                        task_description=prompt["task_description"],
                        code=code,
                        prev_ideas_string=prev_ideas_string,
                        num_reflections=num_reflections,
                    ),
                    client=client,
                    model=model,
                    system_message=idea_system_prompt,
                    msg_history=msg_history,
                )
                
                # Extract JSON from the language model's output
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"
                print(json_output)

                # Iteratively improve the idea
                if num_reflections > 1:
                    for j in range(num_reflections - 1):
                        print(f"Iteration {j + 2}/{num_reflections}")
                        text, msg_history = get_response_from_llm(
                            idea_reflection_prompt.format(
                                current_round=j + 2, num_reflections=num_reflections
                            ),
                            client=client,
                            model=model,
                            system_message=idea_system_prompt,
                            msg_history=msg_history,
                        )
                        json_output = extract_json_between_markers(text)
                        assert json_output is not None, "Failed to extract JSON from LLM output"
                        print(json_output)

                        if "I am done" in text:
                            print(f"Idea generation converged after {j + 2} iterations.")
                            break

                # Add the new idea to the archive
                idea_str_archive.append(json.dumps(json_output))
            except Exception as e:
                print(f"{Fore.RED}Failed to generate idea: {e}{Style.RESET_ALL}")
                continue

        # Save all generated ideas to a file
        ideas = []
        for idea_str in idea_str_archive:
            ideas.append(json.loads(idea_str))

        with open(osp.join(base_dir, "ideas.json"), "w") as f:
            json.dump(ideas, f, indent=4)

        return ideas
    except Exception as e:
        print(f"{Fore.RED}Error in generate_ideas: {e}{Style.RESET_ALL}")
    
    return ideas

# Function to generate ideas in an open-ended manner
def generate_next_idea(
    base_dir,
    client,
    model,
    prev_idea_archive=[],
    num_reflections=5,
    max_attempts=10,
):
    try:
        idea_archive = prev_idea_archive
        original_archive_size = len(idea_archive)

        print(f"Generating idea {original_archive_size + 1}")

        if len(prev_idea_archive) == 0:
            print(f"First iteration, taking seed ideas")
            # seed the archive on the first run with pre-existing ideas
            with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
                seed_ideas = json.load(f)
            for seed_idea in seed_ideas[:1]:
                idea_archive.append(seed_idea)
        else:
            with open(osp.join(base_dir, "experiment.py"), "r") as f:
                code = f.read()
            with open(osp.join(base_dir, "prompt.json"), "r") as f:
                prompt = json.load(f)
            idea_system_prompt = prompt["system"]

            for _ in range(max_attempts):
                try:
                    idea_strings = []
                    for idea in idea_archive:
                        idea_strings.append(json.dumps(idea))
                    prev_ideas_string = "\n\n".join(idea_strings)

                    msg_history = []
                    print(f"Iteration 1/{num_reflections}")
                    text, msg_history = get_response_from_llm(
                        idea_first_prompt.format(
                            task_description=prompt["task_description"],
                            code=code,
                            prev_ideas_string=prev_ideas_string,
                            num_reflections=num_reflections,
                        )
                        + """
Completed ideas have an additional "Score" field which indicates the assessment by an expert ML reviewer.
This is on a standard 1-10 ML conference scale.
Scores of 0 indicate the idea failed either during experimentation, writeup or reviewing.
""",
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    ## PARSE OUTPUT
                    json_output = extract_json_between_markers(text)
                    assert json_output is not None, "Failed to extract JSON from LLM output"
                    print(json_output)

                    # Iteratively improve task.
                    if num_reflections > 1:
                        for j in range(num_reflections - 1):
                            print(f"Iteration {j + 2}/{num_reflections}")
                            text, msg_history = get_response_from_llm(
                                idea_reflection_prompt.format(
                                    current_round=j + 2, num_reflections=num_reflections
                                ),
                                client=client,
                                model=model,
                                system_message=idea_system_prompt,
                                msg_history=msg_history,
                            )
                            ## PARSE OUTPUT
                            json_output = extract_json_between_markers(text)
                            assert (
                                json_output is not None
                            ), "Failed to extract JSON from LLM output"
                            print(json_output)

                            if "I am done" in text:
                                print(
                                    f"Idea generation converged after {j + 2} iterations."
                                )
                                break

                    idea_archive.append(json_output)
                    break
                except Exception as e:
                    print(f"{Fore.YELLOW}Failed to generate idea: {e}{Style.RESET_ALL}")
                    continue

        ## SAVE IDEAS
        with open(osp.join(base_dir, "ideas.json"), "w") as f:
            json.dump(idea_archive, f, indent=4)

        return idea_archive
    except Exception as e:
        print(f"{Fore.RED}Error in generate_next_idea: {e}{Style.RESET_ALL}")
    
    return idea_archive

# Function to handle backoff logging
def on_backoff(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )

# Function to search for papers using the Semantic Scholar API
@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    if not query:
        return None
    rsp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"X-API-KEY": S2_API_KEY},
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        },
    )
    print(f"Response Status Code: {rsp.status_code}")
    print(
        f"Response Content: {rsp.text[:500]}"
    )  # Print the first 500 characters of the response content
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    time.sleep(1.0)
    if not total:
        return None

    papers = results["data"]
    return papers

# Define system message for novelty checking
novelty_system_msg = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top 10 results for any search query will be presented to you with the abstracts.

You will be given {num_rounds} to decide on the paper, but you do not need to use them all.
At any round, you may exit early and decide on the novelty of the idea.
Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

{task_description}
<experiment.py>
{code}
</experiment.py>
"""

# Define prompt for novelty checking
novelty_prompt = '''Round {current_round}/{num_rounds}.
You have this idea:

"""
{idea}
"""

The results of the last query are (empty on first round):
"""
{last_query_results}
"""

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the idea and identify any query that could help you make your decision.
If you have made your decision, add "Decision made: novel." or "Decision made: not novel." to your thoughts.

In <JSON>, respond in JSON format with ONLY the following field:
- "Query": An optional search query to search the literature (e.g. attention is all you need). You must make a query if you have not decided this round.

A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.'''

# Function to check the novelty of ideas
def check_idea_novelty(
    ideas,
    base_dir,
    client,
    model,
    max_num_iterations=10,
):
    try:
        with open(osp.join(base_dir, "experiment.py"), "r") as f:
            code = f.read()
        with open(osp.join(base_dir, "prompt.json"), "r") as f:
            prompt = json.load(f)
            task_description = prompt["task_description"]

        for idx, idea in enumerate(ideas):
            if "novel" in idea:
                print(f"Skipping idea {idx}, already checked.")
                continue

            print(f"\nChecking novelty of idea {idx}: {idea['Name']}")

            novel = False
            msg_history = []
            papers_str = ""

            for j in range(max_num_iterations):
                try:
                    text, msg_history = get_response_from_llm(
                        novelty_prompt.format(
                            current_round=j + 1,
                            num_rounds=max_num_iterations,
                            idea=idea,
                            last_query_results=papers_str,
                        ),
                        client=client,
                        model=model,
                        system_message=novelty_system_msg.format(
                            num_rounds=max_num_iterations,
                            task_description=task_description,
                            code=code,
                        ),
                        msg_history=msg_history,
                    )
                    if "decision made: novel" in text.lower():
                        print("Decision made: novel after round", j)
                        novel = True
                        break
                    if "decision made: not novel" in text.lower():
                        print("Decision made: not novel after round", j)
                        break

                    ## PARSE OUTPUT
                    json_output = extract_json_between_markers(text)
                    assert json_output is not None, "Failed to extract JSON from LLM output"

                    ## SEARCH FOR PAPERS
                    query = json_output["Query"]
                    papers = search_for_papers(query, result_limit=10)
                    if papers is None:
                        papers_str = "No papers found."

                    paper_strings = []
                    for i, paper in enumerate(papers):
                        paper_strings.append(
                            """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                                i=i,
                                title=paper["title"],
                                authors=paper["authors"],
                                venue=paper["venue"],
                                year=paper["year"],
                                cites=paper["citationCount"],
                                abstract=paper["abstract"],
                            )
                        )
                    papers_str = "\n\n".join(paper_strings)

                except Exception as e:
                    print(f"{Fore.YELLOW}Error in novelty check iteration: {e}{Style.RESET_ALL}")
                    continue

            idea["novel"] = novel

        # Save results to JSON file
        results_file = osp.join(base_dir, "ideas.json")
        with open(results_file, "w") as f:
            json.dump(ideas, f, indent=4)

        return ideas
    except Exception as e:
        print(f"{Fore.RED}Error in check_idea_novelty: {e}{Style.RESET_ALL}")
    
    return ideas

# Main execution block
if __name__ == "__main__":
    try:
        # Define constants
        MAX_NUM_GENERATIONS = 32
        NUM_REFLECTIONS = 5
        
        # Set up command-line argument parsing
        import argparse

        parser = argparse.ArgumentParser(description="Generate AI scientist ideas")
        parser.add_argument(
            "--experiment",
            type=str,
            default="nanoGPT",
            help="Experiment to run AI Scientist on.",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="gpt-4o-2024-05-13",
            choices=[
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
            ],
            help="Model to use for AI Scientist.",
        )
        parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="URL for OLLAMA API (default: http://localhost:11434)")
        parser.add_argument(
            "--skip-idea-generation",
            action="store_true",
            help="Skip idea generation and use existing ideas.",
        )
        parser.add_argument(
            "--check-novelty",
            action="store_true",
            help="Check novelty of ideas.",
        )
        args = parser.parse_args()

        # Create the appropriate client based on the selected model
        if args.model == "claude-3-5-sonnet-20240620":
            import anthropic
            print(f"Using Anthropic API with model {args.model}.")
            client_model = "claude-3-5-sonnet-20240620"
            client = anthropic.Anthropic()
        elif args.model.startswith("bedrock") and "claude" in args.model:
            import anthropic
            client_model = args.model.split("/")[-1]
            print(f"Using Amazon Bedrock with model {client_model}.")
            client = anthropic.AnthropicBedrock()
        elif args.model.startswith("vertex_ai") and "claude" in args.model:
            import anthropic
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
            import openai
            ollama_model = args.model.split("/")[1]
            print(f"Using OLLAMA API with model {ollama_model} at {args.ollama_url}")
            client_model = ollama_model
            client = openai.OpenAI(
                base_url=f"{args.ollama_url}/v1",
                api_key="ollama",  # OLLAMA doesn't use API keys, but we need to set a non-empty value
            )

            # Test the connection
            try:
                response = requests.get(f"{args.ollama_url}/api/tags")
                response.raise_for_status()
                print("Successfully connected to OLLAMA API")
                
                # Check if the specified model is available
                available_models = response.json()
                if ollama_model not in [model['name'] for model in available_models['models']]:
                    print(f"{Fore.YELLOW}Warning: Model {ollama_model} not found in available OLLAMA models. Please ensure it's installed.{Style.RESET_ALL}")
                else:
                    print(f"Model {ollama_model} is available on the OLLAMA server.")
            except requests.RequestException as e:
                print(f"{Fore.RED}Failed to connect to OLLAMA API: {e}{Style.RESET_ALL}")
                sys.exit(1)
        else:
            raise ValueError(f"Model {args.model} not supported.")

        # Set up directories
        base_dir = osp.join("templates", args.experiment)
        results_dir = osp.join("results", args.experiment)
        
        # Generate ideas
        ideas = generate_ideas(
            base_dir,
            client=client,
            model=client_model,
            skip_generation=args.skip_idea_generation,
            max_num_generations=MAX_NUM_GENERATIONS,
            num_reflections=NUM_REFLECTIONS,
        )
        
        # Check novelty of ideas if requested
        if args.check_novelty:
            ideas = check_idea_novelty(
                ideas,
                base_dir=base_dir,
                client=client,
                model=client_model,
            )
    except Exception as e:
        print(f"{Fore.RED}Critical error in main execution: {e}{Style.RESET_ALL}")
