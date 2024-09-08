"""
This module provides functions for interacting with various Large Language Models (LLMs) through API calls.
It supports multiple models including GPT-4, Claude, Deepseek Coder, and Llama.

Key functionalities:
1. get_batch_responses_from_llm: Retrieves multiple responses from an LLM for a single input message.
2. get_response_from_llm: Retrieves a single response from an LLM for a given input message.
3. extract_json_between_markers: Extracts and parses JSON data from LLM outputs.

The module uses the following external libraries:
- backoff: For handling API rate limits and timeouts with exponential backoff.
- openai: For interacting with OpenAI's API.
- json: For parsing JSON data.
- colorama: For colored console output.

Input sources:
- The functions in this module receive inputs directly from the calling code, not from separate files.

Output destinations:
- The functions return their results to the calling code.
- Debug information and error messages are printed to the console.

This module is typically used by other parts of the AI scientist project to interact with LLMs.
It does not read from or write to any specific files directly.

File path: ai_scientist/llm.py
"""

# Import necessary libraries
import backoff  # For handling retries and exponential backoff
import openai  # OpenAI API client
import json  # For JSON parsing
from colorama import Fore, Style, init  # For colored console output

# Initialize colorama to enable colored output in the console
init(autoreset=True)

# This function gets multiple responses from an AI model for a single message
# It's useful for getting diverse answers or for ensembling (combining multiple responses)
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_batch_responses_from_llm(
    msg,  # The message to send to the AI
    client,  # The API client (e.g., OpenAI client)
    model,  # The name of the AI model to use
    system_message,  # Instructions for the AI model
    print_debug=False,  # Whether to print debug information
    msg_history=None,  # Previous conversation history
    temperature=0.75,  # Controls randomness in AI responses (0.0 to 1.0)
    n_responses=1,  # Number of responses to generate
):
    # Initialize an empty message history if none is provided
    if msg_history is None:
        msg_history = []

    try:
        # Handle different AI models with specific API calls
        if model in [
            "gpt-4o-2024-05-13",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-08-06",
        ]:
            # Add the new user message to the history
            new_msg_history = msg_history + [{"role": "user", "content": msg}]
            # Make an API call to get responses
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=3000,
                n=n_responses,
                stop=None,
                seed=0,
            )
            # Extract the content from each response
            content = [r.message.content for r in response.choices]
            # Create new message histories for each response
            new_msg_history = [
                new_msg_history + [{"role": "assistant", "content": c}] for c in content
            ]
        
        # Similar process for other models (deepseek-coder, llama)
        elif model == "deepseek-coder-v2-0724":
            # Add the new user message to the history
            new_msg_history = msg_history + [{"role": "user", "content": msg}]
            # Make an API call to get responses
            response = client.chat.completions.create(
                model="deepseek-coder",
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=3000,
                n=n_responses,
                stop=None,
            )
            # Extract the content from each response
            content = [r.message.content for r in response.choices]
            # Create new message histories for each response
            new_msg_history = [
                new_msg_history + [{"role": "assistant", "content": c}] for c in content
            ]
        
        elif model == "llama-3-1-405b-instruct":
            # Add the new user message to the history
            new_msg_history = msg_history + [{"role": "user", "content": msg}]
            # Make an API call to get responses
            response = client.chat.completions.create(
                model="meta-llama/llama-3.1-405b-instruct",
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=3000,
                n=n_responses,
                stop=None,
            )
            # Extract the content from each response
            content = [r.message.content for r in response.choices]
            # Create new message histories for each response
            new_msg_history = [
                new_msg_history + [{"role": "assistant", "content": c}] for c in content
            ]
        
        elif model.startswith("ollama/"):
            # Add the new user message to the history
            new_msg_history = msg_history + [{"role": "user", "content": msg}]
            # Make an API call to get responses
            response = client.chat.completions.create(
                model=model.split("/")[1],  # Extract the actual model name
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=3000,
                n=n_responses,
                stop=None,
            )
            # Extract the content from each response
            content = [r.message.content for r in response.choices]
            # Create new message histories for each response
            new_msg_history = [
                new_msg_history + [{"role": "assistant", "content": c}] for c in content
            ]
        
        # Special handling for Claude models
        elif "claude" in model:
            content, new_msg_history = [], []
            # Get individual responses for Claude models
            for _ in range(n_responses):
                c, hist = get_response_from_llm(
                    msg,
                    client,
                    model,
                    system_message,
                    print_debug=False,
                    msg_history=None,
                    temperature=temperature,
                )
                content.append(c)
                new_msg_history.append(hist)
        
        else:
            # Raise an error if the model is not supported
            raise ValueError(f"Model {model} not supported.")

        # Print debug information if requested
        if print_debug:
            # Just print the first one.
            print()
            print("*" * 20 + " LLM START " + "*" * 20)
            for j, msg in enumerate(new_msg_history[0]):
                print(f'{j}, {msg["role"]}: {msg["content"]}')
            print(content)
            print("*" * 21 + " LLM END " + "*" * 21)
            print()

        # Return the generated content and updated message history
        return content, new_msg_history

    except Exception as e:
        # Print any errors that occur during the process
        print(f"{Fore.RED}Error in get_batch_responses_from_llm: {str(e)}{Style.RESET_ALL}")
        return [], []  # Return empty lists if an error occurs

# This function gets a single response from an AI model
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_response_from_llm(
    msg,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.75,
):
    # Initialize an empty message history if none is provided
    if msg_history is None:
        msg_history = []

    try:
        # Handle different AI models with specific API calls
        if "claude" in model:
            # Special handling for Claude models
            new_msg_history = msg_history + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": msg,
                        }
                    ],
                }
            ]
            response = client.messages.create(
                model=model,
                max_tokens=3000,
                temperature=temperature,
                system=system_message,
                messages=new_msg_history,
            )
            content = response.content[0].text
            new_msg_history = new_msg_history + [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                        }
                    ],
                }
            ]
        
        elif model in [
            "gpt-4o-2024-05-13",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-08-06",
        ]:
            # Handling for GPT-4 models
            new_msg_history = msg_history + [{"role": "user", "content": msg}]
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=3000,
                n=1,
                stop=None,
                seed=0,
            )
            content = response.choices[0].message.content
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        
        elif model == "deepseek-coder-v2-0724":
            # Handling for Deepseek Coder model
            new_msg_history = msg_history + [{"role": "user", "content": msg}]
            response = client.chat.completions.create(
                model="deepseek-coder",
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=3000,
                n=1,
                stop=None,
            )
            content = response.choices[0].message.content
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        
        elif model in ["meta-llama/llama-3.1-405b-instruct", "llama-3-1-405b-instruct"]:
            # Handling for Llama models
            new_msg_history = msg_history + [{"role": "user", "content": msg}]
            response = client.chat.completions.create(
                model="meta-llama/llama-3.1-405b-instruct",
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=3000,
                n=1,
                stop=None,
            )
            content = response.choices[0].message.content
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        
        elif model.startswith("ollama/"):
            new_msg_history = msg_history + [{"role": "user", "content": msg}]
            response = client.chat.completions.create(
                model=model.split("/")[1],  # Extract the actual model name
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=3000,
                n=1,
                stop=None,
            )
            content = response.choices[0].message.content
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        
        else:
            # Raise an error if the model is not supported
            raise ValueError(f"Model {model} not supported.")

        # Print debug information if requested
        if print_debug:
            print()
            print("*" * 20 + " LLM START " + "*" * 20)
            for j, msg in enumerate(new_msg_history):
                print(f'{j}, {msg["role"]}: {msg["content"]}')
            print(content)
            print("*" * 21 + " LLM END " + "*" * 21)
            print()

        # Return the generated content and updated message history
        return content, new_msg_history

    except Exception as e:
        # Print any errors that occur during the process
        print(f"{Fore.RED}Error in get_response_from_llm: {str(e)}{Style.RESET_ALL}")
        return "", []  # Return empty strings if an error occurs

# This function extracts JSON data from the AI model's output
def extract_json_between_markers(llm_output):
    # Define markers that surround the JSON data in the output
    json_start_marker = "```json"
    json_end_marker = "```"

    try:
        # Find the start index of the JSON data
        start_index = llm_output.find(json_start_marker)
        if start_index != -1:
            # Move past the start marker
            start_index += len(json_start_marker)
            # Find the end index of the JSON data
            end_index = llm_output.find(json_end_marker, start_index)
        else:
            # Print a warning if the start marker is not found
            print(f"{Fore.YELLOW}Warning: JSON start marker not found{Style.RESET_ALL}")
            return None

        if end_index == -1:
            # Print a warning if the end marker is not found
            print(f"{Fore.YELLOW}Warning: JSON end marker not found{Style.RESET_ALL}")
            return None

        # Extract the JSON string from the output
        json_string = llm_output[start_index:end_index].strip()
        # Parse the JSON string into a Python object
        parsed_json = json.loads(json_string)
        return parsed_json
    
    except json.JSONDecodeError as e:
        # Handle JSON parsing errors
        print(f"{Fore.RED}Error decoding JSON: {str(e)}{Style.RESET_ALL}")
        return None
    
    except Exception as e:
        # Handle any other unexpected errors
        print(f"{Fore.RED}Unexpected error in extract_json_between_markers: {str(e)}{Style.RESET_ALL}")
        return None
