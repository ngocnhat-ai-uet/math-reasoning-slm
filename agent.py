import os
from pickle import FALSE
import sys
import json
from textwrap import indent
import requests
import argparse
import logging
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from datasets import load_dataset
from prompts import *

load_dotenv()

# --- CONFIGURATION ---
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5-nano")
API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/")

# Global variables for logging
_log_file = None
original_print = print

def log_print(*args, **kwargs):
    """
    Custom print function that writes to both stdout and log file.
    """
    # Print to stdout
    original_print(*args, **kwargs)
    
    # Also write to log file if specified
    if _log_file is not None:
        # Convert all arguments to strings and join them
        message = ' '.join(str(arg) for arg in args)
        _log_file.write(message + '\n')
        _log_file.flush()  # Ensure immediate writing

# Replace the built-in print function
print = log_print

def set_log_file(log_file_path):
    """Set the log file for output."""
    global _log_file
    if log_file_path:
        try:
            _log_file = open(log_file_path, 'w', encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error opening log file {log_file_path}: {e}")
            return False
    return True

def close_log_file():
    """Close the log file if it's open."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None

def get_api_key():
    """
    Retrieves the API key from environment variables.
    Exits if the key is not found.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set the variable in your .env file or environment")
        sys.exit(1)
    return api_key

def read_file_content(filepath):
    """
    Reads and returns the content of a file.
    Exits if the file cannot be read.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

def build_request_payload(system_prompt, question_prompt, other_prompts=None):
    """
    Builds the JSON payload for the Gemini API request, using the
    recommended multi-turn format to include a system prompt.
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question_prompt})
    
    if other_prompts:
        for prompt in other_prompts:
            messages.append({"role": "user", "content": prompt})
    
    return messages

def send_api_request(api_key, messages):
    """
    Sends the request to the OPENAI Compatible API and returns the response.
    """
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=API_URL
        )
        
        extra_args = {}
        if "gemini" in MODEL_NAME.lower():
            extra_args = {
                "extra_body": {
                    "google": {
                        "thinking_config": {
                            "thinking_budget": 32768,
                            "include_thoughts": False
                        }
                    }
                }
            }
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            #temperature=0.1,
            top_p=1.0,
            extra_body=extra_args
        )
        
        return response
    except Exception as e:
        print(f"Error during API request: {e}")
        sys.exit(1)

def extract_text_from_response(response: ChatCompletion):
    """
    Extracts the generated text from the API response JSON.
    Handles potential errors if the response format is unexpected.
    """
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        print("Error: Could not extract text from the API response.")
        print(f"Reason: {e}")
        print("Full API Response:")
        print(response)
        raise e

def extract_detailed_solution(solution: str, marker: str = 'Detailed Solution', after: bool = True) -> str:
    """
    Extracts the text after '### Detailed Solution ###' from the solution string.
    Returns the substring after the marker, stripped of leading/trailing whitespace.
    If the marker is not found, returns an empty string.
    """
    idx = solution.find(marker)
    if idx == -1:
        return ''
    if(after):
        return solution[idx + len(marker):].strip()
    else:
        return solution[:idx].strip()

def verify_solution(problem_statement, solution, verbose=True):

    dsol = extract_detailed_solution(solution)

    newst = f"""
======================================================================
### Problem ###

{problem_statement}

======================================================================
### Solution ###

{dsol}

{step6_verification_remider}
"""
    if(verbose):
        print(">>>>>>> Start verification.")
    messages = build_request_payload(
        system_prompt=step3_verification_system_prompt, 
        question_prompt=newst
    )
    
    if(verbose):
        print(">>>>>>> Verification prompt:")
        print(json.dumps(messages, indent=4))

    res = send_api_request(get_api_key(), messages)
    out = extract_text_from_response(res) 

    if(verbose):
        print(">>>>>>> Verification results:")
        print(json.dumps(out, indent=4))

    check_correctness = """Response in "yes" or "no". Is the following statement saying the solution is correct, or does not contain critical error or a major justification gap?""" \
            + "\n\n" + out 
    check_messages = build_request_payload(system_prompt="", question_prompt=check_correctness)
    r = send_api_request(get_api_key(), check_messages)
    o = extract_text_from_response(r) 

    if(verbose):
        print(">>>>>>> Is verification good?")
        print(json.dumps(o, indent=4))
        
    bug_report = ""

    if("yes" not in o.lower()):
        bug_report = extract_detailed_solution(out, "Detailed Verification", False)

    if(verbose):
        print(">>>>>>>Bug report:")
        print(json.dumps(bug_report, indent=4))
    
    return bug_report, o

def check_if_solution_claimed_complete(solution):
    check_complete_prompt = f"""
Is the following text claiming that the solution is complete?
==========================================================

{solution}

==========================================================

Response in exactly "yes" or "no". No other words.
    """

    messages = build_request_payload(system_prompt="", question_prompt=check_complete_prompt)
    r = send_api_request(get_api_key(), messages)
    o = extract_text_from_response(r)

    print(o)
    return "yes" in o.lower()

def init_explorations(problem_statement, verbose=True, other_prompts=[]):
    messages = build_request_payload(
            system_prompt=step1_prompt,
            question_prompt=problem_statement,
            #other_prompts=["* Please explore all methods for solving the problem, including casework, induction, contradiction, and analytic geometry, if applicable."]
            #other_prompts = ["You may use analytic geometry to solve the problem."]
            other_prompts = other_prompts
        )

    print(f">>>>>> Initial prompt.")
    print(json.dumps(messages, indent=4))

    response1 = send_api_request(get_api_key(), messages)
    output1 = extract_text_from_response(response1)

    print(f">>>>>>> First solution: ") 
    print(json.dumps(output1, indent=4))

    print(f">>>>>>> Self improvement start:")
    improvement_messages = messages.copy()
    improvement_messages.append({"role": "assistant", "content": output1})
    improvement_messages.append({"role": "user", "content": step2_self_improvement_prompt})

    response2 = send_api_request(get_api_key(), improvement_messages)
    solution = extract_text_from_response(response2)
    print(f">>>>>>> Corrected solution: ")
    print(json.dumps(solution, indent=4))
    
    print(f">>>>>>> Check if solution is complete:"  )
    is_complete = check_if_solution_claimed_complete(output1)
    if not is_complete:
        print(f">>>>>>> Solution is not complete. Failed.")
        return None, None, None, None
    
    print(f">>>>>>> Vefify the solution.")
    verify, good_verify = verify_solution(problem_statement, solution, verbose)

    print(f">>>>>>> Initial verification: ")
    print(json.dumps(verify, indent=4))
    print(f">>>>>>> verify results: {good_verify}")
    
    return messages, solution, verify, good_verify

def agent(problem_statement, other_prompts=[]):
    messages, solution, verify, good_verify = init_explorations(problem_statement, True, other_prompts)

    if(solution is None):
        print(">>>>>>> Failed in finding a complete solution.")
        return None

    error_count = 0
    correct_count = 1
    success = False
    for i in range(30):
        print(f"Number of iterations: {i}, number of corrects: {correct_count}, number of errors: {error_count}")

        if("yes" not in good_verify.lower()):
            # clear
            correct_count = 0
            error_count += 1

            #self improvement
            print(">>>>>>> Verification does not pass, correcting ...")
            # establish a new prompt that contains the solution and the verification

            correction_messages = build_request_payload(
                system_prompt=step1_prompt,
                question_prompt=problem_statement,
                #other_prompts=["You may use analytic geometry to solve the problem."]
                other_prompts=other_prompts
            )
            
            correction_messages.append({"role": "assistant", "content": solution})
            
            correction_messages.append({"role": "user", "content": step5_correction_prompt + "\n\n" + verify})

            print(">>>>>>> New prompt:")
            print(json.dumps(correction_messages, indent=4))
            response2 = send_api_request(get_api_key(), correction_messages)
            solution = extract_text_from_response(response2)

            print(">>>>>>> Corrected solution:")
            print(json.dumps(solution, indent=4))


            print(f">>>>>>> Check if solution is complete:"  )
            is_complete = check_if_solution_claimed_complete(solution)
            if not is_complete:
                print(f">>>>>>> Solution is not complete. Failed.")
                return None

        print(f">>>>>>> Verify the solution.")
        verify, good_verify = verify_solution(problem_statement, solution)

        if("yes" in good_verify.lower()):
            print(">>>>>>> Solution is good, verifying again ...")
            correct_count += 1
            error_count = 0
 

        if(correct_count >= 5):
            print(">>>>>>> Correct solution found.")
            print(json.dumps(solution, indent=4))
            return solution

        elif(error_count >= 10):
            print(">>>>>>> Failed in finding a correct solution.")
            return None

    if(not success):
        print(">>>>>>> Failed in finding a correct solution.")
        return None
        
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='IMO Problem Solver Agent')
    parser.add_argument('problem_file', nargs='?', default='problems/imo01.txt', 
                       help='Path to the problem statement file (default: problem_statement.txt)')
    parser.add_argument('--log', '-l', type=str, help='Path to log file (optional)')
    parser.add_argument('--other_prompts', '-o', type=str, help='Other prompts (optional)')
    parser.add_argument("--max_runs", '-m', type=int, default=10, help='Maximum number of runs (default: 10)')
    
    args = parser.parse_args()

    max_runs = args.max_runs
    
    other_prompts = []
    if args.other_prompts:
        other_prompts = args.other_prompts.split(',')

    print(">>>>>>> Other prompts:")
    print(other_prompts)

    # Set up logging if log file is specified
    if args.log:
        if not set_log_file(args.log):
            sys.exit(1)
        print(f"Logging to file: {args.log}")
    
    problem_statement = read_file_content(args.problem_file)

    for i in range(max_runs):
        print(f"\n\n>>>>>>>>>>>>>>>>>>>>>>>>>> Run {i} of {max_runs} ...")
        try:
            sol = agent(problem_statement, other_prompts)
            if(sol is not None):
                print(f">>>>>>> Found a correct solution in run {i}.")
                print(json.dumps(sol, indent=4))
                break
        except Exception as e:
            print(f">>>>>>> Error in run {i}: {e}")
            continue
    
    # Close log file if it was opened
    close_log_file()
