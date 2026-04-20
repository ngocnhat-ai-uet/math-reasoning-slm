import os
from pickle import FALSE
import sys
import json
import ast
from textwrap import indent
import requests
import argparse
import logging
from dotenv import load_dotenv
import ollama
from datasets import load_dataset
from prompts.agent_prompts.v2 import *
from prompts.direct_prompts.v3 import *

load_dotenv()

# --- CONFIGURATION ---
MODEL_NAME = os.getenv("MODEL_NAME", "phi3")
API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

_ollama_client = None

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

def get_ollama_client():
    """Initializes and returns a singleton Ollama client."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = ollama.Client(host=API_URL)
    return _ollama_client

def _swap_http_scheme(url: str) -> str:
    if url.startswith("https://"):
        return "http://" + url[len("https://"):]
    if url.startswith("http://"):
        return "https://" + url[len("http://"):]
    return url

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
        if isinstance(other_prompts, str):
            messages.append({"role": "user", "content": other_prompts})
        else:
            for prompt in other_prompts:
                messages.append({"role": "user", "content": prompt})
    
    return messages

def send_api_request(messages):
    """
    Sends the request to the Ollama API and returns the response.
    """
    try:
        client = get_ollama_client()
        response = client.chat(
            model=MODEL_NAME,
            messages=messages,
            options={"temperature": 0.1},
        )
        return response
    except Exception as e:
        err = str(e)
        if "SSL" in err or "ssl" in err:
            fallback_url = _swap_http_scheme(API_URL)
            if fallback_url != API_URL:
                try:
                    print(f"SSL error with {API_URL}. Retrying with {fallback_url} ...")
                    fallback_client = ollama.Client(host=fallback_url)
                    response = fallback_client.chat(
                        model=MODEL_NAME,
                        messages=messages,
                        options={"temperature": 0.1},
                    )
                    return response
                except Exception as fallback_e:
                    print(f"Retry with {fallback_url} failed: {fallback_e}")
        print(f"Error during API request: {e}")
        print(f"Current OLLAMA_API_URL={API_URL}")
        sys.exit(1)

def extract_text_from_response(response):
    """
    Extracts the generated text from the API response JSON.
    Handles potential errors if the response format is unexpected.
    """
    try:
        return response["message"]["content"]
    except (KeyError, TypeError) as e:
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

    res = send_api_request(messages)
    out = extract_text_from_response(res) 

    if(verbose):
        print(">>>>>>> Verification results:")
        print(json.dumps(out, indent=4))

    check_correctness = """Response in "yes" or "no". Is the following statement saying the solution is correct, or does not contain critical error or a major justification gap?""" \
            + "\n\n" + out 
    check_messages = build_request_payload(system_prompt="", question_prompt=check_correctness)
    r = send_api_request(check_messages)
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
    r = send_api_request(messages)
    o = extract_text_from_response(r)

    print(o)
    return "yes" in o.lower()

def init_explorations(problem_statement, verbose=True, other_prompts=[], check_complete=True):
    messages = build_request_payload(
            system_prompt=step1_prompt,
            question_prompt=problem_statement,
            #other_prompts=["* Please explore all methods for solving the problem, including casework, induction, contradiction, and analytic geometry, if applicable."]
            #other_prompts = ["You may use analytic geometry to solve the problem."]
            other_prompts = other_prompts
        )

    print(f">>>>>> Initial prompt.")
    print(json.dumps(messages, indent=4))

    response1 = send_api_request(messages)
    output1 = extract_text_from_response(response1)

    print(f">>>>>>> First solution: ") 
    print(json.dumps(output1, indent=4))

    print(f">>>>>>> Self improvement start:")
    improvement_messages = messages.copy()
    improvement_messages.append({"role": "assistant", "content": output1})
    improvement_messages.append({"role": "user", "content": step2_self_improvement_prompt})

    response2 = send_api_request(improvement_messages)
    solution = extract_text_from_response(response2)
    print(f">>>>>>> Corrected solution: ")
    print(json.dumps(solution, indent=4))
    
    if check_complete:
        print(f">>>>>>> Check if solution is complete:"  )
        is_complete = check_if_solution_claimed_complete(solution)
        if not is_complete:
            print(f">>>>>>> Solution is not complete. Failed.")
            return None, None, None, None
    else:
        print(">>>>>>> Skip completeness check (--check_complete=off).")
    
    print(f">>>>>>> Vefify the solution.")
    verify, good_verify = verify_solution(problem_statement, solution, verbose)
    
    return messages, solution, verify, good_verify

def agent(problem_statement, other_prompts=[], max_pass=5, max_fail=10, check_complete=True):
    messages, solution, verify, good_verify = init_explorations(
        problem_statement, True, other_prompts, check_complete
    )

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
            response2 = send_api_request(correction_messages)
            solution = extract_text_from_response(response2)

            print(">>>>>>> Corrected solution:")
            print(json.dumps(solution, indent=4))


            if check_complete:
                print(f">>>>>>> Check if solution is complete:"  )
                is_complete = check_if_solution_claimed_complete(solution)
                if not is_complete:
                    print(f">>>>>>> Solution is not complete. Failed.")
                    return None
            else:
                print(">>>>>>> Skip completeness check (--check_complete=off).")

        print(f">>>>>>> Verify the solution.")
        verify, good_verify = verify_solution(problem_statement, solution)

        if("yes" in good_verify.lower()):
            print(">>>>>>> Solution is good, verifying again to reach max_pass threshold...")
            correct_count += 1
            error_count = 0
 

        if(correct_count >= max_pass):
            print(">>>>>>> Correct solution found.")
            print(json.dumps(solution, indent=4))
            return solution

        elif(error_count >= max_fail):
            print(">>>>>>> Failed in finding a correct solution.")
            return None

    if(not success):
        print(">>>>>>> Failed in finding a correct solution.")
        return None

def load_dataset_from_huggingface(dataset_name, split="train", limit=None, idx=None):
    data = load_dataset(dataset_name, split=split)

    # 'if idx:' would evaluate [] as False and incorrectly return the entire dataset (empty list is expected)  
    if idx is not None:
        return data.filter(lambda x: x["problem_idx"] in idx)
    # 'if limit:' would evaluate 0 as False and incorrectly return the entire dataset (zero samples are expected)
    if limit is not None:
        return data.select(range(limit))
    return data
    # Nếu dùng 'if limit:', khi người dùng muốn lấy 0 mẫu (limit=0), nó sẽ trả về toàn bộ data.

def solve_problem(problem_statement):
    messages = build_request_payload(
        system_prompt=straight_prompt,
        question_prompt=problem_statement
    )

    print(f">>>>>> Initial prompt.")
    print(json.dumps(messages, indent=4))

    response = send_api_request(messages)
    output = extract_text_from_response(response)

    print(">>>>>>> Solution:")
    print(json.dumps(output, indent=4))

    return output

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='IMO Problem Solver Agent')
    parser.add_argument('--mode', choices=('direct', 'agent'), default='direct',
                        help='Run the direct solver or the solver-verifier agent (default: direct)')
    parser.add_argument('--log_dir', type=str, help='Directory for per-problem logs when using a dataset (optional)')
    parser.add_argument('--dataset_name', type=str, help='Hugging Face dataset name (optional)')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of dataset samples to run (optional)')
    parser.add_argument('--idx', type=ast.literal_eval, default=None, help="List/tuple of problem_idx to run, e.g. \"[1, 2, 3]\" or \"(1, 2, 3)\" (optional)")
    parser.add_argument('--other_prompts', '-o', type=str, help='Other prompts (optional)')
    parser.add_argument("--max_runs", '-m', type=int, default=10, help='Maximum number of runs (default: 10)')
    parser.add_argument("--max_pass", type=int, default=5, help='Maximum number of correct verifications before success (default: 5)')
    parser.add_argument("--max_fail", type=int, default=10, help='Maximum number of failed verifications before stopping (default: 10)')
    parser.add_argument(
        "--check_complete",
        choices=("on", "off"),
        default="on",
        help='Turn on/off completeness check in agent mode (default: on)',
    )
    
    args = parser.parse_args()

    max_runs = args.max_runs
    max_pass = args.max_pass
    max_fail = args.max_fail
    check_complete = args.check_complete == "on"
    log_dir = args.log_dir
    mode = args.mode
    
    other_prompts = []
    if args.other_prompts:
        other_prompts = args.other_prompts

    if not log_dir:
        print("Error: --log_dir is required.")
        sys.exit(1)
    os.makedirs(log_dir, exist_ok=True)

    data = load_dataset_from_huggingface(args.dataset_name, limit=args.limit, idx=args.idx)

    for sample in data:
        problem_idx = sample["problem_idx"]
        problem_statement = sample["problem"]
        log_file_path = os.path.join(log_dir, f"{problem_idx}.log")

        # OPEN LOGFILE
        if not set_log_file(log_file_path):
            sys.exit(1)

        for i in range(max_runs):
            print(f"\n\n>>>>>>> Run {i} of {max_runs} ...")
            try:
                if mode == "agent":
                    sol = agent(problem_statement, other_prompts, max_pass, max_fail, check_complete)
                else:
                    sol = solve_problem(problem_statement)
                    
                if(sol is not None):
                    if mode == "agent":
                        print(f">>>>>>> Found a correct solution in run {i}.")
                    else:
                        print(f">>>>>>> Generated solution in run {i}.")
                    print(json.dumps(sol, indent=4))
                    break
            except Exception as e:
                print(f">>>>>>> Error in run {i}: {e}")
                continue

        close_log_file()
