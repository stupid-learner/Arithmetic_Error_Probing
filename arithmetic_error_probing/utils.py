"""
Language model result processing module for addition problems
Used to load, process, and analyze language model responses to addition problems
"""
import random
import re
import csv
import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional


def seed_everything(seed: int) -> None:
    """Set seeds for all random number generators to ensure reproducibility

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_digit(num: int, index: Union[int, str], basis: int = 10) -> int:
    """Get a specific digit from a number

    Args:
        num: Input number
        index: Digit position index to retrieve, or "all" for the entire number
        basis: Number base, default is decimal (10)

    Returns:
        The specified digit or the entire number
    """
    if index == "all":
        return num
    elif index == "start":
        return int(str(abs(num))[0])
    index = int(index)
    return (abs(num) // basis**(index-1)) % basis



def random_select_tuples(tuple_list: List[Tuple], n: int, seed: Optional[int] = 42) -> List[Tuple]:
    """Randomly select n elements from a list

    Args:
        tuple_list: List of tuples
        n: Number of elements to select
        seed: Random seed, None if no seed should be set

    Returns:
        List of n randomly selected elements

    Raises:
        ValueError: If n is greater than the list length
    """
    if n > len(tuple_list):
        raise ValueError("n cannot be greater than the length of the tuple list")
    
    if seed is not None:
        random.seed(seed)
    
    return random.sample(tuple_list, n)


def get_balanced_data(prompts: List[Tuple[int, int]], num_per_digit: int = 100, digit_index: int = 3) -> List[Tuple[int, int]]:
    """Get a dataset balanced by specific digit positions

    Args:
        prompts: List of input prompt tuples
        num_per_digit: Number of samples to select per digit
        digit_index: The digit position index to balance by

    Returns:
        Balanced list of prompts
    """
    balanced_prompts = []
    prompt_by_starting_digit = [[] for _ in range(10)]
    
    # Group prompts by the specified digit
    for prompt in prompts:
        digit = get_digit((prompt[0] + prompt[1]), digit_index)
        prompt_by_starting_digit[digit].append(prompt)
    
    # Randomly select the specified number of samples from each digit group
    for digit in range(10):
        if len(prompt_by_starting_digit[digit]) >= num_per_digit:
            balanced_prompts += random_select_tuples(
                prompt_by_starting_digit[digit], 
                num_per_digit
            )

    return balanced_prompts


def parse_addition_result(text: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Parse addition calculation results from text

    Args:
        text: Text containing addition calculation

    Returns:
        Tuple (operand1, operand2, sum), with None in positions where parsing fails
    """
    # Create set of digit characters for quick checking
    digit_chars = set('0123456789')
    
    # Try to parse the input text
    try:
        # Handle cases with "<|begin_of_text|>" (applicable to llama model)
        if "<|begin_of_text|>" in text:
            content = text[text.find("<|begin_of_text|>")+17:]
            parts = content.split('=')
            
            # Process the result part
            result_str = parts[1].strip()
            if result_str.startswith("+"): 
                result_str = result_str[1:]
            
            # Extract pure numeric part
            result = ""
            for char in result_str:
                if char in digit_chars:
                    result += char
                else:
                    break
            
            # Process the operands part
            operands = parts[0].split('+')
            return int(operands[0].strip()), int(operands[1].strip()), int(result) if result else None
        
        # Handle normal equation format (applicable to gemma model)
        elif "=" in text and "<end_of_turn>" not in text:
            parts = text.split('=')
            
            # Process the result part
            result_str = parts[1].strip()
            
            # Find position of the first digit character
            start_idx = None
            for i, char in enumerate(result_str):
                if char in digit_chars:
                    start_idx = i
                    break
            
            if start_idx is None:
                return None, None, None
            
            # Extract continuous digits
            result_str = result_str[start_idx:]
            result = ""
            for char in result_str:
                if char in digit_chars:
                    result += char
                else:
                    break
            
            # Process the operands part
            operands = parts[0].split('+')
            return int(operands[0].strip()), int(operands[1].strip()), int(result) if result else None
    
    except (ValueError, IndexError):
        pass
    
    return None, None, None


def load_model_result_dic(folder: str, lower_bound: int = 0, sum_upper_bound: int = 1000) -> Dict[Tuple[int, int], int]:
    """Load result dictionary based on model type

    Args:
        folder: Data folder name
        lower_bound: Lower bound for results
        sum_upper_bound: Upper bound for results

    Returns:
        Dictionary in the format {(operand1, operand2): sum}
    """
    # Choose loading function based on folder name
    if "wo_equation" in folder:
        return load_wo_equation_results(folder, lower_bound, sum_upper_bound)
    elif folder.startswith("gemma-2-2b-it"):
        return _load_gemma_it_results(folder, lower_bound, sum_upper_bound)
    elif folder == "Meta-Llama-3-8B_sum_data":
        return _load_llama_results(folder, lower_bound, sum_upper_bound)
    elif folder.startswith(("Meta-Llama-3-8B-Instruct","Llama-3.2-3B-Instruct", "Phi-3-mini-4k-instruct")):
        return _load_llama_instruct_results(folder, lower_bound, sum_upper_bound)
    else:
        return _load_generic_results(folder, lower_bound, sum_upper_bound)


def _load_generic_results(folder: str, lower_bound: int = 0, sum_upper_bound: int = 1000) -> Dict[Tuple[int, int], int]:
    """Load results in generic format

    Args:
        folder: Data folder name
        lower_bound: Lower bound for results
        sum_upper_bound: Upper bound for results

    Returns:
        Dictionary in the format {(operand1, operand2): sum}
    """
    result = []
    
    # Build list of CSV files
    csv_files = [f"data_{i}00_to_{i+1}00" for i in range(1, 10)]
    
    # Read all CSV files
    for csv_file in csv_files:
        try:
            with open(f"{folder}/{csv_file}", 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    result.append(tuple(row))
        except FileNotFoundError:
            print(f"Warning: File {folder}/{csv_file} not found, skipped")
    
    # Filter results, excluding those with parentheses or plus signs, and limiting length
    filtered_result = [
        item for item in result 
        if not "(" in item[2] and not "+" in item[2] and 3 <= len(item[2]) <= 4
    ]
    
    # Create result dictionary, further filtering results exceeding upper bound
    result_dic = {}
    for item in filtered_result:
        try:
            a, b, sum_val = int(item[0]), int(item[1]), int(item[2])
            if a + b < sum_upper_bound and sum_val < sum_upper_bound and a + b >= lower_bound:
                result_dic[(a, b)] = sum_val
        except (ValueError, IndexError):
            continue
    
    return result_dic


def _load_llama_results(folder: str, lower_bound: int = 0, sum_upper_bound: int = 1000) -> Dict[Tuple[int, int], int]:
    """Load Llama model results

    Args:
        folder: Data folder name
        lower_bound: Lower bound for results
        sum_upper_bound: Upper bound for results

    Returns:
        Dictionary in the format {(operand1, operand2): sum}
    """
    result = []
    csv_files = [f"{folder}/data_{i}00_to_{i+1}00" for i in range(1, 10)]
    
    # Read all CSV files
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    for item in row:
                        if "<|begin_of_text|>" in item:
                            operand1, operand2, answer = parse_addition_result(item)
                            if operand1 is not None and operand2 is not None and answer is not None:
                                result.append((str(operand1), str(operand2), str(answer)))
        except FileNotFoundError:
            print(f"Warning: File {csv_file} not found, skipped")
    
    # Filter results and create dictionary
    result_dic = {}
    filtered_result = [item for item in result if 3 <= len(item[2]) <= 4]
    
    for item in filtered_result:
        try:
            a, b, sum_val = int(item[0]), int(item[1]), int(item[2])
            if a + b < sum_upper_bound and sum_val < sum_upper_bound and a + b >= lower_bound:
                result_dic[(a, b)] = sum_val
        except (ValueError, IndexError):
            continue
    
    return result_dic


def _load_gemma_results(folder: str, lower_bound: int = 0, sum_upper_bound: int = 1000) -> Dict[Tuple[int, int], int]:
    """Load Gemma model results

    Args:
        folder: Data folder name
        lower_bound: Lower bound for results
        sum_upper_bound: Upper bound for results

    Returns:
        Dictionary in the format {(operand1, operand2): sum}
    """
    result = []
    csv_files = [f"data_{i}00_to_{i+1}00" for i in range(1, 10)]
    
    # Read all CSV files
    for csv_file in csv_files:
        try:
            with open(f"{folder}/{csv_file}", 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    for content in row:
                        for item in content.split("\n"):
                            operand1, operand2, answer = parse_addition_result(item)
                            if operand1 is not None and operand2 is not None and answer is not None:
                                result.append((str(operand1), str(operand2), str(answer)))
        except FileNotFoundError:
            print(f"Warning: File {folder}/{csv_file} not found, skipped")
    
    # Filter results and create dictionary
    result_dic = {}
    filtered_result = [item for item in result if 3 <= len(item[2]) <= 4]
    
    for item in filtered_result:
        try:
            a, b, sum_val = int(item[0]), int(item[1]), int(item[2])
            if a + b < sum_upper_bound and sum_val < sum_upper_bound and a + b >= lower_bound:
                result_dic[(a, b)] = sum_val
        except (ValueError, IndexError):
            continue
    
    return result_dic

def _load_gemma_it_results(folder: str, lower_bound: int = 0, sum_upper_bound: int = 1000) -> Dict[Tuple[int, int], int]:
    """Load Gemma-IT model results

    Args:
        folder: Data folder name
        lower_bound: Lower bound for results
        sum_upper_bound: Upper bound for results

    Returns:
        Dictionary in the format {(operand1, operand2): sum}
    """
    result_dic = {}
    csv_files = [f"data_{i}00_to_{i+1}00" for i in range(1, 10)]
    
    # Read all CSV files
    for csv_file in csv_files:
        try:
            with open(f"{folder}/{csv_file}", 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    # CSV format: operand1, operand2, response_string
                    if len(row) >= 3:
                        try:
                            operand1 = int(row[0])
                            operand2 = int(row[1])
                            response = row[2]
                            
                            # Extract the result from "<<operand1+operand2=result>> <end_of_turn>"
                            start_idx = response.find("=")
                            end_idx = response.find(">>")
                            
                            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                                result_str = response[start_idx+1:end_idx].strip()
                                
                                # Ensure the result is a valid integer
                                if result_str.isdigit() or (len(result_str) > 1 and result_str[0] == "-" and result_str[1:].isdigit()):
                                    answer = int(result_str)
                                    
                                    # Filter by the bounds
                                    if "sum" in folder and "3to5" in folder:
                                        if (operand1 + operand2 < sum_upper_bound and 
                                            answer < sum_upper_bound and 
                                            operand1 + operand2 >= lower_bound):
                                            result_dic[(operand1, operand2)] = answer

                                    elif "sum" in folder:
                                        if (operand1 + operand2 < sum_upper_bound and 
                                            answer < sum_upper_bound and 
                                            operand1 + operand2 >= lower_bound and
                                            3 <= len(str(answer)) <= 4):
                                            result_dic[(operand1, operand2)] = answer
                                    
                                    elif "difference" in folder:
                                        if (operand1 - operand2 >= lower_bound and
                                            answer < sum_upper_bound
                                            and answer >= lower_bound):
                                            result_dic[(operand1, operand2)] = answer
                        except (ValueError, IndexError):
                            continue
        
        except FileNotFoundError:
            print(f"Warning: File {folder}/{csv_file} not found, skipped")
    
    return result_dic

def load_wo_equation_results(folder: str, lower_bound: int = 0, sum_upper_bound: int = 1000) -> Dict[Tuple[int, int], int]:
    result_dic = {}
    csv_files = [f"data_{i}00_to_{i+1}00" for i in range(1, 10)]
    
    # Read all CSV files
    for csv_file in csv_files:
        try:
            with open(f"{folder}/{csv_file}", 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    # CSV format: operand1, operand2, response_string
                    if len(row) >= 3:
                        try:
                            operand1 = int(row[0])
                            operand2 = int(row[1])
                            answer = int(row[2].split("\n")[0])

                                    
                            # Filter by the bounds
                            if "sum" in folder and "3to5" in folder:
                                if (operand1 + operand2 < sum_upper_bound and 
                                    answer < sum_upper_bound and 
                                    operand1 + operand2 >= lower_bound):
                                    result_dic[(operand1, operand2)] = answer

                            elif "sum" in folder:
                                if (operand1 + operand2 < sum_upper_bound and 
                                    answer < sum_upper_bound and 
                                    operand1 + operand2 >= lower_bound and
                                    3 <= len(str(answer)) <= 4):
                                    result_dic[(operand1, operand2)] = answer
                                    
                            elif "difference" in folder:
                                if (operand1 - operand2 >= lower_bound and
                                    answer < sum_upper_bound
                                    and answer >= lower_bound):
                                    result_dic[(operand1, operand2)] = answer
                        except (ValueError, IndexError):
                            continue
        
        except FileNotFoundError:
            print(f"Warning: File {folder}/{csv_file} not found, skipped")
    
    return result_dic



def _load_llama_instruct_results(folder: str, lower_bound: int = 0, sum_upper_bound: int = 1000) -> Dict[Tuple[int, int], int]:
    """Load Llama Instruct model results

    Args:
        folder: Data folder name
        lower_bound: Lower bound for results
        sum_upper_bound: Upper bound for results

    Returns:
        Dictionary in the format {(operand1, operand2): sum}
    """
    result_dic = {}
    csv_files = [f"data_{i}00_to_{i+1}00" for i in range(1, 10)]
    
    # Read all CSV files
    for csv_file in csv_files:
        try:
            with open(f"{folder}/{csv_file}", 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    # CSV format: operand1, operand2, <<operand1+operand2=result>><|eot_id|>
                    if len(row) >= 3:
                        operand1 = int(row[0])
                        operand2 = int(row[1])
                        response = row[2]
                            
                        # Extract the result from "<<operand1+operand2=result>><|eot_id|>"
                        start_idx = response.find("=")
                        end_idx = response.find(">>")
                            
                        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                            result_str = response[start_idx+1:end_idx].strip()
                                
                            # Ensure the result is a valid integer
                            if result_str.isdigit():
                                answer = int(result_str)
                                    
                                # Filter by the bounds
                                if (operand1 + operand2 < sum_upper_bound and 
                                    answer < sum_upper_bound and 
                                    operand1 + operand2 >= lower_bound and
                                    3 <= len(str(answer)) <= 4):  # Apply the same filter as in other functions
                                    result_dic[(operand1, operand2)] = answer

        except FileNotFoundError:
            print(f"Warning: File {folder}/{csv_file} not found, skipped")
    
    return result_dic

def all_equations_correct(text):
    pattern = r"<<(.*?)>>"
    matches = re.findall(pattern, text)
    first_eq_index = -1
    first_left = -1
    first_right = -1
    
    for equation in matches:
        if '=' in equation and not '-' in equation and not '*' in equation and not '/' in equation:
            # Find the position of = in the current equation
            eq_pos_in_match = equation.find('=')
            
            # Find the starting position of this match in the original text
            match_start = text.find('<<' + equation + '>>')
            if match_start != -1:
                # Calculate the absolute position of = in the original text
                eq_pos_in_text = match_start + 2 + eq_pos_in_match  # +2 to skip <<
                
                # Record the position of = in the first equation
                if first_eq_index == -1:
                    first_eq_index = eq_pos_in_text
            
            left, right = equation.split('=', 1)
            if first_left == -1:
                first_left, first_right = left, right
            try:
                if len(right) == 0: 
                    continue  # Empty right side, continue checking the next equation
                left_val = eval(left.strip(), {"__builtins__": None}, {})
                right_val = eval(right.strip(), {"__builtins__": None}, {})
                if left_val != right_val:
                    return False, eq_pos_in_text, left, right
            except:
                pass
    
    return True, first_eq_index, first_left, first_right