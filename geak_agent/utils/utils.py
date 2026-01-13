# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import re
import json
import ast

def extract_function_signatures(code):
    function_defs = []
    pattern = r'def\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)'
    matches = re.finditer(pattern, code)
    
    for match in matches:
        func_name = match.group(1)
        params = match.group(2)
        function_defs.append(f"def {func_name}({params})")
    
    return function_defs

def clear_code(code):
    if  "```python" in code:
        code = code.split("```python")[-1].replace("<|im_end|>", "").replace("<|EOT|>", "")
    if "```" in code:
        code = code.split("```")[0]
    return code

def extract_function_calls(code):
    calls = []
    pattern = r'([a-zA-Z0-9_]+)\s*\(([^)]*)\)'
    matches = re.finditer(pattern, code)
    
    for match in matches:
        func_name = match.group(1)
        args = match.group(2)
        calls.append(f"{func_name}({args})")
    
    return calls

def clear_json(response):
    if type(response) is dict:
        return response
    elif type(response) is not str:
        response = str(response)
    
    # Try to extract JSON from markdown code blocks first
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Fall back to finding raw JSON object
        json_match = re.search(r'(\{[\s\S]*\})', response)
        if json_match:
            json_str = json_match.group(1)
        else:
            return "ERR_SYNTAX"
    
    # Try json.loads first (handles null, true, false)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Fall back to ast.literal_eval with preprocessing
    try:
        json_str_clean = json_str.replace("\n", " ")
        json_str_clean = re.sub(r"(\w)'(\w|\s)", r"\1\\'\2", json_str_clean)
        # Convert JSON keywords to Python
        json_str_clean = json_str_clean.replace('null', 'None')
        json_str_clean = json_str_clean.replace('true', 'True')
        json_str_clean = json_str_clean.replace('false', 'False')
        result = ast.literal_eval(json_str_clean)
        return result
    except (SyntaxError, NameError, AttributeError, ValueError):
        return "ERR_SYNTAX"