# helpers.py
import jsonschema
from io import BytesIO
from typing import Optional, Dict, Any, Union, BinaryIO, List
import json
from string import Formatter
from copy import deepcopy
from .models import ZenbaseFunctionInput, ZenbaseFunctionOutput, BatchFunctionRunResults, BatchFunctionInputList
    
def make_batch_input_file(inputs_list: Any) -> BytesIO:
    json_data = json.dumps(inputs_list).encode('utf-8')
    file_obj = BytesIO(json_data)
    files = {'file': ('batch_input.json', file_obj, 'application/json')}
    return files

def get_top_level_schema_fields(schema):
    """
    Extract top-level field names from a JSON Schema.
    
    Args:
        schema (dict): The JSON Schema to analyze
    
    Returns:
        set: Set of top-level field names found in the schema
    """
    if not isinstance(schema, dict):
        return set()
        
    # Get properties directly defined at the top level
    properties = schema.get("properties", {})
    return set(properties.keys())

def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))

def get_format_fields(template):
    """
    Extract field names from a format string template.
    
    Args:
        template (str): A string with format placeholders like "{name} {age}"
    
    Returns:
        list: List of field names found in the template
    """
    fields = [fname for _, fname, _, _ in Formatter().parse(template) if fname is not None]
    return fields

def get_batch_optimizer_run_results_per_page(batch_run_results: List[dict]) -> BatchFunctionRunResults:
    results = []
    failed_object_ids = []
    for result in batch_run_results:
        object_id = result['object_id']
        outputs = result['outputs']
        if outputs == None:
            failed_object_ids.append(object_id)
        else:
            results.append(ZenbaseFunctionOutput(object_id=object_id, outputs=outputs['output']))
    return BatchFunctionRunResults(results=results, failed_object_ids=failed_object_ids)

def convert_to_openai_response_format(schema: dict) -> dict:
    """
    Convert a JSON schema to OpenAI response format.

    Args:
        schema (dict): The input JSON schema.

    Returns:
        dict: The OpenAI-compatible response format.
    """
    def add_additional_properties_false(schema_part: dict):
        """
        Recursively add additionalProperties=False to all objects in the schema.
        """
        if schema_part.get("type") == "object":
            schema_part["additionalProperties"] = False
            for key, value in schema_part.get("properties", {}).items():
                add_additional_properties_false(value)
        elif schema_part.get("type") == "array":
            add_additional_properties_false(schema_part.get("items", {}))

    schema = deepcopy(schema)
    add_additional_properties_false(schema)

    openai_response = {
        "type": "json_schema",
        "json_schema": {
                    "name": schema['title'] + "Schema",
                    "schema": schema,
                "strict": True
        }
    }
    return openai_response

# def get_inputs_by_object_ids(input_list: List[ZenbaseFunctionInput], object_ids: List[int]) -> List[ZenbaseFunctionInput]:
#     # Create lookup dictionary - only items with object_ids
#     lookup = {item.object_id: item for item in input_list if item.object_id is not None}
    
#     # Return matching inputs, preserving order of requested object_ids
#     return [lookup.get(object_id) for object_id in object_ids]