# models.py
from pydantic import BaseModel
from typing import Optional, Dict, Any, Union, BinaryIO, List
from .validation import matches_schema
from enum import Enum

class ZenbaseConfig(BaseModel):
    api_key: str
    base_url: str = "https://orch.zenbase.ai/api"  # Example base URL
    timeout: Optional[int] = 30

class ZenbaseFunctionConfig(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    input_schema: Optional[dict] = None
    output_schema: Optional[dict] = None

class ZenbaseFunctionInput(BaseModel):
    inputs: dict
    object_id: Optional[int] = None

class ZenbaseFunctionOutput(BaseModel):
    outputs: dict | None
    object_id: Optional[int] = None

class BatchFunctionRunStatusEnum(Enum):
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"
    RUNNING = "RUNNING"

class BatchFunctionRunStatus(BaseModel):
    status: BatchFunctionRunStatusEnum
    total_runs: int
    completed_runs: int
    failed_runs: int

class BatchFunctionRunResults(BaseModel):
    results: List[ZenbaseFunctionOutput]
    failed_object_ids: List[int]

class BatchFunctionInputList(BaseModel):
    items: List[ZenbaseFunctionInput]
    
    def __init__(self, items: List[ZenbaseFunctionInput] = None):
        super().__init__(items=items or [])

    # Inefficient, but we don't expect to use this often
    def get_subset_by_object_ids(self, object_ids: List[int]) -> "BatchFunctionInputList":
        # Create lookup dictionary - only items with object_ids
        lookup = {item.object_id: item for item in self.items if item.object_id is not None}
        
        # Return matching inputs, preserving order of requested object_ids
        return BatchFunctionInputList(items=[lookup.get(object_id) for object_id in object_ids])
    
    # Will raise an error if the inputs are not valid
    def check_valid(self, input_schema: dict) -> bool:
        object_ids = []
        for input_data in self.items:
            if input_data.object_id in object_ids:
                raise ValueError(f"Object ID {input_data.object_id} already exists in the list")
            else:
                object_ids.append(input_data.object_id)

            if not matches_schema(input_schema, input_data.inputs):
                raise ValueError(f"Input data for object ID {input_data.object_id} does not match the schema")
            
        return True
    
    def to_dict_list(self) -> dict:
        return [item.model_dump() for item in self.items]
    
    def to_list(self) -> List[ZenbaseFunctionInput]:
        return self.items
    
    def append(self, item: ZenbaseFunctionInput):
        self.items.append(item)


