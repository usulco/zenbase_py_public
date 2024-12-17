# zenbase_client.py
import os
import requests
from typing import Optional, Dict, Any, Union, BinaryIO, List
import time

from .models import ZenbaseConfig, ZenbaseFunctionConfig, BatchFunctionInputList, BatchFunctionRunStatus, BatchFunctionRunStatusEnum, BatchFunctionRunResults
from .helpers import make_batch_input_file, clamp, get_batch_optimizer_run_results_per_page

class ZenbaseClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        # Initialize with environment variable if api_key not provided
        self.api_key = api_key or os.getenv('ZENBASE_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided or set in ZENBASE_API_KEY environment variable")

        # Create config with defaults
        self.config = ZenbaseConfig(
            api_key=self.api_key,
            base_url=base_url or "https://orch.zenbase.ai/api",
            timeout=timeout
        )
        
        # Initialize session with default headers
        self.session = requests.Session()
        self.optimizer_function_id_cache: Dict[int, int] = {}
        self.batch_run_id_to_function_id_cache: Dict[Union[int, str], int] = {}
        self.function_config_cache: Dict[int, ZenbaseFunctionConfig] = {}

    def _make_request(
            self,
            method: str,
            endpoint: str,
            params: Optional[Dict[str, Any]] = None,
            data: Optional[Dict[str, Any]] = None,
            files: Optional[Dict[str, Union[BinaryIO, tuple]]] = None,
            **kwargs
        ) -> requests.Response:
            """
            Make HTTP request to Zenbase API

            Args:
                method: HTTP method (GET, POST, etc.)
                endpoint: API endpoint path
                params: URL query parameters
                data: Request body data (will be sent as JSON if files=None)
                files: Dict of files to upload. Can be file objects or tuples of (filename, file object)
                **kwargs: Additional arguments to pass to requests.request()

            Returns:
                requests.Response object

            Raises:
                ZenbaseAPIError: If the request fails
            """
            url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
            
            try:
                headers = {}
                headers['Authorization'] = f'Api-Key {self.config.api_key}'
                # headers['Accept'] = 'application/json'
                if not files:
                    headers['Content-Type'] = 'application/json'
                if files:
                    # For multipart/form-data, send data as form fields
                    form_data = data if data else {}
                    response = self.session.request(
                        method=method,
                        url=url,
                        params=params,
                        data=form_data,
                        files=files,
                        headers=headers,
                        timeout=self.config.timeout,
                        **kwargs
                    )
                else:
                    # For regular requests, send data as JSON
                    response = self.session.request(
                        method=method,
                        url=url,
                        params=params,
                        json=data,
                        headers=headers,
                        timeout=self.config.timeout,
                        **kwargs
                    )
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                # Handle API errors appropriately
                raise ZenbaseAPIError(f"API request failed: {str(e)}") from e
            
    def get_optimizer_function_id(self, optimizer_id: int, update_cache: bool = True) -> int:
        # Retrive from cache if available
        if optimizer_id in self.optimizer_function_id_cache:
            return self.optimizer_function_id_cache[optimizer_id]
        response = self._make_request('GET', f'optimizer-configurations/{optimizer_id}').json()
        if 'id' not in response:
            raise ZenbaseAPIError(response['detail'])
        if update_cache:
            self.optimizer_function_id_cache[optimizer_id] = response['function']
        return response['function']

    def get_function_config(self, function_id: int, update_cache: bool = True) -> ZenbaseFunctionConfig:
        # Retrive from cache if available
        if function_id in self.function_config_cache:
            return self.function_config_cache[function_id]
        response = self._make_request('GET', f'functions/{function_id}').json()
        if 'id' not in response:
            raise ZenbaseAPIError(response['detail'])
        function_config = ZenbaseFunctionConfig(
            name=response['name'],
            description=response['description'],
            prompt=response['prompt'],
            input_schema=response['input_schema'],
            output_schema=response['output_schema'],
            model=response['model']
        )
        if update_cache:
            self.function_config_cache[function_id] = function_config
        return function_config


    def start_batch_optimizer_run(self, optimizer_id: int, inputs_list: BatchFunctionInputList) -> int:
        """
        Start a batch run of the optimizer function with the given inputs.

        Args:
            optimizer_id (int): The ID of the optimizer configuration to use
            inputs_list (BatchFunctionInputList): List of inputs to process in batch

        Returns:
            int: The ID of the created batch run

        Raises:
            ZenbaseAPIError: If the API request fails or returns invalid response
        """
        function_id = self.get_optimizer_function_id(optimizer_id)
        input_schema = self.get_function_config(function_id).input_schema

        inputs_list.check_valid(input_schema)
        print(optimizer_id)
        print(inputs_list.to_dict_list())
        response = self._make_request('POST', 'batch-run/', data={"configuration": optimizer_id}, files=make_batch_input_file(inputs_list.to_dict_list())).json()
        if 'id' not in response:
            raise ZenbaseAPIError(response['detail'])
        self.batch_run_id_to_function_id_cache[response['id']] = function_id
        print("Batch run ID:", response['id'])
        return response['id']
    
    def get_batch_optimizer_run_status(self, batch_run_id: int) -> BatchFunctionRunStatus:
        response = self._make_request('GET', f'batch-run/{batch_run_id}/status').json()
        print(response)
        if 'status' not in response:
            raise ZenbaseAPIError(response['detail'])
        return BatchFunctionRunStatus(**response)
    
    def delete_batch_optimizer_run(self, batch_run_id: int) -> Any:
        # TODO: Update this function when the delete function returns something.
        self._make_request('DELETE', f'batch-run/{batch_run_id}')
        # print(response)
        # return response
        # if 'status' not in response:
        #     raise ZenbaseAPIError(response['detail'])
        # return BatchFunctionRunStatus(**response)

    def get_batch_optimizer_run_results(self, batch_run_id: int, block_until_completed: bool = True) -> BatchFunctionRunResults:
        batch_run_status = self.get_batch_optimizer_run_status(batch_run_id)
        if block_until_completed:
            next_sleep_time = 10 # seconds
            while batch_run_status.status == BatchFunctionRunStatusEnum.RUNNING:
                time.sleep(next_sleep_time)
                batch_run_status = self.get_batch_optimizer_run_status(batch_run_id)
                n_remaining = batch_run_status.total_runs - batch_run_status.completed_runs
                next_sleep_time = clamp(n_remaining / 2, 5, 30)
        else:
            if batch_run_status.status == BatchFunctionRunStatusEnum.RUNNING:
                raise ZenbaseAPIError("Batch run not completed")
            
        results = self._make_request('GET', f'function-run-logs/?batch_run={batch_run_id}&page=1').json()
        batch_optimizer_run_results = get_batch_optimizer_run_results_per_page(results['results'])
        count = results['count']
        for page in range(2, (count + 9) // 10):
            results = self._make_request('GET', f'function-run-logs/?batch_run={batch_run_id}&page={page}').json()
            batch_optimizer_run_results_per_page = get_batch_optimizer_run_results_per_page(results['results'])
            batch_optimizer_run_results.results.extend(batch_optimizer_run_results_per_page.results)
            batch_optimizer_run_results.failed_object_ids.extend(batch_optimizer_run_results_per_page.failed_object_ids)
        return batch_optimizer_run_results
    
    def get_batch_run_function_id(self, batch_run_id: Union[int, str]) -> int:
        if batch_run_id in self.batch_run_id_to_function_id_cache:
            return self.batch_run_id_to_function_id_cache[batch_run_id]
        elif isinstance(batch_run_id, int):
            optimizer_id = self._make_request('GET', f'batch-run/{batch_run_id}').json()['configuration']
            function_id = self.get_optimizer_function_id(optimizer_id)
            self.batch_run_id_to_function_id_cache[batch_run_id] = function_id
            return function_id
        else:
            raise ZenbaseAPIError(f"Batch run ID {batch_run_id} not found in local batch run history.")
        
    def update_function_config(self, function_id: int, function_config: ZenbaseFunctionConfig) -> ZenbaseFunctionConfig:
        """
        Updates the configuration of an existing function.

        Args:
            function_id (int): The ID of the function to update
            function_config (ZenbaseFunctionConfig): The new configuration to apply

        Returns:
            ZenbaseFunctionConfig: The updated function configuration

        Raises:
            ZenbaseAPIError: If the update request fails or returns invalid response
        """
        response = self._make_request('PATCH', f'functions/{function_id}', data=function_config.model_dump(exclude_none=True)).json()
        if 'id' not in response:
            raise ZenbaseAPIError(response['detail'])
        
        self.function_config_cache[function_id] = ZenbaseFunctionConfig(**response)
        return self.function_config_cache[function_id]

class ZenbaseAPIError(Exception):
    """Custom exception for Zenbase API errors"""
    pass