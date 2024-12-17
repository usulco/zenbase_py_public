# zenbase_client.py
import os
import asyncio
import aiohttp
from typing import Optional, Dict, Any, Union, BinaryIO, List

from .models import (
    ZenbaseConfig,
    ZenbaseFunctionConfig,
    BatchFunctionInputList,
    BatchFunctionRunStatus,
    BatchFunctionRunStatusEnum,
    BatchFunctionRunResults,
)
from .helpers import make_batch_input_file, clamp, get_batch_optimizer_run_results_per_page
from collections import Counter

class ZenbaseAPIError(Exception):
    """Custom exception for Zenbase API errors"""
    pass


class AsyncZenbaseClient:
    """
    Asynchronous version of the ZenbaseClient using aiohttp for non-blocking I/O.
    """

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

        # Aiohttp session
        self.session = None  # We'll create an aiohttp.ClientSession on-demand or in an async context

        self.optimizer_function_id_cache: Dict[int, int] = {}
        self.batch_run_id_to_function_id_cache: Dict[Union[int, str], int] = {}
        self.function_config_cache: Dict[int, ZenbaseFunctionConfig] = {}

    async def __aenter__(self):
        """Optional: Support async context manager usage."""
        timeout_value = self.config.timeout if self.config.timeout else 30
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_value)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close session when exiting context."""
        if self.session:
            await self.session.close()

    async def _make_async_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Union[BinaryIO, tuple]]] = None,
        **kwargs
    ) -> dict:
        """
        Make HTTP request to Zenbase API asynchronously using aiohttp.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: URL query parameters
            data: Request body data (will be sent as JSON if files=None)
            files: Dict of files to upload. Can be file objects or tuples of (filename, file object)
            **kwargs: Additional arguments to pass to aiohttp.ClientSession.request()

        Returns:
            dict: JSON response from the API (or dict with 'text' if response is not JSON)

        Raises:
            ZenbaseAPIError: If the request fails or returns non-2xx status.
        """
        # Ensure an aiohttp session exists. If not, create one.
        if not self.session:
            timeout_value = self.config.timeout if self.config.timeout else 30
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout_value)
            )

        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Api-Key {self.config.api_key}"
        }

        # Decide how to send data based on whether 'files' is given (mimic the sync approach):
        if files:
            # Prepare multipart/form-data
            form_data = aiohttp.FormData()
            if data:
                # Add each key-value pair in 'data' to the form fields
                for field_name, field_value in data.items():
                    form_data.add_field(field_name, str(field_value))

            # Add files as form fields
            for file_field, file_obj in files.items():
                # If tuple, we expect (filename, filehandle)
                # Or possibly (filename, filehandle, content_type)
                if isinstance(file_obj, tuple):
                    if len(file_obj) == 2:
                        filename, filehandle = file_obj
                        form_data.add_field(file_field, filehandle, filename=filename)
                    elif len(file_obj) == 3:
                        filename, filehandle, content_type = file_obj
                        form_data.add_field(file_field, filehandle, filename=filename, content_type=content_type)
                    else:
                        raise ValueError(f"File tuple must have 2 or 3 elements: {file_obj}")
                else:
                    # If it's just a file object with no filename, or any other shape
                    form_data.add_field(file_field, file_obj)

            # For multipart, we do not manually set Content-Type; aiohttp.FormData does it automatically
            request_json = None
            request_data = form_data

        else:
            # No files -> send JSON
            headers["Content-Type"] = "application/json"
            request_data = None
            request_json = data

        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                json=request_json,
                data=request_data,
                headers=headers,
                **kwargs
            ) as response:
                # Raise exception if non-2xx
                if response.status >= 400:
                    error_text = await response.text()
                    raise ZenbaseAPIError(
                        f"API request failed (status={response.status}): {error_text}"
                    )
                
                # Attempt to parse JSON; fallback to returning raw text if parsing fails
                try:
                    return await response.json()
                except aiohttp.ContentTypeError:
                    text_response = await response.text()
                    return {"text": text_response}

        except aiohttp.ClientError as e:
            raise ZenbaseAPIError(f"API request failed: {str(e)}") from e

    async def get_optimizer_function_id(self, optimizer_id: int, update_cache: bool = True) -> int:
        if optimizer_id in self.optimizer_function_id_cache:
            return self.optimizer_function_id_cache[optimizer_id]

        response = await self._make_async_request('GET', f'optimizer-configurations/{optimizer_id}')
        if 'id' not in response:
            raise ZenbaseAPIError(response.get('detail', "Unknown response format"))

        function_id = response['function']
        if update_cache:
            self.optimizer_function_id_cache[optimizer_id] = function_id

        return function_id

    async def get_function_config(self, function_id: int, update_cache: bool = True) -> ZenbaseFunctionConfig:
        if function_id in self.function_config_cache:
            return self.function_config_cache[function_id]

        response = await self._make_async_request('GET', f'functions/{function_id}')
        if 'id' not in response:
            raise ZenbaseAPIError(response.get('detail', "Unknown response format"))

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

    async def start_batch_optimizer_run(self, optimizer_id: int, inputs_list: BatchFunctionInputList) -> int:
        """
        Start a batch run of the optimizer function with the given inputs.
        """
        function_id = await self.get_optimizer_function_id(optimizer_id)
        input_schema = (await self.get_function_config(function_id)).input_schema

        inputs_list.check_valid(input_schema)
        print("optimizer_id:", optimizer_id)
        print(inputs_list.to_dict_list())

        # Make the batch input file payload
        file_payload = make_batch_input_file(inputs_list.to_dict_list())

        response = await self._make_async_request(
            'POST',
            'batch-run/',
            data={"configuration": optimizer_id},
            files=file_payload
        )
        if 'id' not in response:
            raise ZenbaseAPIError(response.get('detail', "Invalid response from batch-run creation"))

        batch_run_id = response['id']
        self.batch_run_id_to_function_id_cache[batch_run_id] = function_id
        print("Batch run ID:", batch_run_id)
        return batch_run_id

    async def get_batch_optimizer_run_status(self, batch_run_id: int) -> BatchFunctionRunStatus:
        response = await self._make_async_request('GET', f'batch-run/{batch_run_id}/status')
        print(response)
        if 'status' not in response:
            raise ZenbaseAPIError(response.get('detail', "Unknown response format"))
        return BatchFunctionRunStatus(**response)

    async def delete_batch_optimizer_run(self, batch_run_id: int) -> Any:
        """
        Delete the batch run (if API supports deletion).
        """
        # This endpoint returns no body, or some minimal JSON upon success
        await self._make_async_request('DELETE', f'batch-run/{batch_run_id}')
        # Nothing to return, so you can return True or a custom message
        return True

    async def get_batch_optimizer_run_results(self, batch_run_id: int, block_until_completed: bool = True) -> BatchFunctionRunResults:
        batch_run_status = await self.get_batch_optimizer_run_status(batch_run_id)

        if block_until_completed:
            next_sleep_time = 10  # seconds
            while batch_run_status.status == BatchFunctionRunStatusEnum.RUNNING:
                await asyncio.sleep(next_sleep_time)
                batch_run_status = await self.get_batch_optimizer_run_status(batch_run_id)
                n_remaining = batch_run_status.total_runs - batch_run_status.completed_runs
                next_sleep_time = clamp(n_remaining / 2, 5, 30)
        else:
            if batch_run_status.status == BatchFunctionRunStatusEnum.RUNNING:
                raise ZenbaseAPIError("Batch run not completed")

        # Retrieve results from function-run-logs
        response = await self._make_async_request('GET', f'function-run-logs/?batch_run={batch_run_id}&page=1')
        batch_optimizer_run_results = get_batch_optimizer_run_results_per_page(response['results'])
        count = response['count']

        total_pages = (count + 9) // 10  # each page has up to 10 results
        for page in range(2, total_pages + 1):
            response = await self._make_async_request('GET', f'function-run-logs/?batch_run={batch_run_id}&page={page}')
            page_results = get_batch_optimizer_run_results_per_page(response['results'])
            batch_optimizer_run_results.results.extend(page_results.results)
            batch_optimizer_run_results.failed_object_ids.extend(page_results.failed_object_ids)

        counter = Counter(batch_optimizer_run_results.failed_object_ids)
        print("Object IDs with duplicate failed runs:", [item for item, count in counter.items() if count > 1])
        batch_optimizer_run_results.failed_object_ids = list(counter.keys())
        return batch_optimizer_run_results

    async def get_batch_run_function_id(self, batch_run_id: Union[int, str]) -> int:
        if batch_run_id in self.batch_run_id_to_function_id_cache:
            return self.batch_run_id_to_function_id_cache[batch_run_id]
        elif isinstance(batch_run_id, int):
            resp = await self._make_async_request('GET', f'batch-run/{batch_run_id}')
            if 'configuration' not in resp:
                raise ZenbaseAPIError(resp.get('detail', "Unknown response format"))
            optimizer_id = resp['configuration']
            function_id = await self.get_optimizer_function_id(optimizer_id)
            self.batch_run_id_to_function_id_cache[batch_run_id] = function_id
            return function_id
        else:
            raise ZenbaseAPIError(f"Batch run ID {batch_run_id} not found in local batch run history.")

    async def update_function_config(self, function_id: int, function_config: ZenbaseFunctionConfig) -> ZenbaseFunctionConfig:
        """
        Updates the configuration of an existing function asynchronously.
        """
        response = await self._make_async_request(
            'PATCH',
            f'functions/{function_id}',
            data=function_config.model_dump(exclude_none=True)
        )
        if 'id' not in response:
            raise ZenbaseAPIError(response.get('detail', "Unknown response format"))

        updated_config = ZenbaseFunctionConfig(**response)
        self.function_config_cache[function_id] = updated_config
        return updated_config
