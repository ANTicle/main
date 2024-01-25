import datetime
import aiohttp
import asyncio
import logging
import time
import json
from token_management import record_token_usage
from base_functions import append_to_jsonl, load_data_from_file
from endpoint_management import api_endpoint_from_url
from token_management import num_tokens_consumed_from_request
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata

# A class to keep track of the statuses of different tasks
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0     # Keeps track of the number of tasks that have been started
    num_tasks_in_progress: int = 0   # Keeps track of the number of tasks that are currently in progress, script ends when this reaches 0
    num_tasks_succeeded: int = 0    # Keeps track of the number of tasks that have been completed successfully
    num_tasks_failed: int = 0    # Keeps track of the number of tasks that failed
    num_rate_limit_errors: int = 0    # Keeps track of the number of errors due to rate limit exceeding
    num_api_errors: int = 0  #Keeps track of the number of errors due to API issues, excluding rate limit errors, counted above
    num_other_errors: int = 0    # Keeps track of the number of other types of errors
    time_of_last_rate_limit_error: int = 0  # Keeps track of the time when the last rate limit error occurred, used to cool off after hitting rate limits

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        print(self.__dict__)

def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1

# A class to make and manage API requests
@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""
    task_id: int    # The ID of the task that the API request is part of
    request_json: dict     # The JSON request payload for the API request
    token_consumption: int    # The number of tokens consumed by the API request
    attempts_left: int    # The number of attempts left to make the API request
    metadata: dict    # Any metadata associated with the API request
    result: list = field(default_factory=list)    # The result of the API request

    # A method to call the API
    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")
            # capture total_tokens and store it in CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total_tokens = response.get('usage', {}).get('total_tokens')
            record_token_usage(total_tokens)

async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
):
    """
    :param requests_filepath: The path to the file containing the API requests to process.
    :param save_filepath: The path to save the results of the API requests.
    :param request_url: The URL of the API endpoint to make the requests.
    :param api_key: The API key or authorization token to include in the request header.
    :param max_requests_per_minute: The maximum number of requests allowed per minute.
    :param max_tokens_per_minute: The maximum number of tokens allowed per minute.
    :param token_encoding_name: The name of the encoding used for tokens.
    :param max_attempts: The maximum number of attempts to retry a failed request.
    :param logging_level: The level of logging to output.
    :return: None
    """
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}
    # use api-key header for Azure deployments
    if '/deployments' in request_url:
        request_header = {"api-key": f"{api_key}"}
    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 1, 2, 3, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call
    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()
    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")
    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")
        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = json.loads(next(requests))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logging.debug("Read file exhausted")
                            file_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        # update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logging.warn(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # after finishing, log final status
        logging.info(
            f"""Parallel processing complete. Results saved to {save_filepath}"""
        )
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )


def generate_chat_completion_requests(filename, data, prompt, model_name="gpt-4-1106-preview"):
    """
    :param filename: The name of the file to write the chat completions to.
    :param data: A list of chat messages.
    :param prompt: The prompt for generating chat completions.
    :param model_name: The name of the model to use for generating chat completions. Defaults to "gpt-4-1106-preview".
    :return: None

    This method generates chat completion requests and writes them to a file. It takes in the filename to write the completions to, the data which is a list of chat messages, the prompt
    * for generating completions, and an optional model_name parameter which specifies the model to use.

    If the 'data' parameter is not a list, it will be assumed to be a filename and the data will be loaded from the file.

    Each chat completion request consists of two messages - a system message representing the previous chat message and a user message containing the prompt. These messages are then written
    * to the file in JSONL format.
    """
    # Check if 'data' is a list and read from file if necessary
    if not isinstance(data, list):
       data = load_data_from_file(filename)

    # Write chat completions to file
    with open(filename, "w") as f:
        for chat_message in data:
            # Create a list of messages for each request
            messages = [
                {"role": "system", "content": str(chat_message)},
                {"role": "user", "content":  prompt}
            ]
            # Write the messages to the JSONL file
            json_string = json.dumps({"model": model_name, "messages": messages})
            f.write(json_string + "\n")
