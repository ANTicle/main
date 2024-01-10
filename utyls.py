import pandas as pd
from openai import OpenAI
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL & processing the csv
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
import csv
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata

def convert_csv_to_array(input_csv_path, output_python_path):
    # get the csv structure
    # put each element into the array
    data = []
    with open(input_csv_path, "r") as input_file:
        csv_reader = csv.reader(input_file)
        for row in csv_reader:
            if row:
                data.append(row[1])
    with open(output_python_path, "w") as data_file:
        data_file.write(f'data = {data}')
    return data
def save_generated_data_to_csv(filename):
    # takes the data as input
    # Write it into the CSV file
    responses = []

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            responses.append(data)

    # Create a CSV file for writing
    with open('output.csv', 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the header row
        csv_writer.writerow(['Titel', 'Generated Output'])

        # Iterate through the specialists and write data to the CSV
        for response in responses:
            # The 'Title' is declared inside the loop, it's local to the for loop
            Title = response[0]["messages"][0]["content"]

            try:
                generated_data = response[1]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            except:
                generated_data = response[1]["choices"][0]["message"]["content"]

            # List of keywords to replace
            replace_keywords = ['SEO-Titel', 'SEO-Text', 'Titel', 'Teaser', 'Dachzeile', 'Text', 'Liste']

            # Check each keyword
            for keyword in replace_keywords:
                # If the keyword is in the first 30 characters of the title
                if re.search(keyword, Title[:30], re.I):
                    # Replace the whole string with the keyword and break the loop
                    Title = keyword
                    break

            # Write data to the CSV file
            csv_writer.writerow([Title, generated_data])

    print("CSV file created successfully.")
def generate_chat_completion_requests(filename, data, prompt, model_name="gpt-3.5-turbo"):
    # check if 'data' is a list
    if not isinstance(data, list):
        # open the file and read data
        file = os.path.join(os.getcwd(), filename)
        if os.path.isfile(file):
            with open(file, 'r') as f:
                data = f.readlines()  # depending on your file format, adjust this line
    # 'data' is a list or the file is read successfully into a list
    with open(filename, "w") as f:
        for x in data:
            # Create a list of messages for each request
            messages = [
                {"role": "system", "content": str(x)},
                {"role": "user", "content":  prompt}
            ]
            # Write the messages to the JSONL file
            json_string = json.dumps({"model": model_name, "messages": messages})
            f.write(json_string + "\n")

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
    """Processes API requests in parallel, throttling to stay under rate limits."""
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


# dataclasses


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


# functions


def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
    return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1

'''
def input_to_list(input_collection, model="gpt-4-1106-preview", max_tokens=1000):
    """
    Transforms the input to a list using OpenAI's chat model.
    :param input_collection: The input text collection.
    :param model: The OpenAI model to use.
    :param max_tokens: The maximum number of tokens for the response.
    """
    logging.info('Transforming input to list')
    list_prompt = "Read the following input and create a list of all the facts contained within it. "
    full_prompt = list_prompt + input_collection
    requests = [APIRequest(full_prompt, model, max_tokens)]

    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        responses = asyncio.run(process_requests(requests, api_key))
        for response in responses:
            print(f"Response for variable: {response['choices'][0]['message']['content']}")
            logging.info(f"Received response: {response['choices'][0]['message']['content']}")
    except Exception as e:
        logging.error(f'Error processing list: {e}')
        print(f'Error processing list: {e}')
'''

def define_genre_and_create_variables_from_df(input_string, model="gpt-4-1106-preview", max_tokens=1000):
    logging.info('define_genre function called.')
    try:
        client = OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY'),
        )
        logging.info('OpenAI client initialized.')
        print("OpenAI client initialized.")
    except Exception as e:
        logging.error('Failed to initialize OpenAI client: %s' % e)
        print('Failed to initialize OpenAI client:', e)
        return
    with open('prompt.txt', 'r') as file:
        prompt_txt = file.read()
    prompt = f"{prompt_txt}\n{input_string}"
    #print(prompt)
    try:
        response = client.chat.completions.create(model=model,
                                                  messages=[{"role": "system", "content": "You are a journalist."},
                                                            {"role": "user", "content": prompt}],
                                                  max_tokens=max_tokens)
        print(response)
        file_name = response.choices[0].message.content.strip() + '.csv'
        var_name = file_name.split(".")[0]  # removing .csv to get var_name
        logging.info('Processing variable: %s' % var_name)
        print('Processing variable:', var_name)
        return file_name
    except FileNotFoundError:
        logging.error('File %s was not found.' % file_name)
        print('FileNotFoundError:', file_name)
    except Exception as e:
        logging.error('Failed to create response for variable: %s. Error: %s' % (var_name, e))
        print('Failed to create response for variable:', var_name)


if __name__ == "__main__":
    with open('test.txt', 'r') as file:
        prompt = file.read()
    input = define_genre_and_create_variables_from_df(prompt)
    data = convert_csv_to_array(input, "data.py")
    requests_filepath = 'data.py'

    generate_chat_completion_requests(requests_filepath, data, prompt, model_name="gpt-4-1106-preview")

    # Process multiple api requests to ChatGPT
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=requests_filepath,
            save_filepath='output.jsonl',
            request_url="https://api.openai.com/v1/chat/completions",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_requests_per_minute=float(90000),
            max_tokens_per_minute=float(170000),
            token_encoding_name="cl100k_base",
            max_attempts=int(5),
            logging_level=int(20),
        )
    )

    save_generated_data_to_csv('output.jsonl')