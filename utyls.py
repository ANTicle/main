from openai import OpenAI
import aiohttp  # for making API calls concurrently
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL & processing the csv
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
import csv
import datetime
from fuzzywuzzy import fuzz
import pandas as pd
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata

def convert_csv_to_array(input_csv_path, output_python_path):
    """
    :param input_csv_path: The path to the input CSV file.
    :param output_python_path: The path to the output Python file.
    :return: The data array.

    Converts the data in the input CSV file into a Python array and writes it to the output Python file. Each row of the CSV file is treated as an element of the array.

    Example usage:

        input_csv_path = 'input.csv'
        output_python_path = 'output.py'
        data = convert_csv_to_array(input_csv_path, output_python_path)

    This will read the data from 'input.csv', convert it to an array, and write it to 'output.py'. The resulting array will be stored in the 'data' variable.
    """
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
    """
    :param filename: The name of the input file that contains the data to be saved into the CSV file.
    :return: A pandas dataframe containing the saved data.

    """
    # takes the data as input
    # Write it into the CSV file
    responses = []

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            responses.append(data)
    rows = []
    # Create a CSV file for writing
    with open('output.csv', 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Title', 'Generated Output'])

        for response in responses:
            Title = response[0]["messages"][0]["content"]
            try:
                generated_data = response[1]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            except:
                generated_data = response[1]["choices"][0]["message"]["content"]

            replace_keywords = ['SEOTitel', 'SEOText', 'Titel', 'Teaser', 'Dachzeilen', 'Text', 'Liste']

            for keyword in replace_keywords:
                if any(fuzz.ratio(keyword, word) >= 93 for word in
                       Title.split(' ')):  # check if any word in the title is similar to current keyword
                    Title = keyword
                    break

            row = [Title, generated_data]
            csv_writer.writerow(row)
            rows.append(row)

    print("CSV file created successfully.")
    # Create a pandas dataframe for further analysis
    df = pd.DataFrame(rows, columns=['Title', 'Generated Output'])
    print(df)
    return df

def get_text_value(df):
    """
    Check if the title "Text" exists in the DataFrame.

    :param df: A pandas DataFrame containing the data.
    :return: The extracted text value corresponding to the title "Text" if found, otherwise None.
    """
    # Check if the title "Text" exists in the DataFrame
    if 'Text' in df['Title'].values:
        # Extract the Generated Output corresponding to the Title "Text"
        output = df.loc[df['Title'] == 'Text', 'Generated Output'].values[0]
        return str(output)
    else:
        print("Title 'Text' not found in the DataFrame.")
        return None

def generate_chat_completion_requests(filename, data, prompt, model_name="gpt-4-1106-preview"):
    """
    :param filename: str, the name of the file to write the chat completion requests to
    :param data: list or str, the input data for the chat completion requests
    :param prompt: str, the prompt for the user message in each chat completion request
    :param model_name: str, the name of the model to use for generating chat completions (default is "gpt-4-1106-preview")
    :return: None

    Generates chat completion requests based on the given data and writes them to the specified file.
    If 'data' is a list, it will be used as is. Otherwise, it will be treated as a file name and the contents
    of the file will be read into a list. Each element in the list represents a chat message.

    Each chat completion request consists of a system message followed by a user message. The system message
    is the current chat message from 'data' and the user message is the given 'prompt'.

    The chat completion request is written to the file in JSONL format, with each request as a separate line.
    Each JSON object in a line contains the 'model' name and a list of 'messages'. Each message in the list
    has a 'role' ("system" or "user") and 'content' (the actual message text).
    """
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
    """
    :param requests_filepath: The file path to the list of API requests to process.
    :param save_filepath: The file path to save the results of the API requests.
    :param request_url: The URL of the API endpoint to call.
    :param api_key: The API key used for authorization.
    :param max_requests_per_minute: The maximum number of API requests allowed per minute.
    :param max_tokens_per_minute: The maximum number of tokens allowed per minute.
    :param token_encoding_name: The name of the encoding used for tokens.
    :param max_attempts: The maximum number of attempts for each API request.
    :param logging_level: The logging level for the debug information.

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
            # capture total_tokens and store it in CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                total_tokens = response.usage.total_tokens  # assuming 'response' is defined
            except AttributeError:
                print(f"Failed to get total_tokens from response.usage.total_tokens: {response}")
                try:
                    total_tokens = response.usage.total_tokens  # assuming 'response' is defined
                except AttributeError:
                    print(f"Failed to get total_tokens from response.usage.total_tokens: {response}")
                    try:
                        total_tokens = response.get('usage', {}).get('total_tokens')
                    except AttributeError:
                        print(f"Failed to get total_tokens from response: {response}")
                        return

            filename = 'tokens_log.csv'
            history_filename = 'token_count_history.csv'
            headers = ['time', 'tokens']

            if os.path.isfile(filename):
                with open(filename, 'r') as file:
                    last_line = file.readlines()[-1]

                last_timestamp = datetime.datetime.strptime(last_line.split(",")[0], "%Y-%m-%d %H:%M:%S")

                # If last timestamp is not from today, move the existing file contents to the history file
                if last_timestamp.date() != datetime.date.today():

                    if os.path.isfile(history_filename):  # If the history file exists, append contents
                        with open(history_filename, 'a') as history_file:
                            with open(filename, 'r') as f:
                                history_file.write(f.read())
                    else:  # If the history file does not exist, rename the current file
                        os.rename(filename, history_filename)

                    # Create a new 'tokens_log.csv' and write headers
                    with open(filename, 'w', newline='') as tokens_log:
                        writer = csv.writer(tokens_log)
                        writer.writerow(headers)

            # Write the token usage to the current file
            with open(filename, 'a', newline='') as tokens_log:
                writer = csv.writer(tokens_log)
                writer.writerow([timestamp, total_tokens])

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

def compile_chat_request(content, model="gpt-4-1106-preview", max_tokens=1000):
    """
    Function to compile a chat request
    :param input_string: The input string from the user
    :param model: The model to be utilized. Default is "gpt-4-1106-preview"
    :param max_tokens: Maximum tokens for the response. Default is 1000
    :return: The response from the chat completion
    """
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
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a journalist."
            },
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=max_tokens
    )

    # capture total_tokens and store it in CSV
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        total_tokens = response.usage.total_tokens  # assuming 'response' is defined
        print(f"Got response: {response}")
    except AttributeError:
        total_tokens = response.get('usage', {}).get('total_tokens')
        print(f"Failed to get total_tokens from response: {response}")  # assuming 'response' is defined
        return

    filename = 'tokens_log.csv'
    history_filename = 'token_count_history.csv'
    headers = ['time', 'tokens']

    if os.path.isfile(filename):
        with open(filename, 'r') as file:
            last_line = file.readlines()[-1]

        last_timestamp = datetime.datetime.strptime(last_line.split(",")[0], "%Y-%m-%d %H:%M:%S")

        # If last timestamp is not from today, move the existing file contents to the history file
        if last_timestamp.date() != datetime.date.today():

            if os.path.isfile(history_filename):  # If the history file exists, append contents
                with open(history_filename, 'a') as history_file:
                    with open(filename, 'r') as f:
                        history_file.write(f.read())
            else:  # If the history file does not exist, rename the current file
                os.rename(filename, history_filename)

            # Create a new 'tokens_log.csv' and write headers
            with open(filename, 'w', newline='') as tokens_log:
                writer = csv.writer(tokens_log)
                writer.writerow(headers)

    # Write the token usage to the current file
    with open(filename, 'a', newline='') as tokens_log:
        writer = csv.writer(tokens_log)
        writer.writerow([timestamp, total_tokens])

    return response

def facts_to_list(input_string, list_name):
    """
    Transform the input string into a list.

    :param input_string: The input string containing facts.
    :param list_name: The name of the list to create.
    :return: The created list.
    """
    logging.info('Transforming input to list')
    list_prompt = "Lies den folgenden input und erstelle eine sehr detailierte, vollständige Liste aller enthaltenen Fakten. Liste die Fakten mit Spiegelstrichen auf."
    full_prompt = list_prompt + input_string
    list_name = compile_chat_request(full_prompt)
    list_name_choice = list_name.choices[0].message.content.strip()  # Retrieve the first Choice object
    print(list_name_choice)
    return list_name_choice

def compare_facts(input, output):
    """
    :param input: The first text containing facts to be compared
    :param output: The second text containing facts to be compared
    :return: The user's choice after comparing the facts

    This method takes in two texts, `input` and `output`, and compares the facts in these texts. The user is prompted to choose whether there are conflicting facts or missing details between
    * the texts. The method returns the user's choice as a string. The method makes use of the `compile_chat_request` function to display the texts to the user and retrieve their choice
    *.
    """
    list_prompt = '''Agiere als Journalist. Du bekommst im Folgenden zwei Informationsquellen zu einem Thema. Lies sie und überprüfe, ob sich Fakten in den Texten widersprechen. Überprüfe besonders, ob Zahlen übereinstimmen. Wenn sich Fakten wiedersprechen ist das ein Problem. 
    Achte außerdem darauf das der zweite text, keine Fakten enthält, welche so nicht auch in dem ersten Text vorhanden sind. Wenn du in dem zweiten Text Fakten findest, welche nicht in dem ersten enthalten sind ist das ein Problem.
    Der zweite Text darf weniger details einhalten als der erste. Das ist kein Problem. 
    Wenn ein signifikantes Problem vorhanden ist, antworte mit dem Wort "Problem:" gefolgt von einer Beschreibung des Problems. Wenn kein Problem vorhanden ist, sondern nur fehelende Details antworte mit "Fehlende Details:" gefolgt von einer Liste der fehlenden Details.''' + '\n' + '\n' + "Erster Text" + '\n' + str(input) + '\n' + '\n' + "Zweiter Text" + '\n' + str(output)
    compare = compile_chat_request(list_prompt)
    choice = compare.choices[0].message.content.strip()  # Retrieve the first Choice object

    return choice

def define_genre_and_create_variables_from_df(input_string):
    """
    Defines the genre and creates variables from the given input DataFrame.

    :param input_string: The input string to be used as input to the GPT-3 model.
    :return: The name of the prompt file to use.
    """
    logging.info('define_genre function called.')
    with open('prompt.txt', 'r') as file:
        prompt_txt = file.read()
    prompt = f"{prompt_txt}\n{input_string}"
    try:
        response = compile_chat_request(prompt)
        print(response)
        file_name = response.choices[0].message.content.strip() + '.csv'
        print(file_name)
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