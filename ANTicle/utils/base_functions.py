import csv
import json
import logging
import os

import pandas as pd
from openai import OpenAI

from .token_management import record_token_usage


def append_to_jsonl(data, filename: str) -> None:
    """
    Append a JSON string to a JSON Lines file.

    :param data: The data to be appended as a JSON string.
    :param filename: The filename of the JSON Lines file.
    :return: None
    """
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


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
        csv_reader = csv.reader(input_file, delimiter=';')
        header = next(csv_reader)  # Skip the header
        index = header.index("Prompt")
        for row in csv_reader:
            if row:
                print(row)
                data.append(row[index])
    with open(output_python_path, "w") as data_file:
        data_file.write(f'data = {data}')
    return data


def create_dataframe(rows, column1, column2, column3):
    """
    Create a pandas DataFrame from a given list of rows.

    :param rows: A list of rows to be used as data for the DataFrame.
    :return: The created pandas DataFrame.

    """
    df = pd.DataFrame(rows, columns=[column1, column2, column3])
    #print(df)
    return df


def load_data_from_file(filename):
    """
    :param filename: str, the name of the file to read data from
    :return: list, the content read from the file
    Opens the file and reads the content into a list.
    """
    file = os.path.join(os.getcwd(), filename)
    if os.path.isfile(file):
        with open(file, 'r') as f:
            data = f.readlines()  # depending on your file format, adjust this line
        return data
    return None


def linear_chat_request(content, model="gpt-4-1106-preview", max_tokens=1000):
    """
    :param content: The content of the user's message for the chat completion.
    :param model: The model to use for the chat completion. Default is "gpt-4-1106-preview".
    :param max_tokens: The maximum number of tokens to generate in the chat completion. Default is 1000.
    :return: The response from the chat completions API.

    This method compiles a chat request using the OpenAI client. It initializes the OpenAI client with the provided API key, and then makes a request to the chat completions API using the
    * specified model, user message content, and maximum tokens.
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

    try:
        total_tokens = response.usage.total_tokens
    except AttributeError:
        total_tokens = response.get('usage', {}).get('total_tokens')

    record_token_usage(total_tokens)
    return response
