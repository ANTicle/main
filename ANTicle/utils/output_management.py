import csv
import json
import os
from fuzzywuzzy import fuzz
from .base_functions import create_dataframe

REPLACE_KEYWORDS = ['SEOTitel', 'SEOText', 'Titel', 'Teaser', 'Dachzeilen', 'Text', 'Liste']

#move old output to history
def rename_and_clear_output_file(output_file, history_file):
    """
    Rename and clear the output file.

    :param output_file: The path to the output file.
    :type output_file: str
    :param history_file: The desired name for the renamed output file.
    :type history_file: str
    :return: None
    :rtype: None
    """
    if os.path.exists(output_file):
        os.rename(output_file, history_file)
        open(output_file, 'w').close()


def save_generated_data_from_async_req(filename):
    """
    Saves generated data to a CSV file.

    :param filename: The name of the CSV file to save the data to.
    :return: The dataframe containing the saved data.
    """
    responses = get_responses_from_file(filename)
    data_rows = write_responses_to_csv(responses)
    df = create_dataframe(data_rows, 'Title', 'Generated Output')
    return df


def get_responses_from_file(filename):
    """
    Read responses from a file.

    :param filename: The name of the file to read from.
    :return: A list of response data read from the file.
    """
    responses = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            responses.append(data)
    return responses


def write_responses_to_csv(responses):
    """
    Write responses to a CSV file.

    :param responses: a list of responses to write to the CSV file
    :type responses: list
    :return: a list of rows written to the CSV file
    :rtype: list
    """
    rows = []
    with open('./Output_data/output.csv', 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Title', 'Generated Output'])

        for response in responses:
            title, generated_data = extract_data_from_response(response)
            row = [title, generated_data]
            csv_writer.writerow(row)
            rows.append(row)
    print("CSV file created successfully.")
    return rows


def extract_data_from_response(response):
    """
    Extracts data from the given response.

    :param response: The response containing the data.
    :return: A tuple containing the extracted title and generated data.
    """
    title = response[0]["messages"][0]["content"]
    generated_data = extract_generated_data(response)
    for keyword in REPLACE_KEYWORDS:
        if any(fuzz.ratio(keyword, word) >= 93 for word in title.split(' ')):
            title = keyword
            break
    return title, generated_data


def extract_generated_data(response):
    """
    Extracts generated data from a given response.

    :param response: The response containing the generated data.
    :type response: dict
    :return: The extracted generated data.
    :rtype: dict or str
    """
    try:
        return response[1]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    except KeyError:
        return response[1]["choices"][0]["message"]["content"]


def clear_files():
    open('./Output_data/output.csv', 'w').close()
    open('./temp/output.jsonl', 'w').close()
