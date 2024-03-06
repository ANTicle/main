import csv
import json
import os
import re
from rapidfuzz import fuzz, process
from .base_functions import create_dataframe
from collections import defaultdict

from django.http import JsonResponse


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
        # Using rapidfuzz to measure string similarity
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


def add_additional_content():
    """
    Read a txt file and add its content to the second column of the CSV file. Write "Zusatz" to the first column.
    """
    # read the content of the text file
    with open('./Output_data/Fehlende_Details.txt', 'r', encoding='utf-8') as txt_file:
        text_data = txt_file.read()

    # write the content to the CSV file
    with open('./Output_data/output.csv', 'a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        row = ['Zusatz', text_data]
        csv_writer.writerow(row)
    print("Content of text file added to CSV file.")

def csv_to_json(request):
    """
    Convert a CSV file to a JSON response.
    :param request: The HTTP request object.
    :return: A JSON response object containing the converted data.
    """
    data_dict = defaultdict(dict)
    with open('./Output_data/output.csv', 'r') as f:
        reader = csv.reader(f)
        headers = next(reader, None)  # get headers
        for row in reader:
            for header, field in zip(headers, row):
                if header not in ['Text', 'Zusatz']:
                    sub_keys = field.split("\n")  # split on linebreak
                    for idx, key in enumerate(sub_keys):
                        if len(key) > 20:
                            data_dict[row[0]]["sub_key_{:02d}".format(idx)] = key  # store if len > 20
                else:
                    data_dict[row[0]][header] = field  # normal row if 'Text' or 'Zusatz'

    with open('./Output_data/Fehlende_Details.txt', 'r') as f:
        details = f.read()
        data_dict['Zusatz'] = details

    # remove fields shorter than 20 characters
    for key in list(data_dict):  # using list to avoid RuntimeError due to size change
        if len(data_dict[key]) <= 20:
            del data_dict[key]
    print(JsonResponse(data_dict))
    return JsonResponse(data_dict)

