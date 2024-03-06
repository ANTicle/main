import os
import glob
import docx2txt
from django.shortcuts import render

from .async_requests import generate_chat_completion_requests
from .base_functions import convert_csv_to_array
from .define_genre import define_genre_and_create_variables_from_df
from django.http import HttpResponse
from ..forms import InputDataForm
import pandas as pd
import json


def format_form(input_json):
    string_output = ""
    start = False
    try:
        with open(input_json, 'r') as f:
            # attempt to load JSON data from the file
            parsed_json = json.load(f)

        for key in parsed_json:
            if key == "Quelle_1":
                start = True
            if start and parsed_json[key]:
                string_output += f"{key}:\n\n{parsed_json[key]}\n\n"

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json}")
    except FileNotFoundError:
        print(f"Error: File {input_json} not found")

    return string_output

def read_docx(file):
    """
    Read the contents of a .docx file and return a string representation of the text.
    :param file: The file path of the .docx file to be read.
    :type file: str
    :return: The string representation of the text in the .docx file.
    :rtype: str
    """
    text = docx2txt.process(file)
    return text


def read_and_concatenate_files(directory):
    """
    Read and concatenate files in a directory.

    :param directory: The directory containing the files.
    :return: The concatenated contents of the files.
    """
    files = glob.glob(os.path.join(directory, '*'))
    all_contents = ""
    for idx, file in enumerate(files, start=1):
        if file.endswith('.txt'):
            with open(file, 'r') as f:
                contents = f.read()
        elif file.endswith('.docx'):
            contents = read_docx(file)
        else:
            continue  # skip non-txt and non-docx files

        all_contents += f'"Quelle {idx}: " {contents} '

    return all_contents


def process_data(input_collection):
    """
    Process the given input collection.

    :param input_collection: The input collection.
    :return: The file path for chat completion requests and the processed data.
    """
    prompt_b_o_genre = define_genre_and_create_variables_from_df(input_collection)
    print('input: ' + prompt_b_o_genre + ' stop input')
    # Read CSV data
    with open(define_genre_and_create_variables_from_df(input_collection), 'r') as file:
        csv_data = file.read()
    # reading your json into a pandas dataframe
    with open('remaining.json') as f:
        json_data = json.load(f)

    # replacing 'länge_des_Artikels' with 'words'
    csv_data = csv_data.replace('length_of_Artikels', str(json_data['words']))    # replacing 'Thema_des_Artikels' with the value of 'thema'
    csv_data = csv_data.replace('Thema_des_Artikels', str(json_data['thema']))

    # convert the dataframe back to csv
    with open('temp/data.py', 'w') as file:
        file.write(csv_data)
    data = csv_data
    requests_filepath = 'temp/data.py'
    generate_chat_completion_requests(requests_filepath, data, input_collection, model_name="gpt-4-1106-preview")
    return requests_filepath, data

def split_json_string(json_obj):

    # Separate Quelle keys
    quelle_dict = {k: v for k, v in json_obj.items() if k.startswith('Quelle')}

    with open('quelle.json', 'w') as file:
        json.dump(quelle_dict, file)

    # Put the remaining keys in another dictionary
    remaining_dict = {k: v for k, v in json_obj.items() if not k.startswith('Quelle') and k != 'csrfmiddlewaretoken'}

    with open('remaining.json', 'w') as file:
        json.dump(remaining_dict, file)

    return quelle_dict, remaining_dict
