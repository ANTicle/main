import os
import glob
from docx import Document

from async_requests import generate_chat_completion_requests
from base_functions import convert_csv_to_array
from define_genre import define_genre_and_create_variables_from_df


def read_docx(file):
    """
    Read the contents of a .docx file and return a string representation of the text.

    :param file: The file path of the .docx file to be read.
    :type file: str
    :return: The string representation of the text in the .docx file.
    :rtype: str
    """
    doc = Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return ' '.join(full_text)


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
    data = convert_csv_to_array(prompt_b_o_genre, 'temp/data.py')
    requests_filepath = 'temp/data.py'
    generate_chat_completion_requests(requests_filepath, data, input_collection, model_name="gpt-4-1106-preview")
    return requests_filepath, data
