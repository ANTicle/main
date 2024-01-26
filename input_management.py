import os
import glob
from docx import Document

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
