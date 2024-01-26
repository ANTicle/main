import os
import glob
from docx import Document

def read_docx(file):
    doc = Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return ' '.join(full_text)


def read_and_concatenate_files(directory):
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
