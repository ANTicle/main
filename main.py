import os
import json
import csv
import asyncio
import pandas as pd
from utyls import define_genre_and_create_variables_from_df, convert_csv_to_array, generate_chat_completion_requests, process_api_requests_from_file, save_generated_data_to_csv, compare_facts, get_text_value

def validate_and_swap_api_key(token_sum):
    max_api_usage = 500000
    api_key = os.getenv('OPENAI_API_KEY')  # default value
    usage_ratio = abs(token_sum - max_api_usage) / max_api_usage
    if usage_ratio <= 0.05:
        api_key = os.environ.get('OPENAI_API_KEY_2', os.getenv('OPENAI_API_KEY'))
    elif os.environ['OPENAI_API_KEY'] == os.environ.get('OPENAI_API_KEY_2', ''):
        api_key = os.getenv('OPENAI_API_KEY_3')
    elif os.environ['OPENAI_API_KEY'] == os.environ.get('OPENAI_API_KEY_3', ''):
        api_key = os.getenv('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = api_key

def append_output_to_history(output_file, history_file):
    with open(output_file, 'r') as out_f, open(history_file, 'r+') as hist_f:
        if '.jsonl' in output_file:
            output_data = [json.loads(line) for line in out_f]
            history_data = [json.loads(line) for line in hist_f]
            history_data.extend(output_data)
            hist_f.seek(0)
            json.dump(history_data, hist_f)
            hist_f.truncate()
        else:
            reader = csv.reader(out_f)
            writer = csv.writer(hist_f)
            writer.writerows(reader)

def rename_and_clear_output_file(output_file, history_file):
    if os.path.exists(output_file):
        os.rename(output_file, history_file)
        open(output_file, 'w').close()

def print_file_contents(output_file):
    with open(output_file, 'r') as file:
        contents = file.read()
        if contents:
            print(f'Contents of {output_file}', contents)
        else:
            print(f'{output_file} is empty.')

if __name__ == "__main__":
    output_files = ['output.jsonl', 'output.csv']
    history_files = ['history.jsonl', 'history.csv']
    if os.path.exists('tokens_log.csv'):
        df = pd.read_csv('tokens_log.csv')
        token_sum = df['tokens'].sum()
        print(token_sum)
        validate_and_swap_api_key(token_sum)

    for output_file, history_file in zip(output_files, history_files):
        if os.path.exists(output_file) and os.path.exists(history_file):
            append_output_to_history(output_file, history_file)
        else:
            rename_and_clear_output_file(output_file, history_file)
        print_file_contents(output_file)

    with open('test.txt', 'r') as file:
        input_collection = file.read()
    input = define_genre_and_create_variables_from_df(input_collection)
    print(input)
    data = convert_csv_to_array(input, "data.py")
    requests_filepath = 'data.py'
    generate_chat_completion_requests(requests_filepath, data, input_collection, model_name="gpt-4-1106-preview")
    first_loop = True
    while True:
        if not first_loop:
            open('output.csv', 'w').close()
            open('output.jsonl', 'w').close()
        first_loop = False
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
        df = save_generated_data_to_csv('output.jsonl')
        output_text = get_text_value(df)
        hallucination_check = compare_facts(input_collection, output_text)
        print(hallucination_check)
        if "Problem" in hallucination_check:
            continue
        elif "Fehlende Details" in hallucination_check:
            with open('output.txt', 'w') as file:
                file.write(hallucination_check)
            break
        else:
            pass
        # Check if the sum is within 5% of 50000
        if abs(token_sum - 50000) / 50000 <= 0.05:
            os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY_2', os.getenv('OPENAI_API_KEY'))
        elif os.environ['OPENAI_API_KEY'] == os.environ.get('OPENAI_API_KEY_2', ''):
            os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY_3')
        elif os.environ['OPENAI_API_KEY'] == os.environ.get('OPENAI_API_KEY_3', ''):
            os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

    for output_file, history_file in zip(output_files, history_files):
        if os.path.exists(output_file):
            if os.path.exists(history_file):
                # If history file already exists, append the output file content to it
                with open(output_file, 'r') as out_f, open(history_file, 'r+') as hist_f:
                    if '.jsonl' in output_file:
                        output_data = [json.loads(line) for line in out_f]
                        history_data = [json.loads(line) for line in hist_f]
                        history_data.extend(output_data)  # Extend the existing data with the new one
                        hist_f.seek(0)  # Reset the file cursor to the beginning
                        json.dump(history_data, hist_f)
                        hist_f.truncate()  # Remove everything after the current file cursor (i.e., after the newly dumped data)
                    else:
                        reader = csv.reader(out_f)
                        writer = csv.writer(hist_f)
                        writer.writerows(reader)
            else:
                # If history file doesn't exist, rename output file to history file.
                os.rename(output_file, history_file)
            # Clear the output file after moving its data to history file.
            open(output_file, 'w').close()
            # Print out the contents of the output file to check if it's empty
            with open(output_file, 'r') as file:
                contents = file.read()
                if contents:
                    print(f'Contents of {output_file}', contents)
                else:
                    print(f'{output_file} is empty.')

    with open('test.txt', 'r') as file:
        input_collection = file.read()
    input = define_genre_and_create_variables_from_df(input_collection)
    print(input)
    data = convert_csv_to_array(input, "data.py")
    requests_filepath = 'data.py'
    generate_chat_completion_requests(requests_filepath, data, input_collection, model_name="gpt-4-1106-preview")

    first_loop = True
    while True:
        if not first_loop:
            # Clear the output files after the first loop
            open('output.csv', 'w').close()
            open('output.jsonl', 'w').close()
        first_loop = False

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
        df = save_generated_data_to_csv('output.jsonl')
        output_text = get_text_value(df)
        hallucination_check = compare_facts(input_collection, output_text)
        print(hallucination_check)

        if "Problem" in hallucination_check:
            # Restart asyncio.run
            continue

        elif "Fehlende Details" in hallucination_check:
            with open('output.txt', 'w') as file:
                file.write(hallucination_check)
            break
        else:
            # Rerun hallucination_check
            pass