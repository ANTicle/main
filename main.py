import os
import asyncio
import pandas as pd
import datetime
from base_functions import convert_csv_to_array
from token_management import validate_and_swap_api_key
from output_management import rename_and_clear_output_file, save_generated_data_from_async_req
from history_management import append_output_to_history
from define_genre import define_genre_and_create_variables_from_df
from hallucination_check import get_text_value, compare_facts
from async_requests import process_api_requests_from_file, generate_chat_completion_requests
from input_management import read_and_concatenate_files
from setup import update_install_count, update_environment_variables

if __name__ == "__main__":
    output_files = ['output.jsonl', 'output.csv']
    history_files = ['history.jsonl', 'history.csv']
    update_install_count()
    update_environment_variables()
    if os.path.exists('Logging_Files/tokens_log.csv'):
        async_df = pd.read_csv('Logging_Files/tokens_log.csv')
        token_sum = async_df['tokens'].sum()
        print(token_sum)
        validate_and_swap_api_key(token_sum)

    for output_file, history_file in zip(output_files, history_files):
        if os.path.exists(output_file) and os.path.exists(history_file):
            append_output_to_history(output_file, history_file)
        else:
            rename_and_clear_output_file(output_file, history_file)
        print('output cleanup done')

    input_collection = read_and_concatenate_files('Input_data') #update with frontened integration
    print(input_collection)
    prompt_b_o_genre = define_genre_and_create_variables_from_df(input_collection)
    print('input: ' + prompt_b_o_genre + ' stop input')
    data = convert_csv_to_array(prompt_b_o_genre, 'temp/data.py')
    requests_filepath = 'temp/data.py'
    generate_chat_completion_requests(requests_filepath, data, input_collection, model_name="gpt-4-1106-preview")
    first_loop = True
    while True:
        if not first_loop:
            open('Output_data/output.csv', 'w').close()
            open('temp/output.jsonl', 'w').close()
        first_loop = False
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=requests_filepath,
                save_filepath='temp/output.jsonl',
                request_url="https://api.openai.com/v1/chat/completions",
                api_key=os.getenv("OPENAI_API_KEY"),
                max_requests_per_minute= int(os.getenv("max_requests_per_minute")),
                max_tokens_per_minute=int(os.getenv("max_tokens_per_minute")),
                token_encoding_name="cl100k_base",
                max_attempts=int(5),
                logging_level=int(20),
            )
        )
        async_df = save_generated_data_from_async_req('temp/output.jsonl') #save async request output
        output_text = get_text_value(async_df) #sperate text for hallucination check
        hallucination_check = compare_facts(input_collection, output_text)
        print(hallucination_check)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if "Es gibt folgendes Problem:" in hallucination_check:
            with open('Output_data/hallucinations.txt', 'w') as file:
                file.write(hallucination_check + timestamp)
            continue
        elif "Fehlende Details" in hallucination_check:
            with open('Output_data/Fehlende_Details.txt', 'w') as file:
                file.write(hallucination_check)
            break
        else:
            pass

