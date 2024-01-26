import os
import asyncio
from output_management import save_generated_data_from_async_req, clear_files
from history_management import check_output_and_history_files
from hallucination_check import check_hallucinations
from async_requests import process_api_requests_from_file
from input_management import read_and_concatenate_files, process_data
from setup import setup_config

if __name__ == "__main__":
    output_files = ['temp/output.jsonl', 'Output_data/output.csv']
    history_files = ['Logging_Files/history.jsonl', 'Logging_Files/history.csv', ]

    setup_config()
    check_output_and_history_files(output_files, history_files)

    input_collection = read_and_concatenate_files('Input_data')  # update with frontened integration
    print(input_collection)

    requests_filepath, data = process_data(input_collection)

    first_loop = True
    while True:
        if not first_loop:
            clear_files()
        first_loop = False
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=requests_filepath,
                save_filepath='temp/output.jsonl',
                request_url="https://api.openai.com/v1/chat/completions",
                api_key=os.getenv("OPENAI_API_KEY"),
                max_requests_per_minute=int(os.getenv("max_requests_per_minute")),
                max_tokens_per_minute=int(os.getenv("max_tokens_per_minute")),
                token_encoding_name="cl100k_base",
                max_attempts=int(5),
                logging_level=int(20),
            )
        )
        async_df = save_generated_data_from_async_req('temp/output.jsonl')  # save async request output
        if not check_hallucinations(input_collection, async_df):
            break