from django.shortcuts import render
from .forms import InputDataForm  # Ensure to have correct import path
from django.http import HttpResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import os
import pandas as pd
import asyncio
from .utils.output_management import save_generated_data_from_async_req, clear_files, csv_to_json
from .utils.history_management import check_output_and_history_files
from .utils.hallucination_check import check_hallucinations
from .utils.async_requests import process_api_requests_from_file
from .utils.input_management import read_and_concatenate_files, process_data
from .utils.getting_started import setup_config

@method_decorator(csrf_exempt, name='dispatch')
class ANT(View):
    def get(self, request, *args, **kwargs):
        form = InputDataForm()
        return render(request, "index.html", {"form": form})

    def post(self, request, *args, **kwargs):
        form = InputDataForm(request.POST)
        if form.is_valid():
            # Process your form data here
            input_data = form.cleaned_data
            # Use the input_data in subsequent operations

        output_files = ['temp/output.jsonl', 'Output_data/output.csv']
        history_files = ['Logging_Files/history.jsonl', 'Logging_Files/history.csv', ]
        setup_config()
        print('config done')
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
        output_data = csv_to_json(request)
        print(HttpResponse(output_data))
        return HttpResponse(output_data, content_type='application/json')
