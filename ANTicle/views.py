from django.shortcuts import render
from .forms import InputDataForm  # Ensure to have correct import path
from django.http import HttpResponse, JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import os
import pandas as pd
import json
import asyncio
from .utils.output_management import save_generated_data_from_async_req, clear_files, csv_to_json
from .utils.history_management import check_output_and_history_files
from .utils.hallucination_check import check_hallucinations
from .utils.async_requests import process_api_requests_from_file
from .utils.input_management import read_and_concatenate_files, process_data, format_form
from .utils.getting_started import setup_config

@method_decorator(csrf_exempt, name='dispatch')
class ANT(View):
    """
    ANT class

    The ANT class is a view class that handles HTTP requests and responses. It is responsible
    for rendering the index.html template with an empty form for GET requests, and processing
    and returning the output data for POST requests. It also contains various auxiliary methods
    that are used for data processing and file handling.

    Methods:
        - get(request, *args, **kwargs):
            Handles the GET request and renders the index.html template with an empty form.
            :param request: HttpRequest object
            :param args: Additional positional arguments
            :param kwargs: Additional keyword arguments
            :return: Rendered template with the empty form

        - post(request, *args, **kwargs):
            Handles the POST request and processes the input data form. If the form is valid,
            it retrieves the cleaned data. Otherwise, it returns a JSON response with an error
            message indicating that the form data is invalid. It then proceeds to setup the
            configuration, check the output and history files, process the input data, and
            make API requests. If any hallucinations are found in the generated output,
            the process is repeated after clearing the files. Finally, it converts the output
            data into CSV format and returns a JSON response with the converted data.
            :param request: HttpRequest object
            :param args: Additional positional arguments
            :param kwargs: Additional keyword arguments
            :return: JSON response with the converted output data
    """
    def get(self, request, *args, **kwargs):
        form = InputDataForm()
        return render(request, "index.html", {"form": form})

    def post(self, request, *args, **kwargs):
        print('noch ein Test 6')
        """
        :param request: The HTTP request object
        :param args: Optional positional arguments
        :param kwargs: Optional keyword arguments
        :return: The HTTP response with output data in JSON format
        """
        with open('form_data.json', 'w') as outfile:
            json.dump(request.POST, outfile, ensure_ascii=False)
        input_data = None
        if request.POST:
            input_data = format_form('form_data.json')
            print(input_data)


        output_files = ['temp/output.jsonl', 'Output_data/output.csv']
        history_files = ['Logging_Files/history.jsonl', 'Logging_Files/history.csv', ]
        setup_config()
        print('config done')
        check_output_and_history_files(output_files, history_files)

        if input_data:  # check if input_data is not None (if None it means it is not initialised)
            input_collection = input_data
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
