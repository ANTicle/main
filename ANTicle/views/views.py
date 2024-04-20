from django.shortcuts import render
from ANTicle.forms import InputDataForm
from django.http import HttpResponse, JsonResponse
from django.views import View
import csv
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import os
import pandas as pd
import json
from collections import defaultdict
import asyncio
from ANTicle.utils.output_management import save_generated_data_from_async_req, clear_files, csv_to_json, \
    add_additional_content
from ANTicle.utils.history_management import check_output_and_history_files
from ANTicle.utils.hallucination_check import check_hallucinations
from ANTicle.utils.async_requests import process_api_requests_from_file
from ANTicle.utils.input_management import read_and_concatenate_files, process_data, format_form, split_json_string
from ANTicle.utils.getting_started import setup_config


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
        # form.fields['thema'].widget = form.thema(attrs={'style': 'display: block'})

        return render(request, "index.html", {"form": form, "accent_color": "#41efb4"})
        #To Do: hardcode farbe in variable ändern
    def post(self, request, *args, **kwargs):
        """
        :param request: The HTTP request object
        :param args: Optional positional arguments
        :param kwargs: Optional keyword arguments
        :return: The HTTP response with output data in JSON format
        """
        with open('./form_data.json', 'w') as outfile:
            json.dump(request.POST, outfile, ensure_ascii=False)
        input_data = None
        if request.POST:
            with open('./form_data.json') as json_file:
                json_data = json.load(json_file)
                if isinstance(json_data, str):
                    # Convert to a dictionary
                    try:
                        json_data = json.loads(json_data)
                    except json.JSONDecodeError:
                        print('Error: JSON string could not be parsed into a dictionary.')
            split_json_string(json_data)
            input_data = format_form('quelle.json')
            print(input_data)

        output_files = ['./temp/output.jsonl', './Output_data/output.csv']
        history_files = ['./Logging_Files/history.jsonl', './Logging_Files/history.csv']
        setup_config()
        print('config done')
        check_output_and_history_files(output_files, history_files)

        input_collection = None

        if input_data:  # check if input_data is not None (if None it means it is not initialised)
            input_collection = input_data
            print(input_collection)
        if input_collection is not None:
            requests_filepath, data = process_data(input_collection)
            first_loop = True
            while True:
                if not first_loop:
                    clear_files()
                first_loop = False
                asyncio.run(
                    process_api_requests_from_file(
                        requests_filepath=requests_filepath,
                        save_filepath='./temp/output.jsonl',
                        request_url="https://api.openai.com/v1/chat/completions",
                        api_key=os.getenv("OPENAI_API_KEY"),
                        max_requests_per_minute=int(os.getenv("max_requests_per_minute")),
                        max_tokens_per_minute=int(os.getenv("max_tokens_per_minute")),
                        token_encoding_name="cl100k_base",
                        max_attempts=int(5),
                        logging_level=int(20),
                    )
                )
                async_df = save_generated_data_from_async_req('./temp/output.jsonl')  # save async request output
                if not check_hallucinations(input_collection, async_df):
                    break
        add_additional_content()
        data_dict = defaultdict(dict)
        with open('./Output_data/output.csv', 'r') as f:
            reader = csv.reader(f)
            headers = next(reader, None)  # get headers
            for row in reader:
                for header, field in zip(headers, row):
                    if header == "Text:":
                        if 'Text' in data_dict[row[0]]:
                            data_dict[row[0]]['Text'] += field
                        else:
                            data_dict[row[0]]['Text'] = field  # normal row if 'Text'
                    elif header not in ['Zusatz']:
                        sub_keys = field.split("\n")  # split on linebreak
                        for idx, key in enumerate(sub_keys):
                            if len(key) > 20:
                                data_dict[row[0]]["sub_key_{:02d}".format(idx)] = key  # store if len > 20

        # remove fields shorter than 20 characters from data_dict
        for key in list(data_dict.keys()):
            for sub_key in list(data_dict[key].keys()):
                if len(data_dict[key][sub_key]) <= 20:
                    del data_dict[key][sub_key]

        final_dict = dict(data_dict)  # convert defaultdict back to dict
        print('test: ' + json.dumps(final_dict))
        return JsonResponse(final_dict)



