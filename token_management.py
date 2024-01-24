import datetime
import os
import csv

#Total Token Managment
def record_token_usage(total_tokens):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = 'tokens_log.csv'
    history_filename = 'token_count_history.csv'
    headers = ['time', 'tokens']
    if os.path.isfile(filename):
        with open(filename, 'r') as file:
            last_line = file.readlines()[-1]
        last_timestamp = datetime.datetime.strptime(last_line.split(",")[0], "%Y-%m-%d %H:%M:%S")
        if last_timestamp.date() != datetime.date.today():
            if os.path.isfile(history_filename):
                with open(history_filename, 'a') as history_file:
                    with open(filename, 'r') as f:
                        history_file.write(f.read())
            else:
                os.rename(filename, history_filename)

            with open(filename, 'w', newline='') as tokens_log:
                writer = csv.writer(tokens_log)
                writer.writerow(headers)
    with open(filename, 'a', newline='') as tokens_log:
        writer = csv.writer(tokens_log)
        writer.writerow([timestamp, total_tokens])

def validate_and_swap_api_key(token_sum, max_api_usage = 500000):
    """

    Validate and Swap API Key

    Parameters:
    :param token_sum: The total number of tokens used in the API call.
    :param max_api_usage: The maximum number of tokens allowed for API usage. Default value is 500000.

    Returns:
    :return: None

    """
    api_key = os.getenv('OPENAI_API_KEY')  # default value
    usage_ratio = abs(token_sum - max_api_usage) / max_api_usage
    if usage_ratio <= 0.05:
        api_key = os.environ.get('OPENAI_API_KEY_2', os.getenv('OPENAI_API_KEY'))
    elif os.environ['OPENAI_API_KEY'] == os.environ.get('OPENAI_API_KEY_2', ''):
        api_key = os.getenv('OPENAI_API_KEY_3')
    elif os.environ['OPENAI_API_KEY'] == os.environ.get('OPENAI_API_KEY_3', ''):
        api_key = os.getenv('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = api_key

#Async Token Mangement
def compute_chat_tokens(request_json, encoding, completion_tokens):
    """
    Calculate the number of tokens in a chat.

    :param request_json: The JSON object containing the chat messages.
    :type request_json: dict
    :param encoding: The encoding used for tokenization.
    :type encoding: str
    :param completion_tokens: The number of completion tokens.
    :type completion_tokens: int
    :return: The total number of tokens in the chat plus completion tokens.
    :rtype: int
    """
    num_tokens = 0
    for message in request_json["messages"]:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens -= 1
    num_tokens += 2
    return num_tokens + completion_tokens

def compute_prompt_tokens(request_json, encoding, completion_tokens):
    """Computes the total number of tokens in the prompt.

    :param request_json: The JSON object containing the completion request.
    :type request_json: dict
    :param encoding: The encoding to be used.
    :type encoding: str
    :param completion_tokens: The number of completion tokens.
    :type completion_tokens: int
    :return: The total number of tokens in the prompt.
    :rtype: int
    :raises TypeError: If the "prompt" field in the completion request is not a string or a list of strings.
    """
    prompt = request_json["prompt"]
    if isinstance(prompt, str):  # single prompt
        prompt_tokens = len(encoding.encode(prompt))
        num_tokens = prompt_tokens + completion_tokens
        return num_tokens
    elif isinstance(prompt, list):
        prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
        num_tokens = prompt_tokens + completion_tokens * len(prompt)
        return num_tokens
    else:
        raise TypeError(
            'Expecting either string or list of strings for "prompt" field in completion request'
        )
