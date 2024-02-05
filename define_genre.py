from base_functions import linear_chat_request
import logging
from typing import Callable

def define_genre_and_create_variables_from_df(
        input_string: str,
        compile_request_func: Callable[[str], str] = linear_chat_request
) -> str:
    """
    Defines the genre and creates variables from the given input DataFrame.

    :param input_string: The input string to be used as input to compile request function.
    :param compile_request_func: Function to compile a chat request.
    :return: The name of the prompt file to use.
    """
    logging.info('define_genre function started.')

    prompt = _get_prompt_with_input_string(input_string)

    try:
        response = compile_request_func(prompt)
        logging.info('Processing variable: %s' % response.choices[0].message.content.strip())
        print(_get_response_file_name(response))
        return _get_response_file_name(response)
    except FileNotFoundError:
        logging.error('File %s not found.' % _get_response_file_name(response))
    except Exception as e:
        logging.error('Failed to create response for variable. Error: %s' % e)

def _get_prompt_with_input_string(input_string: str) -> str:
    """
    :param input_string: String representing the input to be appended to the prompt.
    :return: A string representing the prompt with the input string appended.
    """
    with open('Prompts/prompt_genre.txt', 'r') as file:
        prompt_txt = file.read()
    return f"{prompt_txt}\n{input_string}"

def _get_response_file_name(response) -> str:
    """
    :param response: The response object that contains the choices and message content.
    :return: The file name for the response, with '.csv' appended.
    """
    return 'Prompts/' + response.choices[0].message.content.strip() + '.csv'