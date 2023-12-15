import pandas as pd
from openai import OpenAI
import os
import logging

# Set up logging
logging.basicConfig(filename='article.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def define_genre_and_create_variables_from_df(input_collection, model="gpt-3.5-turbo-1106", max_tokens=1000):
    logging.info('define_genre function called.')
    try:
        client = OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY'),
        )
        logging.info('OpenAI client initialized.')
        print("OpenAI client initialized.")
    except Exception as e:
        logging.error('Failed to initialize OpenAI client: %s' % e)
        print('Failed to initialize OpenAI client:', e)
        return

    with open('prompt.txt', 'r') as file:
        prompt_txt = file.read()
    prompt = f"{prompt_txt}\n{input_collection['input']}"
    print(prompt)

    try:
        response = client.chat.completions.create(model=model,
                                                  messages=[{"role": "system", "content": "You are a journalist."},
                                                            {"role": "user", "content": prompt}],
                                                  max_tokens=max_tokens)
        print(response)

        file_name = response.choices[0].message.content.strip() + '.csv'
        var_name = file_name.split(".")[0]  # removing .csv to get var_name
        logging.info('Processing variable: %s' % var_name)
        print('Processing variable:', var_name)

        try:
            df = pd.read_csv(file_name)
            logging.info('Read csv file: %s' % file_name)
            print('Read csv file:', file_name)
            return df
        except FileNotFoundError:
            logging.error('File %s was not found.' % file_name)
            print('FileNotFoundError:', file_name)
    except Exception as e:
        logging.error('Failed to create response for variable: %s. Error: %s' % (var_name, e))
        print('Failed to create response for variable:', var_name)


def query_openai_chat_for_each_variable(df, model="gpt-4-1106-preview", max_tokens=1000):
    try:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        logging.info('OpenAI client initialized for chat queries.')
        print("OpenAI client initialized for chat queries.")
    except Exception as e:
        logging.error('Failed to initialize OpenAI client: %s' % e)
        print('Failed to initialize OpenAI client:', e)
        return

    for index, (var_name, prompt) in df.iterrows():
        logging.info('Processing variable: %s' % var_name)
        print('Processing variable:', var_name)
        try:
            response = client.chat.completions.create(model=model, messages=[
                {"role": "system", "content": f"You are a helpful assistant. The topic is {var_name}."},
                {"role": "user", "content": prompt}], max_tokens=max_tokens)
            print(f"Response for variable '{var_name}': {response.choices[0].message.content}")
        except Exception as e:
            logging.error('Error processing variable: %s. Error: %s' % (var_name, e))
            print('Error processing variable:', var_name)


def test_workflow():
    # Read the text file
    with open('test.txt', 'r') as file:
        text = file.read()
    print('Read "test.txt" content.')

    # Use the content as the input_collection with a single key "input"
    input_collection = {"input": text}

    # Call define_genre_and_create_variables_from_df function
    print('Calling define_genre_and_create_variables_from_df().')
    df = define_genre_and_create_variables_from_df(input_collection)

    # Call query_openai_chat_for_each_variable function
    print('Calling query_openai_chat_for_each_variable().')
    query_openai_chat_for_each_variable(df)


# Call the test_workflow() function
print("Starting workflow.")
test_workflow()
print("Workflow complete.")