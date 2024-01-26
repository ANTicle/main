import datetime

from base_functions import linear_chat_request


def get_text_value(df):
    """
    Check if the title "Text" exists in the DataFrame.

    :param df: A pandas DataFrame containing the data.
    :return: The extracted text value corresponding to the title "Text" if found, otherwise None.
    """
    # Check if the title "Text" exists in the DataFrame
    if 'Text' in df['Title'].values:
        # Extract the Generated Output corresponding to the Title "Text"
        output = df.loc[df['Title'] == 'Text', 'Generated Output'].values[0]
        return str(output)
    else:
        print("Title 'Text' not found in the DataFrame.")
        return None

def compare_facts(input, output):
    """
    :param input: The first text containing facts to be compared
    :param output: The second text containing facts to be compared
    :return: The user's choice after comparing the facts

    This method takes in two texts, `input` and `output`, and compares the facts in these texts. The user is prompted to choose whether there are conflicting facts or missing details between
    * the texts. The method returns the user's choice as a string. The method makes use of the `compile_chat_request` function to display the texts to the user and retrieve their choice
    *.
    """
    with open('Prompts/compare_prompt.txt', 'r') as f:
        prompt = f.read()
    compare_prompt = prompt + '\n' + '\n' + "Sammlung an Quellen" + '\n' + str(input) + '\n' + '\n' + "Text" + '\n' + str(output)
    compare = linear_chat_request(compare_prompt)
    choice = compare.choices[0].message.content.strip()  # Retrieve the first Choice object

    return choice


def check_hallucinations(input_collection, async_df):
    """
    Check for hallucinations in the given input collection and async_df.

    :param input_collection: The input collection to compare with the output text.
    :type input_collection: list
    :param async_df: The async DataFrame representing the output text.
    :type async_df: DataFrame
    :return: True if hallucinations are found, False otherwise.
    :rtype: bool
    """
    output_text = get_text_value(async_df)  # separate text for hallucination check
    hallucination_check = compare_facts(input_collection, output_text)
    print(hallucination_check)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "Es gibt folgendes Problem:" in hallucination_check:
        with open('Output_data/hallucinations.txt', 'w') as file:
            file.write(hallucination_check + timestamp)
        return True
    elif "Fehlende Details" in hallucination_check:
        with open('Output_data/Fehlende_Details.txt', 'w') as file:
            file.write(hallucination_check)
        return False
    return True
