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
