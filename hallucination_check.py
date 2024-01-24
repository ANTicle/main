from utyls import compile_chat_request
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
    compare_prompt = '''Agiere als Journalist. Du bekommst im Folgenden zwei Informationsquellen zu einem Thema. Lies sie und überprüfe, ob sich Fakten in den Texten widersprechen. Überprüfe besonders, ob Zahlen übereinstimmen. Wenn sich Fakten wiedersprechen ist das ein Problem. 
    Achte außerdem darauf das der zweite text, keine Fakten enthält, welche so nicht auch in dem ersten Text vorhanden sind. Wenn du in dem zweiten Text Fakten findest, welche nicht in dem ersten enthalten sind ist das ein Problem.
    Der zweite Text darf weniger details einhalten als der erste. Das ist kein Problem. 
    Wenn ein signifikantes Problem vorhanden ist, antworte mit dem Wort "Es gibt folgendes Problem:" gefolgt von einer Beschreibung des Problems. Wenn kein Problem vorhanden ist, sondern nur fehelende Details antworte mit "Fehlende Details:" gefolgt von einer Liste der fehlenden Details.''' + '\n' + '\n' + "Erster Text" + '\n' + str(input) + '\n' + '\n' + "Zweiter Text" + '\n' + str(output)
    compare = compile_chat_request(compare_prompt)
    choice = compare.choices[0].message.content.strip()  # Retrieve the first Choice object

    return choice
