import os

#move old output to history
def rename_and_clear_output_file(output_file, history_file):
    """
    Rename and clear the output file.

    :param output_file: The path to the output file.
    :type output_file: str
    :param history_file: The desired name for the renamed output file.
    :type history_file: str
    :return: None
    :rtype: None
    """
    if os.path.exists(output_file):
        os.rename(output_file, history_file)
        open(output_file, 'w').close()
