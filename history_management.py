import json
import csv
import os
def append_output_to_history(output_file, history_file):
    """
    Append output from 'output_file' to 'history_file'.

    :param output_file: The path to the file containing the output data.
    :param history_file: The path to the file where the output data will be appended.
    :return: None
    """
    with open(output_file, 'r') as out_f, open(history_file, 'r+') as hist_f:
        if '.jsonl' in output_file:
            output_data = [json.loads(line) for line in out_f]
            history_data = [json.loads(line) for line in hist_f]
            history_data.extend(output_data)
            hist_f.seek(0)
            json.dump(history_data, hist_f)
            hist_f.truncate()
        else:
            reader = csv.reader(out_f)
            writer = csv.writer(hist_f)
            writer.writerows(reader)

        # Empty the output file by removing it and recreating it as an empty file
        os.remove(output_file)
        open(output_file, 'w').close()
