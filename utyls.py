import csv

def append_to_log(filename, timestamp, total_tokens):
    with open(filename, mode='a', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([timestamp, total_tokens])
