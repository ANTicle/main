import os
import subprocess
import sys

import pandas as pd

from .token_management import validate_and_swap_api_key


def execute_install_and_run():
    """
    Execute the installation and running process. This method performs the following steps:

    1. Install the required Python packages.
    2. Update the environment variables.
    3. Print the updated environment variables.

    :return: None.
    """
    install_python_packages()
    update_environment_variables()
    print_environment_variables()

def install_python_packages():
    """
    Install Python packages listed in the requirements.txt file.

    :return: None
    """
    with open('requirements.txt', 'r') as file:
        packages = file.readlines()
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def update_environment_variables():
    """
    Updates environment variables based on the settings in the 'settings.txt' file.

    :return: None
    """
    defaults = {
        "Token_Daily_Max": "500000",
        "max_requests_per_minute": "90000",
        "max_tokens_per_minute": "170000",
    }
    # Read settings from file
    with open('settings.txt', 'r') as file:
        config_values = file.read()

    config_lines = config_values.split("\n")
    for line in config_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        key, config_value = line.split("=")
        config_value = config_value if config_value else defaults.get(key, None)
        print(config_value)
        if config_value is None:
            print(f"Warning: Ignored the setting {key} due to None value.")
            continue

        if key == "OPENAI_API_KEY" and not config_value and "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY is empty and it must be set.")

        if not os.environ.get(key):
            os.environ[key] = config_value

def print_environment_variables():
    """
    Prints the environment variables and their values.

    :return: None
    """
    for var, value in os.environ.items():
        print(f"{var}: {value}")

def check_install_count():
    """

    Check Install Count

    This function reads the install count from a file and returns it. If the file does not exist or if the value in the file is not a valid integer, it resets the count to 0.

    :return: The install count as an integer.

    """
    try:
        with open("temp/install_count.txt", "r") as f:
            try:
                install = int(f.read())
            except ValueError:
                # Non-integer value in the file, resetting count to 0
                install = 0
    except FileNotFoundError:
        # File does not exist yet, this must be the first run
        install = 0
    finally:
        return install
def update_install_count():
    # Increase counter
    install = check_install_count()
    if install == 0:
        execute_install_and_run()
        install += 1
    print(install)
    # Write updated counter back into the file
    with open("temp/install_count.txt", "w") as f:
        f.write(str(install))

    return install


def setup_config():
    update_install_count()
    update_environment_variables()
    if os.path.exists('Logging_Files/tokens_log.csv'):
        async_df = pd.read_csv('Logging_Files/tokens_log.csv')
        token_sum = async_df['tokens'].sum()
        print(f"Token sum: {token_sum}")
        validate_and_swap_api_key(token_sum)
