import os
import subprocess
import sys

def execute_install_and_run():
    install_python_packages()
    update_environment_variables()
    print_environment_variables()

def install_python_packages():
    with open('requirements.txt', 'r') as file:
        packages = file.readlines()
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def update_environment_variables():
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
    for var, value in os.environ.items():
        print(f"{var}: {value}")

def check_install_count():
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