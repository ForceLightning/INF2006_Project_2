# Task 5 - Sentiment Analysis

# Requirements
- Python >= 3.10, < 3.12

# Installation
Install the Python dependencies from the project root folder with:
```shell
pip install -r requirements.txt
```
or
```shell
pipenv install -r requirements.txt
```
or with the Pipfile
```shell
pipenv install
```

# Usage
For UNIX, run:
```shell
PYTHONPATH=./src OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python -m task_5.Task5 --data_dir=$(pwd)/data
```

For Windows, run:
```shell
$env:PYTHONPATH='./src'; $env:OBJC_DISABLE_INITIALIZE_FORK_SAFETY='YES'; python -m task_5.Task5 --data_dir=$(pwd)/data
```

If using Pipenv and a `.env` file is set with the environment variables above, run:
```shell
pipenv run python -m task_5.Task5 --data-dir=$(pwd)/data
```