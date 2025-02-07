# Task 1 - Preprocessing
This task preprocesses the dataset and combines it into a single output CSV.

# Requirements
- Python >= 3.10, < 3.12
- Hadoop 3.3.5

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
PYTHONPATH=./src python -m task_1.Task1 --data_dir=$(pwd)/data
```
For Windows, run:
```shell
$env:PYTHONPATH = './src'; python -m task_1.Task1 --data_dir=$(pwd)/data
```
If dependencies are installed with Pipenv, setup the `.env` file with the environment variables set. An example is given in [example.env](/example.env)
```shell
pipenv run python -m task_1.Task1 --data_dir=$(pwd)/data
```