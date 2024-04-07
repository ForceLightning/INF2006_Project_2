# Task 4 - To calculate the mean and median values of the trust points for each channel.
This task utilzes Spark to process the data, calculates the mean and median trust points of each channel and outputs the result to a CSV file.

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
PYTHONPATH=./src python -m task_4.Task4 
```
For Windows, run:
```shell
$env:PYTHONPATH = './src'; python -m task_4.Task4
```
If dependencies are installed with Pipenv, setup the `.env` file with the environment variables set. An example is given in [example.env](/example.env)
```shell
pipenv run python -m task_4.Task4
```