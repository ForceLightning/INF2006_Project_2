# INF2006 Assignment 2

# Requirements
- Python >= 3.10, < 3.12
- Hadoop 3.3.5

# Installation
Install the Python dependencies with:
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
For all python programs, they can be run as modules. Ensure that they are being run from the project root directory itself.

UNIX:
```shell
PYTHONPATH=./src python -m task_<n>.<filename> --args...
```

Windows:
```shell
$env:PYTHONPATH = './src'; python -m task_<n>.<filename> --args...
```

See the individual readmes in the task folders for further instructions.
- [Task 1](/src/task_1/README.md)
- [Task 2](/src/task_2/README.md)
- [Task 3](/src/task_3/README.md)
- [Task 4](/src/task_4/README.md)
- [Task 5](/src/task_5/README.md)
- [Task 6](/src/task_6/README.md)
