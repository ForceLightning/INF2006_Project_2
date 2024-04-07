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

# Building Documentation
```shell
export PYTHONPATH=./src
python -m sphinx-apidoc -o docs/source ./src/
python -m sphinx-build -b html docs/source docs/build/html

cd src/task_2/
mvn javadoc:javadoc

cd ../task_3/
mvn javadoc:javadoc
```

The documentation for tasks 1, 3, 4, 5, and the utility module will be found in `docs/build/html/`, while task 2's and 3's Javadoc pages will be found in `src/task_[n]/target/site/apidocs/`
