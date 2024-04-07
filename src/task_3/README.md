# Task 3 - Top countries by negative sentiment
This task aggregates the top $N$ countries by their negative sentiment (or any sentiment in $\{\text{Positive}, \text{Negative}, \text{Neutral}\}$)

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
## Preprocessing
For UNIX, run:
```shell
PYTHONPATH=./src python -m task_3.task_3_preprocessing --data_dir=$(pwd)/data --output_dir=$(pwd)/data/task_3 --processed_csv_dir=$(pwd)/data/processed/task_3
```
For Windows, run:
```shell
$env:PYTHONPATH = './src'; python -m task_3.task_3_preprocessing --data_dir=$(pwd)/data --output_dir=$(pwd)/data/task_3 --processed_csv_dir=$(pwd)/data/processed/task_3
```
If dependencies are installed with Pipenv, setup the `.env` file with the environment variables set. An example is given in [example.env](/example.env)
```shell
pipenv run python -m task_3.task_3_preprocessing --data_dir=$(pwd)/data --output_dir=$(pwd)/data/task_3 --processed_csv_dir=$(pwd)/data/processed/task_3
```

## Main MapReduce Task
Run
```shell
java -jar Task3.jar data/task_3/<output_of_preprocessing>.csv output data/ISO-3166-alpha3.tsv 5 negative
```
Where the arguments are
- Preprocessed input CSV
- Output directory
- Path to ISO-3166 Country Codes TSV
- Top $N$ countries to truncate to
- Sentiment to count