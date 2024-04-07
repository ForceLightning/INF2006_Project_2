# Task 2 - Top countries complaints
This task aggregates the top 5 countries by their complaint reasons.

# Requirements
- Python >= 3.10, < 3.12
- Hadoop 3.3.5

# Usage
## Preprocessing
For UNIX, run:
```shell
PYTHONPATH=./src python -m utils.util --data_dir=$(pwd)/data --output_dir=$(pwd)/data/processed/task_2 --remove_newline
```
For Windows, run:
```shell
$env:PYTHONPATH = './src'; python -m utils.util --data_dir=$(pwd)/data --output_dir=$(pwd)/data/processed/task_2 --remove_newline
```

## Main MapReduce Task
Run
```shell
java -jar Task2.jar data/processed/task_2/<output_of_preprocessing>.csv output/task_3
```
Where the arguments are:
- Preprocessed input CSV.
- Output directory (must be empty)