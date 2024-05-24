# Precision & Recall for LLM

# Install

```shell
git clone git@github.com:AlexVerine/pr-4-llm.git
cd pr-4-llm/
pip install -e ./
```

# Structure
```
├── pr_llm
│   ├── data
│   │   ├── [data processing]
│   ├── evaluate
│   │   ├── [evaluation for PR]
│   ├── mauve
│   │   ├── [clone of MAUVE repo]
│   ├── PR
│   │   ├── [functions for computing our Precision and Recall metrics]
│   └── utils.py >> general util functions
├── pyproject.toml >> for dependencies
├── README.md

```
Code is tailored for being run on a Slurm cluster.

# Env

For using the same scripts but with different file paths we use `python-dotenv`. This implies that you should create a `.env` file at the root of the folder, with some environment variables

- `DATA_PATH`: path to the reference datasets in a HF format.
- `RESULT_PATH`: path to save the results.
- `CHECKPOINT_PATH`: path from where to load the models.
- `TMP_ENV`: name of the env variables (defined at runtime) that will points to a tmp path for storing distributed generations (depend on the jobs).

# Data

Each dataset can be downloaded and preprocess using the correspinding script in `pr_llm/data`.

# Generations

Generation scripts are available in `pr_llm/generation`, with a script launcher for each task.

# Evaluation

Evaluation scripts are `pr_llm/evaluation`, as well with the scripts for launching the evaluation.

Evaluation for MAUVE, SelfBLEU and Distinct-N is from [mauve](https://github.com/krishnap25/mauve) and [mauve-exp](https://github.com/krishnap25/mauve-experiments).
