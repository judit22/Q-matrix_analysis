# NCDM-based Cognitive Diagnosis and Skill Discovery

This repository contains two experimental workflows for working with Neural Cognitive Diagnosis Models (NCDM):

1. **Synthetic data experiments**, where simulated student--item response data are generated and the model is evaluated based on how well it can recover latent student abilities and item--skill parameters.
2. **Real-life data experiments**, where the model is trained on real educational response data and used to infer item--skill relationships, which are then compared to an expert-defined Q-matrix.

The project is intended for research on cognitive diagnosis, Q-matrix validation, skill discovery, and the evaluation of neural-network-based models in educational data mining.

---

## Repository structure

```text
.
├── README.md
├── Real_life_data_notebook.ipynb
├── utils.py
├── example_Q.csv
├── example_dataset.csv
│
└── synthetic_data/
    ├── Synthetic_data_notebook.ipynb
    └── cognitive_simulator/
        ├── __init__.py
        ├── utils.py
        ├── params.py
        └── simulation.py
```

### Main components

| Path | Description |
|---|---|
| `synthetic_data/Synthetic_data_notebook.ipynb` | Notebook for generating synthetic response data and evaluating NCDM parameter recovery. |
| `synthetic_data/cognitive_simulator/` | Local simulator package used to generate synthetic student abilities, item parameters, and response data. |
| `Real_life_data_notebook.ipynb` | Notebook for training NCDM on real-life educational data and discovering stable item--skill relationships. |
| `utils.py` | Helper functions used by the real-life data workflow, including preprocessing, data transformation, model training, and stability analysis. |
| `example_Q.csv` | Example expert-defined Q-matrix / item--skill mapping. |
| `example_dataset.csv` | Example student response dataset used by the real-life workflow. |

---

## What the project does

### 1. Synthetic data workflow

The synthetic data workflow evaluates whether an NCDM model can recover known latent structures from controlled data.

The notebook:

1. Generates synthetic student--exercise response data using the local `cognitive_simulator` package.
2. Converts the generated data into the format required by the `EduCDM` implementation of NCDM.
3. Constructs a Q-matrix from the generated item--skill dependencies.
4. Splits the data into training, validation, and test sets.
5. Trains an NCDM model.
6. Extracts learned student ability embeddings and item difficulty/skill parameters.
7. Compares the recovered parameters with the true synthetic parameters.
8. Reports recovery errors such as mean absolute error and squared error.

This workflow is useful for validating whether the model can recover latent educational parameters when the ground truth is known.

### 2. Real-life data workflow

The real-life workflow applies NCDM to educational response data where the true latent structure is not directly observable.

The notebook:

1. Loads an expert-defined Q-matrix from `example_Q.csv`.
2. Loads student response data from `example_dataset.csv`.
3. Filters and preprocesses the data.
4. Applies train/validation splitting.
5. Trains an NCDM model on student--item interactions.
6. Extracts learned item--skill relevance values from the model.
7. Compares the inferred skill structure with the expert-defined Q-matrix.
8. Repeats training multiple times to identify stable skill predictions.

The final output is a comparison table containing the original expert-defined skills and the stable skills inferred by the model across repeated training runs.

---

## Installation

This repository does not currently include a `requirements.txt` file. The required packages can be installed manually.

### 1. Clone the repository

```bash
git clone https://github.com/judit22/Q-matrix_analysis.git
cd Q-matrix_analysis
```

### 2. Create and activate a virtual environment

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

Or using Conda:

```bash
conda create -n ncdm-skill-discovery python=3.10
conda activate ncdm-skill-discovery
```

### 3. Install dependencies

```bash
pip install numpy pandas scikit-learn torch matplotlib jupyter EduCDM EduData
```

The notebooks use the following Python packages:

| Package | Used for |
|---|---|
| `numpy` | Numerical computation and array operations. |
| `pandas` | Loading, transforming, and storing tabular data. |
| `torch` | Neural network training and tensor operations. |
| `scikit-learn` | Train/test/validation splitting. |
| `EduCDM` | NCDM implementation. |
| `EduData` | Supporting package commonly used with `EduCDM`. |
| `matplotlib` | Optional plotting and result visualization. |
| `jupyter` | Running the notebooks. |

If you only run the real-life notebook and all required model utilities are already included in `utils.py`, the core requirements are typically:

```bash
pip install numpy pandas torch scikit-learn jupyter EduCDM
```

---

## How to run the notebooks

Start Jupyter from the repository root:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

### Running the synthetic data notebook

Open:

```text
synthetic_data/Synthetic_data_notebook.ipynb
```

Because this notebook imports the local simulator package:

```python
from cognitive_simulator import *
```

it should be run from inside the `synthetic_data/` directory, where the `cognitive_simulator` folder is located.

The notebook defines experiment ranges such as:

```python
ABILITIES_RANGE = range(5, 11)
STUDENTS_RANGE = range(500, 1600, 100)
EXERCISES_RANGE = range(50, 110, 10)
```

These settings control the number of simulated skills, students, and exercises. The notebook then iterates over all combinations and stores aggregated results in a CSV file, such as:

```text
df_students_items.csv
```

Generated synthetic datasets may also be saved as CSV files with names containing the number of students, abilities, and items.

### Running the real-life data notebook

Open:

```text
Real_life_data_notebook.ipynb
```

This notebook expects the following files to be available in the repository root:

```text
utils.py
example_Q.csv
example_dataset.csv
```

The notebook loads the files using:

```python
df1 = pd.read_csv("example_Q.csv", encoding="utf-8", sep=";")
df2 = pd.read_csv("example_dataset.csv", encoding="utf-8", sep=",")
```

The expected input structure is:

| File | Expected role |
|---|---|
| `example_Q.csv` | Expert-defined Q-matrix or item--skill mapping. |
| `example_dataset.csv` | Student response dataset containing user, item, and response information. |

The workflow uses the following main configuration values:

```python
MIN_SOLVED_TASKS = 30
SUCCESS_THRESHOLD = 0
NUM_RUNS = 20
STABILITY_THRESHOLD = 0.3
```

These parameters define the minimum number of completed tasks per student, the binarization threshold, the number of repeated model trainings, and the minimum stability ratio for accepting a predicted skill as stable.

---

## Output of the workflows

### Synthetic data output

The synthetic workflow produces:

- generated synthetic student--item response data,
- generated Q-matrices,
- trained NCDM models,
- recovered student ability estimates,
- recovered item--skill dependency estimates,
- error metrics comparing recovered and true latent parameters,
- aggregated CSV files containing experimental results.

The main goal is to measure whether NCDM can recover known latent ability and item dependency structures from simulated data.

### Real-life data output

The real-life workflow produces a comparison between expert-defined and model-inferred item--skill mappings.

The final table has the following conceptual structure:

| Column | Description |
|---|---|
| `original` | Expert-defined skill assignment from the Q-matrix. |
| `soft_0`, `soft_1`, ..., `soft_n` | Predicted skills from repeated NCDM training runs. |
| `stable_soft` | Skills that appeared in more than the selected stability threshold across repeated runs. |

For example, if `NUM_RUNS = 20` and `STABILITY_THRESHOLD = 0.3`, then a skill is considered stable for an item if it appears in more than 30% of the 20 repeated predictions.

---

## Methodological overview

### Neural Cognitive Diagnosis Model

The repository uses the Neural Cognitive Diagnosis Model (NCDM), a neural-network-based cognitive diagnosis model that learns representations of students, items, and knowledge concepts from response data.

In this project, NCDM is used not only for predicting student performance, but also for analyzing the learned item--skill relevance structure. The learned item difficulty / skill-related parameters are interpreted as indicators of which skills are relevant for each task.

### Stability-based skill selection

Because neural models can be sensitive to random initialization and training variation, the real-life workflow repeats the training process multiple times. Instead of relying on a single model run, the pipeline identifies skills that appear consistently across repeated trainings.

A predicted skill is considered stable if it appears in more than a predefined proportion of runs:

```python
STABILITY_THRESHOLD = 0.3
```

This reduces the effect of random training noise and provides a more robust inferred Q-matrix.

### Synthetic validation

The synthetic workflow is used to validate the model in a controlled setting. Since the true student abilities and item--skill dependencies are known, the model's learned parameters can be compared directly against the ground truth.

This makes it possible to evaluate whether NCDM can recover latent structures before applying the same idea to real-life educational data.

---

## Example usage

### Synthetic experiment

```python
run_test(
    n_students=500,
    n_exercises=50,
    n_abilities=5,
    filename="test_results_500p_5a_50i.csv",
    results_filename="df_students_items.csv"
)
```

This generates a synthetic dataset with 500 students, 50 exercises, and 5 abilities, trains an NCDM model, and appends the recovery results to the selected results file.

### Real-life stability analysis

```python
NUM_RUNS = 20
STABILITY_THRESHOLD = 0.3

compare_df = pd.DataFrame(columns=["original"])
compare_df["original"] = items["knowledge_code"]

all_probs = []

for i in range(NUM_RUNS):
    actual, probs = training(items, results, item_weights)
    all_probs.append(probs)
    compare_df[f"soft_{i}"] = actual["soft"]

freq_soft = compute_skill_frequency(compare_df)
stable_soft = extract_stable_skills(
    freq_soft,
    NUM_RUNS,
    min_ratio=STABILITY_THRESHOLD
)

compare_df["stable_soft"] = stable_soft
```

This repeatedly trains the model and extracts the stable predicted skills for each item.

---

## Notes and assumptions

- The synthetic notebook should be executed from the `synthetic_data/` directory so that the local `cognitive_simulator` package can be imported correctly.
- The real-life notebook should be executed from the repository root so that `utils.py`, `example_Q.csv`, and `example_dataset.csv` are available.
- The repository does not include a `requirements.txt`; dependencies must be installed manually.
- The real-life notebook assumes that the expert Q-matrix and response dataset follow the column structure expected by `utils.py` and the notebook.
- The NCDM model may produce slightly different predictions across runs due to random initialization and stochastic optimization; therefore, repeated training and stability filtering are recommended.

---

## Suggested citation / acknowledgement

If this repository is used in academic work, please cite the repository associated with the project. 
Author: Judit Tevesz & Dr. Bertalan Forstner
Title: A Robust Neural Cognitive Diagnosis Tool for Q-Matrix Recovery in Data-Scarce Educational Settings
URL: https://github.com/judit22/Q-matrix_analysis

---

## License

This project is licensed under the MIT License.

Copyright (c) 2025-2026 Judit Tevesz and Dr. Bertalan Forstner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
