import numpy as np
import pandas as pd

def irt(beta, theta, discrimination=20, std_dev=0.15):
    return 1 / (1 + np.exp(discrimination * (beta - std_dev - theta)))

def binomial_not_all_zero(n, p, k):
    while True:
        result = np.random.binomial(n, p, k)
        if np.sum(result) != 0:
            return result

def save_to_df(abilities, results):
    results_df = pd.DataFrame()
    results_df['person_id'] = range(1, len(abilities) + 1)
    for i in range(len(abilities[0])):
        results_df[f'ability_{i+1}'] = [ability[i] for ability in abilities]
    results_df['result'] = results
    return results_df

def save_as_csv(abilities, results, filename='results.csv'):
    results_df = save_to_df(abilities, results)
    results_df.to_csv(filename, index=False, float_format='%.10f')



def save_as_datasynthesis_csv(data, filename='results.csv'):
    rows = []

    # Eredmények feldolgozása
    for entry in data:
        person_id = entry['person_id']
        exercise_id = entry['exercise_id']
        result = entry['result']
        ability = entry['ability']  # Személy képességei
        exercise = entry['exercise']  # Feladat részképességei
        weight = entry['weight']  # A feladat súlya

        # Sor létrehozása a DataFrame-hez
        row = {
            'person_id': person_id,
            'exercise_id': exercise_id,
            'result': result
        }

        # Képességek hozzáadása a sorhoz
        for i, ability_value in enumerate(ability):
            row[f'ability_{i + 1}'] = ability_value

        # Az exercise részképességeinek hozzáadása a sorhoz
        for i, exercise_value in enumerate(exercise):
            row[f'exercise_skill_{i + 1}'] = exercise_value
            row[f'exercise_weight_{i + 1}'] = weight[i]  # A súly hozzáadása

        # A sor hozzáadása a listához
        rows.append(row)

    # DataFrame létrehozása a sorokból
    results_df = pd.DataFrame(rows)

    # CSV exportálás
    results_df.to_csv(filename, index=False, float_format='%.10f')
