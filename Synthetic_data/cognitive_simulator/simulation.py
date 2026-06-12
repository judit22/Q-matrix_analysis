import numpy as np
from .params import Param
from .utils import irt, save_as_csv, save_as_datasynthesis_csv, binomial_not_all_zero

def generate_abilities(n, k):
    return np.array([np.random.normal(1, 0.15, k) for _ in range(n)])

def simulate_exercise_params(n, exercise, exercise_weights=[], params=[], param_types: list[Param] = [], binary=False, csv_name=None, abilities=None):
    if len(params) != 0:
        k = len(exercise)
        exercise = np.array(exercise)
        param_vector = np.ones(k)
        for i, param in enumerate(params):
            param_vector[param_types[i].ability_index] = param_types[i].get_multiplier(param)
        exercise = exercise * np.array(param_vector)
    
    abilities, results = simulate_exercise(n, exercise, exercise_weights, binary, abilities)
    if csv_name is not None:
        save_as_csv(abilities, results, filename=csv_name)
    return abilities, results

def simulate_exercise(n, exercise, exercise_weights=[], binary=False, abilities_matrix=None):
    k = len(exercise)
    try:
        if (abilities_matrix == None):
            abilities_matrix = generate_abilities(n, k)
    except ValueError:
        pass
    results = np.array([solve_exercise(abilities, exercise, exercise_weights, binary=binary) for abilities in abilities_matrix])
    return abilities_matrix, results

def solve_exercise(person_abilities, exercise, exercise_weights=[], epsilon=0.02, binary=False):
    if len(person_abilities) != len(exercise):
        raise ValueError("Incompatible input sizes")
    m = len(exercise)
    exercise = np.array(exercise)
    not_null_m = np.sum(exercise != 0)
    if len(exercise_weights) == 0:
        exercise_weights = [1] * m
    else:
        for i in range(m):
            if exercise[i] == 0:
                exercise_weights[i] = 0
    exercise_weights = np.array(exercise_weights) / np.sum(exercise_weights) * not_null_m
    product = 1
    for i in range(m):
        theta = person_abilities[i] + np.random.uniform(-epsilon, epsilon)
        beta = exercise[i]
        if beta != 0:
            product *= (irt(beta, theta) ** exercise_weights[i])
    if binary:
        return int(product > 0.5)
    return product

def data_synthesis(number_of_people, number_of_exercises, number_of_abilities=5, binary=False, csv_name=None, relevant_ability_probability=None):
    n, m, k = number_of_people, number_of_exercises, number_of_abilities
    if(relevant_ability_probability==None):
        if k>=3:
            relevant_ability_probability = 3/k
        else:
            relevant_ability_probability = 1
    abilities = np.array([np.random.normal(1, 0.15, k) for _ in range(n)])
    exercises = [np.random.normal(1, 0.15, k) * binomial_not_all_zero(1, relevant_ability_probability, k) for _ in range(m)]
    exercise_weights = [np.random.uniform(0, 1, k) for _ in range(m)]
    results = np.array([[solve_exercise(ability, exercise, exercise_weights[i], binary=binary) for i, exercise in enumerate(exercises)] for ability in abilities])

    data = []
    for person_idx, (ability, person_results) in enumerate(zip(abilities, results)):
        for exercise_idx, result in enumerate(person_results):
            data.append({
                'person_id': person_idx + 1,
                'exercise_id': exercise_idx + 1,
                'ability': ability,
                'exercise': exercises[exercise_idx],
                'weight': exercise_weights[exercise_idx],
                'result': result
            })
    if csv_name is not None:
        save_as_datasynthesis_csv(data, filename=csv_name)
    return data
