import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from EduData import get_data
import warnings
import matplotlib.pyplot as plt
import logging
import NCDM_EarlyStopping
from tqdm import tqdm
from scipy.special import logit
from scipy.stats import zscore
from collections import Counter
from sklearn.model_selection import train_test_split

def def_items(items):
    item2knowledge = {}
    knowledge_set = set()
    for i, s in items.iterrows():
        item_id = s['item_id']
        knowledge_codes = list(set(map(int, s['knowledge_code'].split(','))))
        item2knowledge[item_id] = knowledge_codes
        knowledge_set.update(knowledge_codes)
    return item2knowledge, knowledge_set


def transform(user, item, item2knowledge, score, knowledge_n, batch_size=32):
    
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def transform_to_set(train_data, val_data, item2knowledge, knowledge_n):
  """
  Converts train, validation, and test DataFrames into PyTorch DataLoaders.

  Args:
    train_data (pd.DataFrame): Training data with columns 'user_id', 'item_id', and 'score'.
    val_data (pd.DataFrame): Validation data with columns 'user_id', 'item_id', and 'score'.
    item2knowledge (dict): Mapping from item IDs to lists of knowledge codes.
    knowledge_n (int): Total number of unique knowledge codes.

  Returns:
    tuple: (train_set, valid_set) as PyTorch DataLoaders.
  """
  print()
  train_set, valid_set = [
    transform(data["user_id"], data["item_id"], item2knowledge, data["score"], knowledge_n=knowledge_n, batch_size=32)
    for data in [train_data, val_data]
  ]
  return train_set, valid_set



def num_user_item_knowledge(train_data, val_data, knowledge_set):
    """
    Calculates the number of unique users, items, and knowledge codes from the provided datasets.

    Args:
        train_data (pd.DataFrame): Training data with a 'user_id' and 'item_id' column.
        val_data (pd.DataFrame): Validation data with an 'item_id' column.
        knowledge_set (set): Set of all unique knowledge codes.

    Returns:
        tuple: (user_n, item_n, knowledge_n)
            user_n (int): Maximum user ID in train_data.
            item_n (int): Maximum item ID across all datasets.
            knowledge_n (int): Maximum knowledge code in knowledge_set.
    """
    user_n = np.max(train_data['user_id'])
    item_n = np.max([np.max(train_data['item_id']), np.max(val_data['item_id'])])
    knowledge_n = np.max(list(knowledge_set))

    return user_n, item_n, knowledge_n


def split_train_valid(results, valid_size=0.2, random_state=None):

    train_data, val_data = train_test_split(
        results,
        test_size=valid_size,
        random_state=random_state
    )

    train_data.index = range(0,len(train_data))
    val_data.index = range(0,len(val_data))

    return train_data, val_data

def train_model(model, train_set, valid_set, knowledge_n, item_n, user_n, item_weights):
    model = model
    model.train(train_set, valid_set, epoch=100, device="cpu", item_weights = item_weights)
    model.save("ncdm.snapshot")



def training(items, results, item_weights):
    item2knowledge, knowledge_set = def_items(items)

    train_data, val_data = split_train_valid(results)

    item_counts = train_data["item_id"].value_counts().sort_index()
    num_items = train_data["item_id"].max() + 1

    item_weights = torch.ones(num_items)
    for item_id, freq in item_counts.items():
        item_weights[item_id] = 1.0 / np.sqrt(freq)
    item_weights /= item_weights.mean()

    user_n, item_n, knowledge_n = num_user_item_knowledge(train_data, val_data, knowledge_set)
    train, valid = transform_to_set(train_data, val_data, item2knowledge, knowledge_n)

    model = NCDM_EarlyStopping.NCDM(knowledge_n, item_n, user_n)
    train_model(model, train, valid, knowledge_n, item_n, user_n, item_weights)

    soft_preds = []


    weights = model.ncdm_net.k_difficulty.weight.detach().cpu().numpy()

    for i in range(len(weights)):
        row_raw = weights[i]

        #SOFT RANKING (top-k)
        k = 3
        topk = np.argsort(row_raw)[-k:] + 1
        soft_preds.append(topk)


    df = pd.DataFrame({
        "id": np.arange(len(weights)),
        "soft": soft_preds,
    })
    weights = model.ncdm_net.k_difficulty.weight.detach().cpu().numpy()

    probs = torch.sigmoid(model.ncdm_net.k_difficulty.weight).detach().cpu().numpy()


    return df,probs

def compute_skill_frequency(compare_df):
    
    skill_freq_per_item = []

    run_cols = [c for c in compare_df.columns]

    for i in range(len(compare_df)):
        counter = Counter()

        for col in run_cols:
            skills = compare_df.iloc[i][col]

            if isinstance(skills, (list, np.ndarray)):
                for s in skills:
                    counter[int(s)] += 1

        skill_freq_per_item.append(counter)

    return skill_freq_per_item

def extract_stable_skills(skill_freq_per_item, num_runs, min_ratio = 0.3):
    stable = []

    for counter in skill_freq_per_item:
        skills = []

        for skill, freq in counter.items():
            if freq / num_runs >= min_ratio:
                skills.append(skill)

        stable.append(skills)

    return stable
