import os

import pandas as pd
import xgboost as xgb
import numpy as np
from itertools import combinations

from keras_tuner import Objective, RandomSearch, HyperModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_preprocessing import preprocess_data
from creatingPairsForTrainingSetOne_SNNXGBoost import trainingSetOne


def predict_rankings_using_similarity(model, test_data, similarity_scores, combined_team_pairs):
    """Predict rankings using the provided model, test data, and similarity scores."""
    scores = []

    for i in range(len(test_data)):
        team = test_data.iloc[i]["team identifier"]

        print(f"Processing team: {test_data.iloc[i]['Team']} with identifier: {team}")

        # Retrieve similarity scores for the current team against all other teams
        team_similarity_scores = []
        for j, (team_a, team_b) in enumerate(combined_team_pairs):
            if team == team_a or team == team_b:
                team_similarity_scores.append(similarity_scores[j])

        # If no similarity scores are found, append a default score (e.g., 0.0)
        if not team_similarity_scores:
            print(f"No similarity scores found for team: {team}")
            scores.append(0.0)
        else:
            dtest_team_similarity = xgb.DMatrix(pd.DataFrame(team_similarity_scores, columns=["similarity"]))
            preds_team = model.predict(dtest_team_similarity)
            scores.append(np.sum(preds_team))

    # Rank the teams based on the predicted scores
    test_data["Predicted Score"] = scores
    test_data = test_data.sort_values(by="Predicted Score", ascending=False)
    test_data["Predicted Rank"] = range(1, len(test_data) + 1)

    ordered_by_identifier = test_data.sort_values(by="team identifier")

    # Extract the 'Predicted Rank' column as an array
    predicted_ranks_array = test_data["team identifier"].to_numpy()

    return test_data[["Team", "Rank", "Predicted Rank"]], predicted_ranks_array


files = [
   "../My_datasets/2012_13.xlsx",
    "../My_datasets/2013_14.xlsx",
    "../My_datasets/2014_15.xlsx",
    "../My_datasets/2015_16.xlsx",
    "../My_datasets/2016_17.xlsx",
    "../My_datasets/2017_18.xlsx",
    "../My_datasets/2018_19.xlsx"

]
datasets = [pd.read_excel(file) for file in files]

# Preprocess datasets
numerical_cols_data = datasets[0].select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols_data = [col for col in numerical_cols_data if col != "team identifier"]
categorical_cols_data = datasets[0].select_dtypes(include=['object']).columns.tolist()
categorical_cols_data = [col for col in categorical_cols_data if col != "Team"]
preprocessed_datasets = [preprocess_data(df, numerical_cols_data, categorical_cols_data) for df in datasets]

# Prepare test data
test_data = preprocessed_datasets[-1][["Team", "team identifier"]].copy()
test_data["Rank"] = test_data.index + 1

combined_pairs_trainingSetOne, combined_labels_trainingSetOne, combined_pairs_trainingSetOneXGB, combined_team_pairs = trainingSetOne()

with open('SNN_XGBoost_similarity_scores_trainingSetOne.txt', 'r') as f:
    similarity_scoresTrainingSetOne = [float(line.strip()[1:-1]) for line in f.readlines()]


bst = xgb.Booster()
bst.load_model('SNN_XGBoosttrainingSetOne.xgb')

predicted_rankings, predicted_ranks_array = predict_rankings_using_similarity(
    bst,
    test_data,
    similarity_scoresTrainingSetOne,
    combined_team_pairs
)

with open("predicted_ranks_trainingSetOne.txt", "w") as file:
    for i in predicted_ranks_array:
        file.write(f"{i}\n")
print(predicted_ranks_array)

print(predicted_rankings)