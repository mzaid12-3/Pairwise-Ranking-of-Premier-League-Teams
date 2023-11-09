import os
import pandas as pd
import xgboost as xgb
import numpy as np
from itertools import combinations
from keras_tuner import Objective, RandomSearch, HyperModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_preprocessing import preprocess_data
from SNN_and_XGBoost.creatingPairsForTrainingSetFive_SNNXGBoost import trainingSetOne

combined_pairs_trainingSetOne, combined_labels_trainingSetOne, combined_pairs_trainingSetOneXGB, combined_team_pairs = trainingSetOne()
print("combined_pairs_trainingSetOne: ", combined_pairs_trainingSetOneXGB.shape)


x_last_epoch = np.round(np.loadtxt(f'activations_a_epoch_trainingSetFive49.txt'),1)
y_last_epoch = np.round(np.loadtxt(f'activations_a_epoch_trainingSetFive49.txt'),1)

activation_diff = x_last_epoch - y_last_epoch
# Split the combined_pairs_trainingSetOneXGB after the 51st feature
part1 = combined_pairs_trainingSetOneXGB[:, :51]
part2 = combined_pairs_trainingSetOneXGB[:, 51:]
print("Shape of part1:", part1.shape)
print("Shape of part2:", part2.shape)
print("Shape of x_last_epoch:", x_last_epoch.shape)
print("Shape of y_last_epoch:", y_last_epoch.shape)

# Concatenate the three parts: part1, x_last_epoch, and y_last_epoch
new_combined_pairs = np.hstack((part1, x_last_epoch, part2, y_last_epoch))

# Replace the original combined_pairs_trainingSetOneXGB with the new concatenated array
combined_pairs_trainingSetOneXGB = new_combined_pairs

print("Updated combined_pairs_trainingSetOneXGB shape:", combined_pairs_trainingSetOneXGB.shape)

combined_pairs_df = pd.DataFrame(combined_pairs_trainingSetOneXGB)
pairwise_X = combined_pairs_df

pairwise_y = combined_labels_trainingSetOne
X_train, X_val, y_train, y_val = train_test_split(pairwise_X, pairwise_y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

class XGBHyperModel(HyperModel):
    def build(self, hp):
        param = {
            'max_depth': hp.Int('max_depth', 3, 10, 1),
            'eta': hp.Float('eta', 0.01, 0.5, step=0.01),
            'subsample': hp.Float('subsample', 0.5, 1),
            'colsample_bytree': hp.Float('colsample_bytree', 0.5, 1),
            'gamma': hp.Float('gamma', 0, 5),
            'min_child_weight': hp.Int('min_child_weight', 1, 10),
            'lambda': hp.Float('lambda', 0.01, 1),
            'alpha': hp.Float('alpha', 0.01, 1),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        return param

class XGBTuner(RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        params = self.hypermodel.build(hp)
        evals_result = {}
        bst = xgb.train(params, dtrain, evals=[(dval, 'eval')], early_stopping_rounds=10, verbose_eval=False, evals_result=evals_result)
        last_eval = evals_result['eval']['logloss'][-1]
        self.oracle.update_trial(trial.trial_id, {'val_logloss': last_eval})
        self.save_model(trial.trial_id, bst)

    def save_model(self, trial_id, model, step=0):
        fname = os.path.join(self.get_trial_dir(trial_id), f'model_{step}.xgb')
        model.save_model(fname)

    def load_model(self, trial_id, step=0):
        fname = os.path.join(self.get_trial_dir(trial_id), f'model_{step}.xgb')
        model = xgb.Booster()
        model.load_model(fname)
        return model

tuner = XGBTuner(XGBHyperModel(), objective=Objective('val_logloss', direction='min'), max_trials=50, directory='xgb_tuner', project_name='xgb_tuning')
tuner.search()

best_hp = tuner.get_best_hyperparameters()[0]
best_params = {
    'max_depth': best_hp.get('max_depth'),
    'eta': best_hp.get('eta'),
    'subsample': best_hp.get('subsample'),
    'colsample_bytree': best_hp.get('colsample_bytree'),
    'gamma': best_hp.get('gamma'),
    'min_child_weight': best_hp.get('min_child_weight'),
    'lambda': best_hp.get('lambda'),
    'alpha': best_hp.get('alpha'),
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

num_round = 60
bst_aggregated = xgb.train(best_params, dtrain, num_round, evals=[(dval, 'eval')], early_stopping_rounds=10)
bst_aggregated.save_model('XGBoosttrainingSetFive.xgb')

def load_best_model(model_path):
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst

def predict_using_train_data(model, train_data):
    dtrain_full = xgb.DMatrix(train_data)
    predictions = model.predict(dtrain_full)
    return predictions

bst = load_best_model('XGBoosttrainingSetFive.xgb')
predictions_train = predict_using_train_data(bst, pairwise_X)
print(predictions_train)


files = [
    "../My_datasets/2012_13.xlsx",
    "../My_datasets/2013_14.xlsx",
    "../My_datasets/2014_15.xlsx",
    "../My_datasets/2015_16.xlsx",
    "../My_datasets/2016_17.xlsx",
    "../My_datasets/2017_18.xlsx",
    "../My_datasets/2018_19.xlsx",
    "../My_datasets/2019_20.xlsx",
    "../My_datasets/2020_21.xlsx",
    "../My_datasets/2021_22.xlsx"

]

season_standings = []

# Reading and processing each Excel file
for file in files:
    df = pd.read_excel(file)
    # Extract the team identifier and its position based on index
    standings_dict = {row['team identifier']: idx + 1 for idx, row in df.iterrows()}
    season_standings.append(standings_dict)

actual_rankings = [11,
30,
22,
6,
8,
23,
35,
14,
33,
25,
12,
3,
7,
27,
2,
32,
5,
15,
18,
16
]

# Assuming combined_team_pairs is an array of tuples where each tuple is (team1, team2)
# Extract pairs containing the teams in actual_rankings
filtered_pairs = [pair for pair in combined_team_pairs if pair[0] in actual_rankings or pair[1] in actual_rankings]

# Convert these pairs to feature differences similar to how you prepared your training data
filtered_data = [pairwise_X.iloc[i] for i, pair in enumerate(combined_team_pairs) if pair in filtered_pairs]

# Convert to DataFrame and then to DMatrix for prediction
filtered_data_df = pd.DataFrame(filtered_data)
dtest_filtered = xgb.DMatrix(filtered_data_df)

# Predict using the trained model
predictions_filtered = bst.predict(dtest_filtered)

# Sum predictions for each team based on your logic
team_prediction_sums = {team: 0 for team in actual_rankings}

for idx, pair in enumerate(filtered_pairs):
    team1 = pair[0]
    team2 = pair[1]

    try:
        team_prediction_sums[team1] += predictions_filtered[idx]
        print(team_prediction_sums[team1], ":", predictions_filtered[idx])
        team_prediction_sums[team2] += predictions_filtered[idx]
    except KeyError:
        print(f"Team identifier {team2} not found in team_scores dictionary.")
        if team2 not in actual_rankings:
            print(f"Team identifier {team2} is also not in the original team_ids list.")


total_matches = len(combined_pairs_trainingSetOne) / 2  # Each team appears in half the pairs
average_probabilities = {team_id: score / total_matches for team_id, score in team_prediction_sums.items()}

season_weights = [1, 2, 3, 4, 5, 6, 7,8, 9 ,10]
weighted_standings = []
for team_id in actual_rankings:
    team_weighted_standing = sum(
        season_standings[season].get(team_id, len(actual_rankings) + 1) * season_weights[season] for season in range(10))
    weighted_standings.append(team_weighted_standing)

max_weighted_value = len(actual_rankings) * sum(season_weights)
normalized_standings = [(max_weighted_value - standing) / max_weighted_value for standing in weighted_standings]

w1 = 0.7
w2 = 0.3

adjusted_scores = {team_id: w1 * average_probabilities[team_id] + w2 * normalized_standings[actual_rankings.index(team_id)] for
                   team_id in actual_rankings}
for team_id, score in adjusted_scores.items():
    print(f"Team ID: {team_id}, Score: {score}")

scalar_scores = {team_id: np.sum(score) for team_id, score in adjusted_scores.items()}
sorted_teams_scores = sorted(scalar_scores.items(), key=lambda x: x[1], reverse=True)

# Display the final predictions

for team, prediction in sorted_teams_scores:
    print(f"Team {team}: {prediction}")

sorted_rankings = [team[0] for team in sorted_teams_scores]
print(sorted_rankings)
with open('predicted_rankingsTrainingSetFive.txt', 'w') as f:
    for team in sorted_rankings:
        f.write(f"{team}\n")
