import os

import pandas as pd
import xgboost as xgb
import numpy as np
from itertools import combinations

from keras_tuner import Objective, RandomSearch, HyperModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_preprocessing import preprocess_data
# Load datasets
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
datasets = [pd.read_excel(file) for file in files]
# Extract numerical and categorical columns from the first dataset
numerical_cols_data = datasets[0].select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols_data = [col for col in numerical_cols_data if col != "team identifier"]

# Preprocess datasets
# Exclude 'Team' when extracting categorical columns
categorical_cols_data = datasets[0].select_dtypes(include=['object']).columns.tolist()
categorical_cols_data = [col for col in categorical_cols_data if col != "Team"]

# Preprocess datasets
preprocessed_datasets = [preprocess_data(df, numerical_cols_data, categorical_cols_data) for df in datasets]

# Check columns in the last preprocessed dataset
# print(preprocessed_datasets[-1].columns)



# Combine the first six datasets for training
train_data = pd.concat(preprocessed_datasets[:-1], ignore_index=True)

# Extract team identifiers and rankings from the seventh dataset for testing
test_data = preprocessed_datasets[-1][["Team", "team identifier"]].copy()
test_data["Rank"] = test_data.index + 1
print(test_data["team identifier"])
# Normalize the features in the training data


# Aggregate features by team identifier
aggregated_data = train_data.groupby("team identifier").mean().reset_index()
# Prepare pairwise comparison data for aggregated data
all_pairwise_data = []



# Creating pairwise ranking data for each season separately.
# ...

for dataset in preprocessed_datasets[:-1]:  # Exclude the last dataset (test dataset)
    # Drop non-numeric columns
    numeric_dataset = dataset.select_dtypes(include=['int64', 'float64']).drop("team identifier", axis=1)

    for i, j in combinations(numeric_dataset.index, 2):
        feature_diff = numeric_dataset.iloc[i] - numeric_dataset.iloc[j]

        # If i < j, then label = 1, indicating team i is ranked higher than team j
        label = 1 if i < j else 0
        all_pairwise_data.append((feature_diff, label))

        # Data Augmentation: Add the reverse pair with the opposite label
        feature_diff_reverse = numeric_dataset.iloc[j] - numeric_dataset.iloc[i]
        label_reverse = 1 - label
        all_pairwise_data.append((feature_diff_reverse, label_reverse))

# ...


pairwise_X = pd.DataFrame([item[0] for item in all_pairwise_data])
pairwise_y = [item[1] for item in all_pairwise_data]
print(pairwise_y)

# Split the data and convert it to DMatrix
X_train, X_val, y_train, y_val = train_test_split(
    pairwise_X,
    pairwise_y,
    test_size=0.2,
    random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)


# Define the XGBoost hypermodel to only build the parameter dictionary
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

# Define the custom tuner
class XGBTuner(RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        params = self.hypermodel.build(hp)

        # Train the model and get evaluation results
        evals_result = {}
        bst = xgb.train(params, dtrain, evals=[(dval, 'eval')],
                        early_stopping_rounds=10, verbose_eval=False, evals_result=evals_result)

        # Get the last evaluation result
        last_eval = evals_result['eval']['logloss'][-1]  # Changed rmse to logloss

        # Report the result to the tuner
        self.oracle.update_trial(trial.trial_id, {'val_logloss': last_eval})  # Changed val_rmse to val_logloss
        self.save_model(trial.trial_id, bst)

    def save_model(self, trial_id, model, step=0):
        fname = os.path.join(self.get_trial_dir(trial_id), f'model_{step}.xgb')
        model.save_model(fname)

    def load_model(self, trial_id, step=0):
        fname = os.path.join(self.get_trial_dir(trial_id), f'model_{step}.xgb')
        model = xgb.Booster()
        model.load_model(fname)
        return model

# Use the custom tuner to search for the best hyperparameters
tuner = XGBTuner(
    XGBHyperModel(),
    objective=Objective('val_logloss', direction='min'),
    max_trials=50,
    directory='xgb_tuner',
    project_name='xgb_tuning'
)

tuner.search()
# Display the best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print("Best max_depth:", best_hp.get('max_depth'))
print("Best eta:", best_hp.get('eta'))
print("Best subsample:", best_hp.get('subsample'))
print("Best colsample_bytree:", best_hp.get('colsample_bytree'))
print("Best gamma:", best_hp.get('gamma'))
print("Best min_child_weight:", best_hp.get('min_child_weight'))
print("Best lambda (L2 regularization):", best_hp.get('lambda'))
print("Best alpha (L1 regularization):", best_hp.get('alpha'))
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

bst_aggregated.save_model('XGBoosttrainingSetFour.xgb')