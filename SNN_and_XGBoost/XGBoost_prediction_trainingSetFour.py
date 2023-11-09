import os

import numpy as np
import pandas as pd
import xgboost as xgb

from keras_tuner import Objective, RandomSearch, HyperModel
from sklearn.model_selection import train_test_split

from creatingPairsForTrainingSetFive_SNNXGBoost import trainingSetOne

with open('SNN_XGBoost_similarity_scores_trainingSetFive.txt', 'r') as f:
    similarity_scoresTrainingSetOne = [float(line.strip()[1:-1]) for line in f.readlines()]

combined_pairs_trainingSetOne, combined_labels_trainingSetOne, combined_pairs_trainingSetOneXGB, combined_team_pairs = trainingSetOne()
print("combined_pairs_trainingSetOne:  ",combined_pairs_trainingSetOne.shape)
print("similarity_scoresTrainingSetOne:  ",len(similarity_scoresTrainingSetOne))


similarity_scoresTrainingSetOne = np.array(similarity_scoresTrainingSetOne)
similarity_reshaped = similarity_scoresTrainingSetOne[:, np.newaxis, np.newaxis]



# Repeat the reshaped array along the second axis
similarity_repeated = np.repeat(similarity_reshaped, 2, axis=1)

# Concatenate the arrays along the last axis
con = np.concatenate((combined_pairs_trainingSetOne, similarity_repeated), axis=2)
con = con.reshape(con.shape[0], -1)

pairwise_y = combined_labels_trainingSetOne
pairwise_X = con



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

bst_aggregated.save_model('SNN_XGBoosttrainingSetFive.xgb')