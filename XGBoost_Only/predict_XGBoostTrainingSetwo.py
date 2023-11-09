import numpy as np
import xgboost as xgb
import pandas as pd
from data_preprocessing import preprocess_data


def load_best_model(model_path):
    """Load the saved XGBoost model."""
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst


def predict_rankings(model, test_data, aggregated_data):
    """Predict rankings using the provided model and test data."""
    scores = []
    for i in range(len(test_data)):
        team_id = test_data.iloc[i]["team identifier"]
        team_data_aggregated = aggregated_data[aggregated_data["team identifier"] == team_id]

        if not team_data_aggregated.empty:
            team_features = team_data_aggregated.iloc[0].drop("team identifier")
            pairwise_data_temp = [team_features - aggregated_data.iloc[j].drop("team identifier") for j in
                                  range(len(aggregated_data))]
            pairwise_X_temp = pd.DataFrame(pairwise_data_temp)

            dtest_pairwise_temp = xgb.DMatrix(pairwise_X_temp)
            preds_temp = model.predict(dtest_pairwise_temp)

            scores.append(np.mean(preds_temp))
        else:
            scores.append(0.0)

    # Rank the teams based on the aggregated scores
    test_data["Predicted Score"] = scores
    test_data = test_data.sort_values(by="Predicted Score", ascending=False)
    test_data["Predicted Rank"] = range(1, len(test_data) + 1)

    ordered_by_identifier = test_data.sort_values(by="team identifier")

    # Extract the 'Predicted Rank' column as an array
    predicted_ranks_array = test_data["team identifier"].to_numpy()

    return test_data[["Team", "Rank", "Predicted Rank"]], predicted_ranks_array


if __name__ == "__main__":
    # Load datasets
    files = [
         "../My_datasets/2012_13.xlsx",
        "../My_datasets/2013_14.xlsx",
        "../My_datasets/2014_15.xlsx",
        "../My_datasets/2015_16.xlsx",
        "../My_datasets/2016_17.xlsx",
        "../My_datasets/2017_18.xlsx",
        "../My_datasets/2018_19.xlsx",
"../My_datasets/2019_20.xlsx"
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

    # Aggregate features by team identifier
    train_data = pd.concat(preprocessed_datasets[:-1], ignore_index=True)
    aggregated_data = train_data.groupby("team identifier").mean().reset_index()

    # Load the best model
    bst = load_best_model('XGBoosttrainingSetTwo.xgb')

    # Predict rankings
    predicted_rankings, predicted_ranks_array = predict_rankings(bst, test_data, aggregated_data)
    print(predicted_rankings)




    print(predicted_ranks_array)

    with open('predicted_rankingsTrainingSetTwo.txt', 'w') as f:
        for team in predicted_ranks_array:
            f.write(f"{team}\n")









