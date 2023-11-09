# Load the data
from itertools import combinations
from data_preprocessing import preprocess_data
import numpy as np
import pandas as pd


def compute_labels_using_rank_difference(data, team_to_rank):
    pairs = []
    team_pairs = []
    labels = []
    for team1, team2 in combinations(data['team identifier'], 2):
        pairs.append([
            data[data['team identifier'] == team1].drop('team identifier', axis=1).values[0],
            data[data['team identifier'] == team2].drop('team identifier', axis=1).values[0]
        ])
        team_pairs.append((team1, team2))
        rank_team1 = team_to_rank[team1]
        rank_team2 = team_to_rank[team2]
        # labels.append(rank_team1 - rank_team2)
        labels.append(1 if rank_team1 < rank_team2 else 0)
    pairs = np.array(pairs)
    labels = np.array(labels)
    return team_pairs, pairs, labels


def trainingSetOne():
    data = pd.read_excel("../My_datasets/2012_13.xlsx")
    data_2013_2014 = pd.read_excel("../My_datasets/2013_14.xlsx")
    data_2014_2015 = pd.read_excel("../My_datasets/2014_15.xlsx")
    data_2015_2016 = pd.read_excel("../My_datasets/2015_16.xlsx")
    data_2016_2017 = pd.read_excel("../My_datasets/2016_17.xlsx")
    data_2017_2018 = pd.read_excel("../My_datasets/2017_18.xlsx")
    data_2018_2019 = pd.read_excel("../My_datasets/2018_19.xlsx")
    data_2019_2020 = pd.read_excel("../My_datasets/2019_20.xlsx")
    data_2020_2021 = pd.read_excel("../My_datasets/2020_21.xlsx")




    # 2012/13
    numerical_cols_data = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols_data = data.select_dtypes(include=['object']).columns
    data_preprocessed = preprocess_data(data, numerical_cols_data, categorical_cols_data)
    data_preprocessed['team identifier'] = data['team identifier']
    # 2013/14
    numerical_cols_data_2013_2014 = data_2013_2014.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols_data_2013_2014 = data_2013_2014.select_dtypes(include=['object']).columns
    data_preprocessed_data_2013_2014 = preprocess_data(data_2013_2014, numerical_cols_data_2013_2014,
                                                       categorical_cols_data_2013_2014)
    data_preprocessed_data_2013_2014['team identifier'] = data_2013_2014['team identifier']
    # 2014/15
    numerical_cols_data_2014_2015 = data_2014_2015.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols_data_2014_2015 = data_2014_2015.select_dtypes(include=['object']).columns
    data_preprocessed_data_2014_2015 = preprocess_data(data_2014_2015, numerical_cols_data_2014_2015,
                                                       categorical_cols_data_2014_2015)
    data_preprocessed_data_2014_2015['team identifier'] = data_2014_2015['team identifier']
    # 2015/16
    numerical_cols_data_2015_2016 = data_2015_2016.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols_data_2015_2016 = data_2015_2016.select_dtypes(include=['object']).columns
    data_preprocessed_data_2015_2016 = preprocess_data(data_2015_2016, numerical_cols_data_2015_2016,
                                                       categorical_cols_data_2015_2016)
    data_preprocessed_data_2015_2016['team identifier'] = data_2015_2016['team identifier']
    # 2016/17
    numerical_cols_data_2016_2017 = data_2016_2017.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols_data_2016_2017 = data_2016_2017.select_dtypes(include=['object']).columns
    data_preprocessed_data_2016_2017 = preprocess_data(data_2016_2017, numerical_cols_data_2016_2017,
                                                       categorical_cols_data_2016_2017)
    data_preprocessed_data_2016_2017['team identifier'] = data_2016_2017['team identifier']
    # 2017/18
    numerical_cols_data_2017_2018 = data_2017_2018.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols_data_2017_2018 = data_2017_2018.select_dtypes(include=['object']).columns
    data_preprocessed_data_2017_2018 = preprocess_data(data_2017_2018, numerical_cols_data_2017_2018,
                                                       categorical_cols_data_2017_2018)
    data_preprocessed_data_2017_2018['team identifier'] = data_2017_2018['team identifier']
    # 2018/19
    numerical_cols_data_2018_2019 = data_2018_2019.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols_data_2018_2019 = data_2018_2019.select_dtypes(include=['object']).columns
    data_preprocessed_data_2018_2019 = preprocess_data(data_2018_2019, numerical_cols_data_2018_2019,
                                                       categorical_cols_data_2018_2019)
    data_preprocessed_data_2018_2019['team identifier'] = data_2018_2019['team identifier']
    # 2019/20
    numerical_cols_data_2019_2020 = data_2019_2020.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols_data_2019_2020 = data_2019_2020.select_dtypes(include=['object']).columns
    data_preprocessed_data_2019_2020 = preprocess_data(data_2019_2020, numerical_cols_data_2019_2020,
                                                       categorical_cols_data_2019_2020)
    data_preprocessed_data_2019_2020['team identifier'] = data_2019_2020['team identifier']
    # 2020/21
    numerical_cols_data_2020_2021 = data_2020_2021.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols_data_2020_2021 = data_2020_2021.select_dtypes(include=['object']).columns
    data_preprocessed_data_2020_2021 = preprocess_data(data_2020_2021, numerical_cols_data_2020_2021,
                                                       categorical_cols_data_2020_2021)
    data_preprocessed_data_2020_2021['team identifier'] = data_2020_2021['team identifier']


    #2012/13
    team_to_rank_2012_13 = {}
    actual_rankings_2012_13 = data_preprocessed['team identifier'].values
    for team, rank in zip(data_preprocessed['team identifier'], actual_rankings_2012_13):
        team_to_rank_2012_13[team] = rank

    # Compute the labels using difference in rankings for 2012/13 season
    team_pairs, pairs, labels = compute_labels_using_rank_difference(data_preprocessed, team_to_rank_2012_13)
    flattened_pairs = pairs.reshape(pairs.shape[0], -1)


    #2013/14
    team_to_rank_2013_14 = {}
    actual_rankings_2013_14 = data_preprocessed_data_2013_2014['team identifier'].values
    for team, rank in zip(data_preprocessed_data_2013_2014['team identifier'], actual_rankings_2013_14):
        team_to_rank_2013_14[team] = rank

    # Compute the labels using difference in rankings for 2012/13 season
    team_pairs_2013_2014, pairs_2013_14, labels_2013_14 = compute_labels_using_rank_difference(data_preprocessed_data_2013_2014, team_to_rank_2013_14)
    flattened_pairs_2013_14 = pairs_2013_14.reshape(pairs_2013_14.shape[0], -1)


    #2014/15
    team_to_rank_2014_15 = {}
    actual_rankings_2014_15 = data_preprocessed_data_2014_2015['team identifier'].values
    for team, rank in zip(data_preprocessed_data_2014_2015['team identifier'], actual_rankings_2014_15):
        team_to_rank_2014_15[team] = rank

    #Compute the labels using difference in rankings for 2012/13 season
    team_pairs_2014_2015, pairs_2014_15, labels_2014_15 = compute_labels_using_rank_difference(data_preprocessed_data_2014_2015,
                                                                                               team_to_rank_2014_15)
    flattened_pairs_2014_15 = pairs_2014_15.reshape(pairs_2014_15.shape[0], -1)

    #2015/16
    team_to_rank_2015_16 = {}
    actual_rankings_2015_16 = data_preprocessed_data_2015_2016['team identifier'].values
    for team, rank in zip(data_preprocessed_data_2015_2016['team identifier'], actual_rankings_2015_16):
        team_to_rank_2015_16[team] = rank

    #Compute the labels using difference in rankings for 2012/13 season
    team_pairs_2015_2016, pairs_2015_16, labels_2015_16 = compute_labels_using_rank_difference(
        data_preprocessed_data_2015_2016,
        team_to_rank_2015_16)

    flattened_pairs_2015_16 = pairs_2015_16.reshape(pairs_2015_16.shape[0], -1)


    #2016/17
    team_to_rank_2016_17 = {}
    actual_rankings_2016_17 = data_preprocessed_data_2016_2017['team identifier'].values
    for team, rank in zip(data_preprocessed_data_2016_2017['team identifier'], actual_rankings_2016_17):
        team_to_rank_2016_17[team] = rank

    #Compute the labels using difference in rankings for 2012/13 season
    team_pairs_2016_2017, pairs_2016_17, labels_2016_17 = compute_labels_using_rank_difference(
        data_preprocessed_data_2016_2017,
        team_to_rank_2016_17)
    flattened_pairs_2016_17 = pairs_2016_17.reshape(pairs_2016_17.shape[0], -1)

    #2017/18
    team_to_rank_2017_18 = {}
    actual_rankings_2017_18 = data_preprocessed_data_2017_2018['team identifier'].values
    for team, rank in zip(data_preprocessed_data_2017_2018['team identifier'], actual_rankings_2017_18):
        team_to_rank_2017_18[team] = rank

    #Compute the labels using difference in rankings for 2012/13 season
    team_pairs_2017_2018, pairs_2017_18, labels_2017_18 = compute_labels_using_rank_difference(
        data_preprocessed_data_2017_2018,
        team_to_rank_2017_18)
    flattened_pairs_2017_18 = pairs_2017_18.reshape(pairs_2017_18.shape[0], -1)

    #2018/19
    team_to_rank_2018_19 = {}
    actual_rankings_2018_19 = data_preprocessed_data_2018_2019['team identifier'].values
    for team, rank in zip(data_preprocessed_data_2018_2019['team identifier'], actual_rankings_2018_19):
        team_to_rank_2018_19[team] = rank

    # Compute the labels using difference in rankings for 2012/13 season
    team_pairs_2018_2019, pairs_2018_2019, labels_2018_2019 = compute_labels_using_rank_difference(
        data_preprocessed_data_2018_2019,
        team_to_rank_2018_19)
    flattened_pairs_2018_2019 = pairs_2018_2019.reshape(pairs_2018_2019.shape[0], -1)

    # 2019/20
    team_to_rank_2019_20 = {}
    actual_rankings_2019_20 = data_preprocessed_data_2019_2020['team identifier'].values
    for team, rank in zip(data_preprocessed_data_2019_2020['team identifier'], actual_rankings_2019_20):
        team_to_rank_2019_20[team] = rank

    # Compute the labels using difference in rankings for 2012/13 season
    team_pairs_2019_2020, pairs_2019_2020, labels_2019_2020 = compute_labels_using_rank_difference(
        data_preprocessed_data_2019_2020,
        team_to_rank_2019_20)
    flattened_pairs_2019_2020 = pairs_2019_2020.reshape(pairs_2019_2020.shape[0], -1)

    # 2020/21
    team_to_rank_2020_21 = {}
    actual_rankings_2020_21 = data_preprocessed_data_2020_2021['team identifier'].values
    for team, rank in zip(data_preprocessed_data_2020_2021['team identifier'], actual_rankings_2020_21):
        team_to_rank_2020_21[team] = rank

    # Compute the labels using difference in rankings for 2012/13 season
    team_pairs_2020_2021, pairs_2020_2021, labels_2020_2021 = compute_labels_using_rank_difference(
        data_preprocessed_data_2020_2021,
        team_to_rank_2020_21)
    flattened_pairs_2020_2021 = pairs_2020_2021.reshape(pairs_2020_2021.shape[0], -1)




    combined_team_pairs = team_pairs + team_pairs_2013_2014 + team_pairs_2014_2015 + team_pairs_2015_2016 +  team_pairs_2016_2017 + team_pairs_2017_2018+team_pairs_2018_2019+ team_pairs_2019_2020+ team_pairs_2020_2021

    combined_pairs_trainingSetOne = np.concatenate((pairs, pairs_2013_14, pairs_2014_15, pairs_2015_16, pairs_2016_17,
                                                    pairs_2017_18, pairs_2018_2019, pairs_2019_2020, pairs_2020_2021), axis=0)
    print("shape from function:    " ,combined_pairs_trainingSetOne.shape)
    combined_labels_trainingSetOne = np.concatenate((labels, labels_2013_14, labels_2014_15, labels_2015_16, labels_2016_17,
                                                     labels_2017_18, labels_2018_2019, labels_2019_2020, labels_2020_2021), axis=0)
    combined_pairs_trainingSetOneXGB = np.concatenate((flattened_pairs, flattened_pairs_2013_14, flattened_pairs_2014_15, flattened_pairs_2015_16,
                                                       flattened_pairs_2016_17, flattened_pairs_2017_18, flattened_pairs_2018_2019, flattened_pairs_2019_2020, flattened_pairs_2020_2021, flattened_pairs_2020_2021), axis=0)

    return combined_pairs_trainingSetOne, combined_labels_trainingSetOne, combined_pairs_trainingSetOneXGB, combined_team_pairs

