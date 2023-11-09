# predicting 2021 22
# Necessary Imports
import tf
from keras.models import Model, Sequential
from keras.layers import Input, subtract, Dense, Dropout
from keras import backend as K
from keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from creatingPairsForTrainingSetFour import trainingSetOne
import tensorflow as tf
import numpy as np
# Splitting the data

combined_pairs_trainingSetOne, combined_labels_trainingSetOne, _, team_pairs = trainingSetOne()
team_pairs = np.array(team_pairs).astype('float32')



X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
    combined_pairs_trainingSetOne[:, 0], combined_pairs_trainingSetOne[:, 1], combined_labels_trainingSetOne,
    test_size=0.2, random_state=42)


# Assuming the necessary imports are already in place
# def adjusted_ranknet_loss(y_true, y_pred):
#     y_true = tf.cast(y_true, tf.float32)  # Ensure y_true is of type float32
#     prob = 1.0 / (1.0 + K.exp(-y_pred))
#     return K.binary_crossentropy(y_true, prob)
def adjusted_ranknet_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    return K.log(1 + K.exp(-y_true * y_pred))
# Define the base network outside
def base_network(input_shape, hp):
    model = Sequential()
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(hp.Int(f'dense_units_{i}', min_value=16, max_value=135, step=17), activation='relu'))
        model.add(Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)))
    model.add(Dense(1, activation='linear'))
    return model


def build_model(hp):
    input_shape = (X1_train.shape[1],)  # Assuming X1_train is a 2D array
    print("X1_train shape:", input_shape )
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)

    base_net = base_network(input_shape, hp)

    score1 = base_net(input1)
    score2 = base_net(input2)

    subtracted = subtract([score1, score2])

    model = Model(inputs=[input1, input2], outputs=subtracted)
    optimizer = Adam(learning_rate=hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2, 1e-1]))
    model.compile(optimizer=optimizer, loss=adjusted_ranknet_loss)
    return model


# Setting up Keras Tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,  # Change this based on how many trials you wish to have
    executions_per_trial=1,
    directory='random_search_dir',
    project_name='ranknet')

# Searching for the best hyperparameters
tuner.search([X1_train, X2_train], y_train,
             batch_size=16,
             epochs=50,
             validation_data=([X1_val, X2_val], y_val),
             verbose=1)

# Displaying the results of the hyperparameter search
tuner.results_summary()
# Get the best hyperparameters and build the final model
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
final_model = build_model(best_hyperparameters)

# Train the final model (you can increase epochs here if necessary)
final_model.fit([X1_train, X2_train], y_train, batch_size=16, epochs=50, validation_data=([X1_val, X2_val], y_val))
print(best_hyperparameters)

print("Best Hyperparameters:")
for key, value in best_hyperparameters.values.items():
    print(f"{key}: {value}")


team_ids = [11,
8,
3,
14,
30,
22,
27,
15,
23,
7,
6,
12,
33,
35,
16,
5,
18,
1,
19,
28

]
# Extract the base network from the RankNet model
base_net_model = final_model.layers[2]


scores = base_net_model.predict(combined_pairs_trainingSetOne)


flat_teams = team_pairs.flatten()

# Calculate the average score for each team from the scores
avg_scores = np.sum(scores, axis=1)

# Create a dictionary to hold cumulative scores for each team
team_scores_dict = {}

# Populate the dictionary with scores
for team, score in zip(flat_teams, avg_scores):
    if team in team_scores_dict:
        team_scores_dict[team] += score
    else:
        team_scores_dict[team] = score
print("what ",team_scores_dict)
# Filter out scores for specified team_ids
filtered_scores = []
for tid in team_ids:
    if tid in team_scores_dict:
        filtered_scores.append((tid, team_scores_dict[tid]))
    else:
        filtered_scores.append((tid, 0))
print(filtered_scores)


for i, (team_id, score) in enumerate(filtered_scores):
    if score < 0:
        filtered_scores[i] = (team_id, score * -1)


print("filtered scores: ", filtered_scores)

def sort_key(item):
    return item[1]

sorted_teams = sorted(filtered_scores, key=sort_key, reverse=True)

# Extract only the team IDs for final sorted list
sorted_team_ids = []
for team in sorted_teams:
    sorted_team_ids.append(team[0])

print("Teams ranked in descending order based on scores:")
print(sorted_team_ids)

with open("predicted_rankingsTrainingSetFour.txt", "w") as file:
    for score in sorted_team_ids:
        file.write(f"{score}\n")

