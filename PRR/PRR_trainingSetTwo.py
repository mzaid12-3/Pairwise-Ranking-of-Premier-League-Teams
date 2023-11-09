from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, Dropout
from kerastuner import HyperModel, RandomSearch

from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate
import numpy as np
from creatingPairsForTrainingSetTwo import trainingSetOne
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np

combined_pairs_trainingSetOne, combined_labels_trainingSetOne, _, team_pairs = trainingSetOne()
X = combined_pairs_trainingSetOne
X = X.reshape(1140, -1)
# Parameters for the transformer block
d_model = X.shape[1]  # Dimension of the input vectors


# num_heads = 4  # Number of attention heads
# ff_dim = 256  # Hidden layer size in feed forward network inside transformer
# dropout_rate = 0.1  # Dropout rate
# num_transformer_blocks = 2  # Number of transformer blocks

# Define a Transformer block
def transformer_block(inputs, num_heads, ff_dim, dropout_rate):
    # Add a sequence dimension
    inputs = tf.expand_dims(inputs, 1)

    # Multi-head Self Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    out1 = Add()([inputs, attention_output])
    out1 = LayerNormalization(epsilon=1e-6)(out1)

    # Feed-forward network
    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)  # Set the output dimension to match the input dimension
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = Add()([out1, ffn_output])
    out2 = LayerNormalization(epsilon=1e-6)(out2)

    # Remove the sequence dimension
    out2 = tf.squeeze(out2, 1)

    return out2


class TransformerDualInputHyperModel(HyperModel):
    def __init__(self, input_shape1, input_shape2):
        self.input_shape1 = input_shape1
        self.input_shape2 = input_shape2

    def build(self, hp):
        input1 = tf.keras.Input(shape=self.input_shape1, name="input1")
        input2 = tf.keras.Input(shape=self.input_shape2, name="input2")

        flattened_input1 = tf.keras.layers.Flatten()(input1)  # Flattening the player pairs

        merged_input = tf.keras.layers.Concatenate(axis=-1)([flattened_input1, input2])

        # Hyperparameters
        # num_heads = 4  # Number of attention heads
        # ff_dim = 256  # Hidden layer size in feed forward network inside transformer
        # dropout_rate = 0.1  # Dropout rate
        # num_transformer_blocks = 2  # Number of transformer blocks

        num_heads = hp.Int('num_heads', 2, 4, step=2)
        ff_dim = hp.Int('ff_dim', 128, 256, step=64)
        dropout_rate = hp.Float('dropout_rate', 0.0, 0.2, step=0.1)
        num_transformer_blocks = hp.Int('num_transformer_blocks', 1, 2)

        x = merged_input
        for _ in range(num_transformer_blocks):
            x = transformer_block(x, num_heads, ff_dim, dropout_rate)

        # Here, you can add your output layers based on the task, for example:
        outputs = tf.keras.layers.Dense(30, activation='softmax')(x)  # Since there are 30 classes

        model = tf.keras.Model(inputs=[input1, input2], outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model


# Assume X_train is your input data
encoded_labels = to_categorical(combined_labels_trainingSetOne, num_classes=30)
team_pairs = np.array(team_pairs)

(train_pairs, val_pairs, train_teams, val_teams, train_labels, val_labels) = train_test_split(
    combined_pairs_trainingSetOne,
    team_pairs,  # Include team_pairs in the split
    encoded_labels,
    test_size=0.1,
    random_state=42
)

input_shape1 = (2, 51)
input_shape2 = (2,)
dual_hypermodel = TransformerDualInputHyperModel(input_shape1, input_shape2)

tuner = RandomSearch(
    dual_hypermodel,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1
)

tuner.search(
    [train_pairs, train_teams], train_labels,
    validation_data=([val_pairs, val_teams], val_labels),
    batch_size=30, epochs=20
)

# 1. Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]

# 2. Print the best hyperparameters
print("Best hyperparameters found:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")




best_hp = tuner.get_best_hyperparameters()[0]


best_model = dual_hypermodel.build(best_hp)


combined_pairs_trainingSetOne, combined_labels_trainingSetOne, _, team_pairs = trainingSetOne()
encoded_labels = to_categorical(combined_labels_trainingSetOne, num_classes=30)
print(combined_labels_trainingSetOne.shape)
team_pairs = np.array(team_pairs)
best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
best_model.fit([combined_pairs_trainingSetOne, team_pairs], encoded_labels, epochs=20, batch_size=30)



files = [
    "../My_datasets/2012_13.xlsx",
    "../My_datasets/2013_14.xlsx",
    "../My_datasets/2014_15.xlsx",
    "../My_datasets/2015_16.xlsx",
    "../My_datasets/2016_17.xlsx",
    "../My_datasets/2017_18.xlsx",
    "../My_datasets/2018_19.xlsx"

]

season_standings = []

# Reading and processing each Excel file
for file in files:
    df = pd.read_excel(file)
    # Extract the team identifier and its position based on index
    standings_dict = {row['team identifier']: idx + 1 for idx, row in df.iterrows()}
    season_standings.append(standings_dict)

print(season_standings)
predictions = best_model.predict([combined_pairs_trainingSetOne, team_pairs])
# Team identifiers
team_ids = [8,
11,
22,
3,
15,
14,
7,
30,
10,
1,
16,
5,
6,
12,
23,
27,
35,
2,
19,
28

]
# print(team_pairs[:5])
# Extract and aggregate scores for individual teams from pair predictions
team_scores = {team_id: 0 for team_id in team_ids}
for idx, pair in enumerate(team_pairs):
    team1 = pair[0]
    team2 = pair[1]

    try:
        team_scores[team1] += predictions[idx]
        print(team_scores[team1], ":", predictions[idx])
        team_scores[team2] += predictions[idx]
    except KeyError:
        print(f"Team identifier {team2} not found in team_scores dictionary.")
        if team2 not in team_ids:
            print(f"Team identifier {team2} is also not in the original team_ids list.")

# Normalize these aggregated scores to get average probabilities
total_matches = len(combined_pairs_trainingSetOne) / 2  # Each team appears in half the pairs
average_probabilities = {team_id: score / total_matches for team_id, score in team_scores.items()}

# Weights for standings
season_weights = [1, 2, 3, 4, 5, 6, 7]
weighted_standings = []
for team_id in team_ids:
    team_weighted_standing = sum(
        season_standings[season].get(team_id, len(team_ids) + 1) * season_weights[season] for season in range(7))
    weighted_standings.append(team_weighted_standing)

# Normalize the weighted standings
max_weighted_value = len(team_ids) * sum(season_weights)
normalized_standings = [(max_weighted_value - standing) / max_weighted_value for standing in weighted_standings]

# Weights for combining average probabilities and standings
w1 = 0.7
w2 = 0.3

# Adjust the average probabilities
adjusted_scores = {team_id: w1 * average_probabilities[team_id] + w2 * normalized_standings[team_ids.index(team_id)] for
                   team_id in team_ids}
for team_id, score in adjusted_scores.items():
    print(f"Team ID: {team_id}, Score: {score}")

# Sort teams based on adjusted scores
scalar_scores = {team_id: np.sum(score) for team_id, score in adjusted_scores.items()}

# Sort teams based on these scalar scores
sorted_teams = [team for team, _ in sorted(scalar_scores.items(), key=lambda x: x[1], reverse=True)]
sorted_teams_scores = sorted(scalar_scores.items(), key=lambda x: x[1], reverse=True)

print(sorted_teams_scores)
print(sorted_teams)

with open('predicted_rankingsTrainingSetTwo.txt', 'w') as f:
    for team in sorted_teams:
        f.write(f"{team}\n")
