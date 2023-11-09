import pandas as pd
import numpy as np
from keras_tuner import RandomSearch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K
from itertools import combinations
from creatingPairsForTrainingSetThree import trainingSetOne
from kerastuner.engine.hypermodel import HyperModel
from sklearn.model_selection import train_test_split





def euclidean_distance(vects):
    x = vects[0]
    y = vects[1]
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 5
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)







# Hypermodel class for Keras Tuner
class SiameseHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        input = Input(shape=self.input_shape)
        x = Flatten()(input)
        for _ in range(hp.Int('num_layers', 1, 3)):
            x = Dense(units=hp.Int('units', min_value=10, max_value=70, step=3), activation='relu')(x)
            x = Dropout(rate=hp.Float('dropout', min_value=0.005, max_value=0.3, step=0.01))(x)
        x = Dense(5, activation='relu')(x)
        base_network = Model(inputs=input, outputs=x)

        input_a = Input(shape=self.input_shape)
        print(input_a.shape)

        input_b = Input(shape=self.input_shape)
        print(input_b.shape)
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
        # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([input_a, input_b])

        model = Model([input_a, input_b], distance)
        model.compile(loss=contrastive_loss,
                      optimizer=RMSprop(learning_rate=hp.Choice('learning_rate', [1e-1, 1e-2, 1e-3, 1e-4])))

        return model


combined_pairs_trainingSetOne, combined_labels_trainingSetOne, _, _ = trainingSetOne()
combined_labels_trainingSetOne = np.array(combined_labels_trainingSetOne, dtype=np.float32)

input_shape = combined_pairs_trainingSetOne.shape[-1]
hypermodel = SiameseHyperModel(input_shape)

tuner = RandomSearch(
    hypermodel,
    objective='loss',
    max_trials=10,
    executions_per_trial=2,
    directory='siamese_tuning',
    project_name='siamese_network'
)



(train_pairs, val_pairs, train_labels, val_labels) = train_test_split(
    combined_pairs_trainingSetOne,
    combined_labels_trainingSetOne,
    test_size=0.1,
    random_state=42
)


tuner.search(
    [train_pairs[:, 0], train_pairs[:, 1]], train_labels,
    validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
    batch_size=30, epochs=50
)


best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters()[0]
best_model.save('SNN_trainingSetThree.keras')
