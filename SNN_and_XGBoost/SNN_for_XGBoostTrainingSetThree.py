import numpy as np
import os
from keras_tuner import RandomSearch
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K
from SNN_and_XGBoost.creatingPairsForTrainingSetThree_SNNXGBoost import trainingSetOne
from kerastuner.engine.hypermodel import HyperModel
from sklearn.model_selection import train_test_split
import tensorflow as tf


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 5
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

class SaveArraysCallback(tf.keras.callbacks.Callback):
    def __init__(self, activation_model, data):
        super(SaveArraysCallback, self).__init__()
        self.activation_model = activation_model
        self.data = data
        print("SaveArraysCallback instantiated")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Ending epoch {epoch}")
        activations_a, activations_b = self.activation_model.predict(self.data)
        if epoch ==49:
            np.savetxt(f'activations_a_epoch_trainingSetThree{epoch}.txt', activations_a)
            np.savetxt(f'activations_b_epoch_trainingSetThree{epoch}.txt', activations_b)


    def on_train_begin(self, logs=None):
        print("Training started")

    def on_train_end(self, logs=None):
        print("Training ended")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")
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

    def activation_model(self, base_network):
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        return Model([input_a, input_b], [processed_a, processed_b])


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

best_hp = tuner.get_best_hyperparameters()[0]
model = hypermodel.build(best_hp)
activation_model_instance = hypermodel.activation_model(model.layers[-3])

model.fit(
    [train_pairs[:, 0], train_pairs[:, 1]], train_labels,
    validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
    batch_size=30, epochs=50,
    callbacks=[SaveArraysCallback(activation_model=activation_model_instance, data=[combined_pairs_trainingSetOne[:, 0], combined_pairs_trainingSetOne[:, 1]])]
)

best_model = tuner.get_best_models(num_models=1)[0]
best_model.save('best_siamese_model_forXGBoost_trainingSetThree.keras')
activations_a_before_training, activations_b_before_training = activation_model_instance.predict([combined_pairs_trainingSetOne[:, 0], combined_pairs_trainingSetOne[:, 1]])


