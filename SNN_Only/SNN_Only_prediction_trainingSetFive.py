from tensorflow.keras.models import load_model
from SNN_Only.SNN_Only_trainingSetFive import   contrastive_loss
from creatingPairsForTrainingSetFive import trainingSetOne


def get_similarity_scores():
    # Load the saved model
    model = load_model('SNN_trainingSetFive.keras',
                       custom_objects={'contrastive_loss': contrastive_loss},
                       safe_mode=False)

    # Load the data on which you want to predict
    combined_pairs_trainingSetOne, combined_labels_trainingSetOne, _, _ = trainingSetOne()
    # with open("team_pairsTrainingSetOne.txt", "w") as file:
    #     for pair in combined_pairs_trainingSetOne:
    #         file.write(f"{pair[0]} {pair[1]}\n")
    # Predict similarity scores
    similarity_scores = model.predict([combined_pairs_trainingSetOne[:, 0], combined_pairs_trainingSetOne[:, 1]])

    return similarity_scores


if __name__ == '__main__':
    scores = get_similarity_scores()
    print(scores)
    with open("SNN_Only_similarity_scoresTrainingSetFive.txt", "w") as file:
        for score in scores:
            file.write(f"{score}\n")

