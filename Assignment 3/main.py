import naiveBayes
import perceptron
import numpy as np
import util
import os
import random
import time
import matplotlib.pyplot as plt

DIGIT_PIC_WIDTH = 28
DIGIT_PIC_HEIGHT = 28
FACE_PIC_WIDTH = 60
FACE_PIC_HEIGHT = 70

def basicFeatureExtractionDigit(pic):
    features = util.Counter()
    for x in range(DIGIT_PIC_WIDTH):
        for y in range(DIGIT_PIC_HEIGHT):
            features[(x, y)] = 1 if pic.getPixel(x, y) > 0 else 0
    return features

def basicFeatureExtractionFace(pic):
    features = util.Counter()
    for x in range(FACE_PIC_WIDTH):
        for y in range(FACE_PIC_HEIGHT):
            features[(x, y)] = 1 if pic.getPixel(x, y) > 0 else 0
    return features

def plot_metrics(training_data_usage, training_times, error_rates, std_devs, validation_accuracies, test_accuracies, classifier_name, data_type):
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.plot(training_data_usage, training_times, marker='o')
    plt.title(f"{classifier_name} ({data_type}): Training Time")
    plt.xlabel("Training Data Usage (%)")
    plt.ylabel("Time (s)")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.errorbar(training_data_usage, error_rates, yerr=std_devs, marker='o', label='Error Rate')
    plt.title(f"{classifier_name} ({data_type}): Error Rate")
    plt.xlabel("Training Data Usage (%)")
    plt.ylabel("Error Rate (%)")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(training_data_usage, validation_accuracies, marker='o')
    plt.title(f"{classifier_name} ({data_type}): Validation Accuracy")
    plt.xlabel("Training Data Usage (%)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(training_data_usage, test_accuracies, marker='o')
    plt.title(f"{classifier_name} ({data_type}): Test Accuracy")
    plt.xlabel("Training Data Usage (%)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{classifier_name}_{data_type}_metrics.png")
    plt.close()

if __name__ == '__main__':
    np.set_printoptions(linewidth=400)
    classifiers = {
        "Naive Bayes": lambda labels: naiveBayes.NaiveBayesClassifier(labels),
        "Perceptron": lambda labels: perceptron.PerceptronClassifier(labels, 10)
    }

    data_types = ["face", "digit"]

    TRAINING_DATA_USAGE_SET = [round(i * 0.1, 1) for i in range(1, 11)]

    for data_type in data_types:
        if data_type == "digit":
            legalLabels = range(10)
            training_file = "data/digitdata/trainingimages"
            training_labels_file = "data/digitdata/traininglabels"
            validation_file = "data/digitdata/validationimages"
            validation_labels_file = "data/digitdata/validationlabels"
            test_file = "data/digitdata/testimages"
            test_labels_file = "data/digitdata/testlabels"
            pic_width, pic_height = DIGIT_PIC_WIDTH, DIGIT_PIC_HEIGHT
            feature_extraction = basicFeatureExtractionDigit
        else:
            legalLabels = range(2)
            training_file = "data/facedata/facedatatrain"
            training_labels_file = "data/facedata/facedatatrainlabels"
            validation_file = "data/facedata/facedatavalidation"
            validation_labels_file = "data/facedata/facedatavalidationlabels"
            test_file = "data/facedata/facedatatest"
            test_labels_file = "data/facedata/facedatatestlabels"
            pic_width, pic_height = FACE_PIC_WIDTH, FACE_PIC_HEIGHT
            feature_extraction = basicFeatureExtractionFace

        for classifier_name, classifier_func in classifiers.items():
            classifier = classifier_func(legalLabels)

            training_times = []
            error_rates = []
            std_devs = []
            validation_accuracies = []
            test_accuracies = []

            for TRAINING_DATA_USAGE in TRAINING_DATA_USAGE_SET:
                print(f"Starting training with {int(TRAINING_DATA_USAGE * 100)}% of the data ({classifier_name}, {data_type})...")

                TRAINING_SET_SIZE = int(len(open(training_labels_file, "r").readlines()) * TRAINING_DATA_USAGE)
                VALIDATION_SET_SIZE = int(len(open(validation_labels_file, "r").readlines()))
                TEST_SET_SIZE = int(len(open(test_labels_file, "r").readlines()))

                randomOrder = random.sample(range(len(open(training_labels_file, "r").readlines())), TRAINING_SET_SIZE)

                rawTrainingData = util.loadDataFileRandomly(training_file, randomOrder, pic_width, pic_height)
                trainingLabels = util.loadLabelFileRandomly(training_labels_file, randomOrder)

                rawValidationData = util.loadDataFile(validation_file, VALIDATION_SET_SIZE, pic_width, pic_height)
                validationLabels = util.loadLabelFile(validation_labels_file, VALIDATION_SET_SIZE)

                rawTestData = util.loadDataFile(test_file, TEST_SET_SIZE, pic_width, pic_height)
                testLabels = util.loadLabelFile(test_labels_file, TEST_SET_SIZE)

                trainingData = list(map(feature_extraction, rawTrainingData))
                validationData = list(map(feature_extraction, rawValidationData))
                testData = list(map(feature_extraction, rawTestData))

                start_time = time.time()
                classifier.train(trainingData, trainingLabels, validationData, validationLabels)
                end_time = time.time()

                print(f"Training completed for {int(TRAINING_DATA_USAGE * 100)}% of the data. Time taken: {end_time - start_time:.2f} seconds.")
                training_times.append(end_time - start_time)

                print("Starting testing phase...")
                guesses = classifier.classify(testData)
                correct = sum(guesses[i] == int(testLabels[i]) for i in range(len(testLabels)))
                test_accuracy = 100.0 * correct / len(testLabels)
                test_accuracies.append(test_accuracy)

                validation_guesses = classifier.classify(validationData)
                validation_correct = sum(validation_guesses[i] == int(validationLabels[i]) for i in range(len(validationLabels)))
                validation_accuracy = 100.0 * validation_correct / len(validationLabels)
                validation_accuracies.append(validation_accuracy)

                test_accuracies_run = []
                for run in range(5):
                    randomOrder = random.sample(range(len(trainingLabels)), TRAINING_SET_SIZE)
                    rawTrainingData = util.loadDataFileRandomly(training_file, randomOrder, pic_width, pic_height)
                    trainingData = list(map(feature_extraction, rawTrainingData))
                    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
                    guesses = classifier.classify(testData)
                    correct = sum(guesses[i] == int(testLabels[i]) for i in range(len(testLabels)))
                    test_accuracy = 100.0 * correct / len(testLabels)
                    test_accuracies_run.append(100.0 - test_accuracy)

                error_rates.append(np.mean(test_accuracies_run))
                std_devs.append(np.std(test_accuracies_run))
                print(f"Testing completed for {int(TRAINING_DATA_USAGE * 100)}% of the data.")

            plot_metrics(
                [x * 100 for x in TRAINING_DATA_USAGE_SET],
                training_times,
                error_rates,
                std_devs,
                validation_accuracies,
                test_accuracies,
                classifier_name,
                data_type
            )
