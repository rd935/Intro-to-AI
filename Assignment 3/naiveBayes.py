import random
import util
import math


class NaiveBayesClassifier:

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1  # this is the smoothing parameter, ** use it in your train method **
        # self.automaticTuning = False  # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.features = list(set([f for datum in trainingData for f in datum.keys()]));

        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        result = []
        numberOfLabel = []
        numberOfSamples = len(trainingLabels)

        # total count of # in each feature
        countOfLabel = util.Counter()
        for key in trainingData[0]:
            countOfLabel[key] = 0

        for datum in trainingData:
            for key in datum:
                if datum[key] == 0:
                    countOfLabel[key] += 1

        for key in countOfLabel:
            countOfLabel[key] = countOfLabel[key] / numberOfSamples

        for label in self.legalLabels:
            result.append(util.Counter())
            numberOfLabel.append(0)
            for key in trainingData[0]:
                result[label][key] = 0

            for i in range(len(trainingLabels)):
                if int(trainingLabels[i]) == label:
                    numberOfLabel[label] += 1
                    for key in trainingData[i]:
                        if trainingData[i][key] == 0:
                            result[label][key] += 1

        countOfValidation = len(validationLabels)
        pOfLabel = []
        for label in self.legalLabels:
            count = 0
            for i in range(len(validationLabels)):
                if int(validationLabels[i]) == label:
                    count += 1
            pOfLabel.append(count / len(validationLabels))

        bestK = 1
        bestValue = 0
        for i in range(len(kgrid)):
            correct = 0
            for j in range(len(validationLabels)):
                realAnswer = int(validationLabels[j])
                probability = []
                for label in self.legalLabels:
                    logValue = math.log(pOfLabel[label])
                    for key in validationData[j]:
                        if validationData[j][key] == 0:
                            calculate1 = (result[label][key] + kgrid[i]) / (numberOfLabel[label] + 2 * kgrid[i])
                            logValue += math.log(calculate1)
                        else:
                            calculate2 = ((numberOfLabel[label] - result[label][key]) + kgrid[i]) / (
                                        numberOfLabel[label] + 2 * kgrid[i])
                            logValue += math.log(calculate2)
                    probability.append(logValue)
                answer = probability.index(max(probability))
                if answer == realAnswer:
                    correct += 1
            correct = correct / countOfValidation * 100
            if correct > bestValue:
                bestValue = correct
                bestK = kgrid[i]

        self.setSmoothing(bestK)
        self.result = result
        self.numberOfLabel = numberOfLabel
        self.pOfLabel = pOfLabel


    def classify(self, testData):

        guesses = []
        self.posteriors = []
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        logJoint = util.Counter()

        "*** YOUR CODE HERE ***"
        for label in self.legalLabels:
            logValue = math.log(self.pOfLabel[label])
            for key in datum:
                if datum[key] == 0:
                    calculate1 = (self.result[label][key] + self.k) / (self.numberOfLabel[label] + 2 * self.k)
                    logValue += math.log(calculate1)
                else:
                    calculate2 = ((self.numberOfLabel[label] - self.result[label][key]) + self.k) / (
                                self.numberOfLabel[label] + 2 * self.k)
                    logValue += math.log(calculate2)
            logJoint[label] = logValue
        return logJoint