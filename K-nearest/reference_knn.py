# Johann Gabriel S. Lagman
# 2024-10799
#
# CMSC 170 LAB  |  EF-1L
# Exercise 06: Predicting Rainfall using KNN
# > jgsLagman_exer06.py
#
# How to use the program:
#   1. Run the program using 'python3 jgsLagman_exer06.py'
#   2. Input k and p values when prompted (k=5, p=2 for the example in the handout)
#


from pathlib import Path
from collections import Counter


#=================================================================================================|
# [1] - INPUT FILE READING FUNCTIONS
#=================================================================================================|

def readWeatherData(filename):
    filePath = Path(filename)
    lines = filePath.read_text().splitlines()

    data = []

    # skip CSV header
    for line in lines[1:]:
        stripped = line.strip()

        if stripped:
            parts = stripped.split(",")
            features = list(map(float, parts[:-1]))
            label = parts[-1]
            data.append((features, label))

    return data


def splitDataset(data):
    # use 70% of the data for training and 30% for testing
    splitIndex = int(len(data) * 0.70)

    trainingSet = data[:splitIndex]
    testingSet = data[splitIndex:]

    return trainingSet, testingSet



#=================================================================================================|
# [2] - DISTANCE FUNCTIONS (Minkowski)
#=================================================================================================|

def computeMinkowskiDistance(x,v,p):
    total = 0

    # compute distance for every dimension present in the vectors
    for xi, vi in zip(x,v):
        total += (abs(xi-vi)**p)
    distance = total **(1/p)
    
    return distance



#=================================================================================================|
# [3] - NEIGHBOR FUNCTIONS
#=================================================================================================|

def getNearestNeighbors(trainVectors,trainLabels,testVector,k,p):
    distances = []

    # compute distance from test vector to every training vector
    for vector, label in zip(trainVectors,trainLabels):
        distance = computeMinkowskiDistance(testVector,vector,p)
        distances.append((distance, label))

    # SORT by distance (ascending order)
    distances.sort(key=lambda item: item[0])
    neighbors = distances[:k]

    return neighbors



#=================================================================================================|
# [4] - CLASSIFICATION FUNCTION
#=================================================================================================|

def classifyVector(trainVectors,trainLabels,testVector,k,p):
    neighbors = getNearestNeighbors(trainVectors,trainLabels,testVector,k,p)
    votes = Counter()

    # count occurrences of labels
    for _, label in neighbors:
        votes[label] += 1

    # determine predicted class based on majority vote
    predictedClass = (votes.most_common(1)[0][0])

    return predictedClass



#=================================================================================================|
# [5] - KNN FUNCTIONS
#=================================================================================================|

def processTestSet(trainVectors,trainLabels,testVectors,testLabels,k,p):
    correctPredictions = 0
    startIndex = (len(trainVectors)+1)

    # process each test vector, compare prediction with actual label
    #   enumerate() to get index of test vector for display purposes,
    #       zip() to combine testVectors and testLabels for iteration,
    #           iterate through both lists.
    for index, (testVector,actualLabel) in enumerate(zip(testVectors, testLabels),start=startIndex):
        prediction = classifyVector(trainVectors,trainLabels,testVector,k,p)
        print(f"{index} {testVector}  ->  {prediction}")  # print test vector and prediction

        # add to correct predictions if prediction matches actual label
        if prediction == actualLabel:
            correctPredictions += 1

    # calculate accuracy as percentage
    accuracy = (correctPredictions/len(testVectors)) * 100

    return accuracy



#=================================================================================================|
# [6] - DATASET FUNCTIONS
#=================================================================================================|

def separateFeaturesAndLabels(dataset):
    vectors = []
    labels = []

    for vector, label in dataset:
        vectors.append(vector)
        labels.append(label)

    return vectors, labels



#=================================================================================================|
# [7] - READ DATASET, INPUT k AND p, DISPLAY RESULTS
#=================================================================================================|

datasetFile = ("Exercise 06 - dataset/weather_forecast_data.csv")

data = readWeatherData(datasetFile)
trainingSet, testingSet = (splitDataset(data))

trainVectors, trainLabels = (separateFeaturesAndLabels(trainingSet))
testVectors, testLabels = (separateFeaturesAndLabels(testingSet))

k = int(input("Enter k: "))
p = int(input("Enter p: "))


#-----------------------------------------------------------------------|


print("\nK-Nearest Neighbors for Weather Forecast Data")
print("JGSLagman, 2024-10799")

print(f"\nDataset size: {len(data)}")
print(f"Train size: {len(trainingSet)} (70%)")
print(f"Test size: {len(testingSet)} (30%)\n")
print(f"k={k}, p={p}\n")

print(
    "Test Index "
    "['Temperature', 'Humidity', "
    "'Wind_Speed', 'Cloud_Cover', "
    "'Pressure']  ->  Prediction"
)

accuracy = processTestSet(trainVectors,trainLabels,testVectors,testLabels,k,p)

print("-" * 50)
print(f"ACCURACY: {accuracy:.1f}%")


