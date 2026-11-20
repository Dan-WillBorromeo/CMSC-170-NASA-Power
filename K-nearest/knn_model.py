# Module: K-Nearest Neighbors (Regression & Classification)
# > K-nearest/knn_model.py
#
# Disclosures:
#   - No AI tools were used in the development of this script.
#   - Adapted from original Exercise 06 implementation (reference_knn.py).
#
# Author: Lagman, Johann Gabriel S.
# Date Created  : 12 November 2026
# Date Modified : 19 November 2026
# Version       : 1.0.1

from pathlib import Path
from collections import Counter
import math


# =================================================================================================|
# [1] - DATA LOADING & FEATURE CONSTRUCTION (Site Tables & NASA POWER Weather CSVs)
# =================================================================================================|

def readSiteTableData(rawSiteCsv, zscoreSiteCsv, targetName="irr_mean", taskType="regression"):
    # Load pre-standardized climate features from site_table_zscore.csv and pair them
    #   with target values from site_table.csv matched by cell_key.
    #
    # site_table.csv columns (14):
    #   [0] cell_key, [1] lat, [2] lon, [3] n_farms, [4] total_mw,
    #   [5] irr_mean, [6] irr_std, [7] irr_season_gap, [8] clear_day_frac,
    #   [9] temp_mean, [10] rh_mean, [11] rain_mean, [12] rainy_day_frac, [13] wind_mean
    #
    # site_table_zscore.csv data row format (10 columns):
    #   [0] cell_key,
    #   [1] irr_mean_z, [2] irr_std_z, [3] irr_season_gap_z, [4] clear_day_frac_z,
    #   [5] temp_mean_z, [6] rh_mean_z, [7] rain_mean_z, [8] rainy_day_frac_z, [9] wind_mean_z

    rawLines = Path(rawSiteCsv).read_text().splitlines()
    zLines = Path(zscoreSiteCsv).read_text().splitlines()

    # Map cell_key -> raw row dictionary from site_table.csv
    rawHeader = rawLines[0].strip().split(",")
    rawLookup = {}

    for line in rawLines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(",")
        cellKey = parts[0]
        rawLookup[cellKey] = {rawHeader[i]: float(parts[i]) for i in range(1, len(rawHeader))}

    # Feature index mapping inside z-score data rows (1-based after cell_key)
    zFeatureMap = {
        "irr_mean": 1,
        "irr_std": 2,
        "irr_season_gap": 3,
        "clear_day_frac": 4,
        "temp_mean": 5,
        "rh_mean": 6,
        "rain_mean": 7,
        "rainy_day_frac": 8,
        "wind_mean": 9
    }

    dataset = []
    cellKeys = []

    for line in zLines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(",")
        cellKey = parts[0]
        if cellKey not in rawLookup:
            continue

        zValues = list(map(float, parts[1:]))  # 9 Z-scored climate features

        # Exclude target feature from X if predicting one of the 9 climate variables
        if targetName in zFeatureMap:
            excludeIdx = zFeatureMap[targetName] - 1
            features = [val for idx, val in enumerate(zValues) if idx != excludeIdx]
        else:
            features = zValues

        rawTargetVal = rawLookup[cellKey][targetName]
        target = _formatSiteTarget(rawTargetVal, targetName, taskType)

        dataset.append((features, target))
        cellKeys.append(cellKey)

    return dataset, cellKeys


def _formatSiteTarget(rawValue, targetName, taskType):
    # Format site-level target variable for regression or classification.

    if taskType == "regression":
        return rawValue
    elif taskType == "classification":
        if targetName == "irr_mean":
            if rawValue < 4.85:
                return "Low"
            elif rawValue <= 5.08:
                return "Moderate"
            else:
                return "High"
        elif targetName == "total_mw":
            return "High_Capacity" if rawValue >= 50.0 else "Standard_Capacity"
        else:
            return str(rawValue)
    else:
        raise ValueError("taskType must be 'regression' or 'classification'")


def readPowerWeatherFile(filename, lagDays=1, taskType="regression"):
    # Read a single NASA POWER weather CSV file (e.g., 'weather/14.000_121.250.csv'),
    #   skips metadata headers, filters missing (-999) rows, and builds feature vectors.
    #
    # Columns in NASA POWER CSV:
    #   [0] YEAR, [1] MO, [2] DY, [3] ALLSKY_SFC_SW_DWN,
    #   [4] T2M, [5] T2M_RANGE, [6] RH2M, [7] PS, [8] WS10M, [9] PRECTOTCORR
    
    filePath = Path(filename)
    lines = filePath.read_text().splitlines()

    rawRows = []
    inDataSection = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("YEAR,MO,DY"):
            inDataSection = True
            continue

        if inDataSection:
            parts = stripped.split(",")
            values = list(map(float, parts))

            # Filter out NASA POWER missing data flag (-999)
            if -999.0 in values or -999 in values:
                continue

            rawRows.append(values)

    dataset = []


    # Build lagged features (predict day t using previous lagDays)
    if lagDays > 0:
        for t in range(lagDays, len(rawRows)):
            features = []
            # Include month as seasonal indicator AND past lagDays weather & irradiance
            features.append(rawRows[t][1])  # MO (Month)

            for lag in range(1, lagDays + 1):
                prevRow = rawRows[t - lag]
                # Append ALLSKY_SFC_SW_DWN, T2M, T2M_RANGE, RH2M, PS, WS10M, PRECTOTCORR
                features.extend(prevRow[3:])

            targetIrradiance = rawRows[t][3]
            target = _formatTarget(targetIrradiance, taskType)
            dataset.append((features, target))
    else:
        # Same-day estimation (predict day t irradiance from day t surface weather)
        for row in rawRows:
            features = [row[1]] + row[4:]  # MO + T2M, T2M_RANGE, RH2M, PS, WS10M, PRECTOTCORR
            targetIrradiance = row[3]
            target = _formatTarget(targetIrradiance, taskType)
            dataset.append((features, target))

    return dataset



def readMultipleWeatherFiles(weatherDir, cellKeys=None, lagDays=1, taskType="regression"):
    # Load AND combine data across multiple coordinate CSVs inside the 'weather/' directory.
    # If cellKeys is provided (ex. from Philippine-Commercial-Solar-Farms.csv), load only those.
    
    dirPath = Path(weatherDir)
    combinedData = []

    if cellKeys:
        uniqueKeys = sorted(set(cellKeys))
        files = [dirPath / f"{key}.csv" for key in uniqueKeys if (dirPath / f"{key}.csv").exists()]
    else:
        files = sorted(dirPath.glob("*.csv"))

    for filePath in files:
        fileData = readPowerWeatherFile(filePath, lagDays=lagDays, taskType=taskType)
        combinedData.extend(fileData)

    return combinedData



def _formatTarget(irradianceValue, taskType):
    # Format target as continuous float (regression) or discrete tier label (classification).
    
    if taskType == "regression":
        return irradianceValue
    elif taskType == "classification":
        if irradianceValue < 3.5:
            return "Low"
        elif irradianceValue <= 5.5:
            return "Moderate"
        else:
            return "Optimal"
    else:
        raise ValueError("taskType must be 'regression' or 'classification'")





# =================================================================================================|
# [2] - DATASET SPLITTING & FEATURE SCALING (From Scratch)
# =================================================================================================|

def splitDataset(data, trainRatio=0.80):
    # Split dataset into training and testing sets (default 80/20 split).
    
    splitIndex = int(len(data) * trainRatio)
    trainingSet = data[:splitIndex]
    testingSet = data[splitIndex:]
    return trainingSet, testingSet


def separateFeaturesAndLabels(dataset):
    # Separate list of (features, target) tuples into parallel vectors and targets lists.
    vectors = []
    targets = []

    for vector, target in dataset:
        vectors.append(vector)
        targets.append(target)

    return vectors, targets


def computeMinMaxBounds(trainVectors):
    # Compute column-wise minimum and maximum bounds FROM training vectors ONLY.
    
    numFeatures = len(trainVectors[0])
    minVals = [min(row[col] for row in trainVectors) for col in range(numFeatures)]
    maxVals = [max(row[col] for row in trainVectors) for col in range(numFeatures)]
    return minVals, maxVals


def normalizeVectors(vectors, minVals, maxVals):
    # Apply Min-Max scaling [0, 1] using precomputed training bounds.
    
    scaledVectors = []
    for row in vectors:
        scaledRow = []
        for col, val in enumerate(row):
            denom = maxVals[col] - minVals[col]
            scaledVal = (val - minVals[col]) / denom if denom != 0 else 0.0
            scaledRow.append(scaledVal)
        scaledVectors.append(scaledRow)
    return scaledVectors





# =================================================================================================|
# [3] - DISTANCE & NEIGHBOR FUNCTIONS (Minkowski)
# =================================================================================================|

def computeMinkowskiDistance(x, v, p):
    # Compute Minkowski distance (p=1: Manhattan, p=2: Euclidean).
    
    total = 0.0
    for xi, vi in zip(x, v):
        total += (abs(xi - vi) ** p)
    return total ** (1.0 / p)


def getNearestNeighbors(trainVectors, trainTargets, testVector, k, p):
    # Find the k nearest training neighbors sorted ascending by distance.
    
    distances = []

    for vector, target in zip(trainVectors, trainTargets):
        distance = computeMinkowskiDistance(testVector, vector, p)
        distances.append((distance, target))

    distances.sort(key=lambda item: item[0])
    return distances[:k]





# =================================================================================================|
# [4] - PREDICTION FUNCTIONS (Regression & Classification)
# =================================================================================================|

def predictRegression(trainVectors, trainTargets, testVector, k, p, weighted=False):
    # Predict continuous solar irradiance using average (or distance-weighted average) of k neighbors.
    
    neighbors = getNearestNeighbors(trainVectors, trainTargets, testVector, k, p)

    if not weighted:
        total = sum(target for _, target in neighbors)
        return total / len(neighbors)

    weightedSum = 0.0
    weightTotal = 0.0
    for dist, target in neighbors:
        weight = 1.0 / (dist + 1e-8)
        weightedSum += weight * target
        weightTotal += weight

    return weightedSum / weightTotal


def classifyVector(trainVectors, trainLabels, testVector, k, p):
    # Predict discrete class label using majority vote (from reference_knn.py).
    
    neighbors = getNearestNeighbors(trainVectors, trainLabels, testVector, k, p)
    votes = Counter()

    for _, label in neighbors:
        votes[label] += 1

    return votes.most_common(1)[0][0]





# =================================================================================================|
# [5] - EVALUATION METRICS & BATCH TESTING
# =================================================================================================|

def computeRMSE(actual, predicted):
    # Root Mean Squared Error (RMSE) for regression.
    
    sqErrors = [(a - pred) ** 2 for a, pred in zip(actual, predicted)]
    return math.sqrt(sum(sqErrors) / len(actual))


def computeMAE(actual, predicted):
    # Mean Absolute Error (MAE) for regression.
    
    absErrors = [abs(a - pred) for a, pred in zip(actual, predicted)]
    return sum(absErrors) / len(actual)


def computeRSquared(actual, predicted):
    # Coefficient of Determination (R^2) for regression.
    
    meanActual = sum(actual) / len(actual)
    ssRes = sum((a - pred) ** 2 for a, pred in zip(actual, predicted))
    ssTot = sum((a - meanActual) ** 2 for a in actual)
    return 1.0 - (ssRes / ssTot) if ssTot != 0 else 0.0


def computeAccuracy(actual, predicted):
    # Classification accuracy percentage.
    
    correct = sum(1 for a, pred in zip(actual, predicted) if a == pred)
    return (correct / len(actual)) * 100.0


def evaluateKNN(trainVectors, trainTargets, testVectors, testTargets, k=5, p=2, taskType="regression", weighted=False):
    # Run batch inference on testVectors and returns predictions AND metric dictionary.
    
    predictions = []

    for testVector in testVectors:
        if taskType == "regression":
            pred = predictRegression(trainVectors, trainTargets, testVector, k, p, weighted=weighted)
        else:
            pred = classifyVector(trainVectors, trainTargets, testVector, k, p)
        predictions.append(pred)

    if taskType == "regression":
        metrics = {
            "RMSE": computeRMSE(testTargets, predictions),
            "MAE": computeMAE(testTargets, predictions),
            "R2": computeRSquared(testTargets, predictions)
        }
    else:
        metrics = {
            "Accuracy": computeAccuracy(testTargets, predictions)
        }

    return predictions, metrics


def evaluateLOOCV(dataset, k=3, p=2, taskType="regression", weighted=True):
    # Run Leave-One-Out Cross-Validation (LOOCV) for small cell-level site tables (N=39).

    vectors, targets = separateFeaturesAndLabels(dataset)
    predictions = []

    for i in range(len(vectors)):
        trainX = vectors[:i] + vectors[i + 1:]
        trainY = targets[:i] + targets[i + 1:]
        testX = vectors[i]

        if taskType == "regression":
            pred = predictRegression(trainX, trainY, testX, k, p, weighted=weighted)
        else:
            pred = classifyVector(trainX, trainY, testX, k, p)
        predictions.append(pred)

    if taskType == "regression":
        metrics = {
            "RMSE": computeRMSE(targets, predictions),
            "MAE": computeMAE(targets, predictions),
            "R2": computeRSquared(targets, predictions)
        }
    else:
        metrics = {
            "Accuracy": computeAccuracy(targets, predictions)
        }

    return predictions, metrics


