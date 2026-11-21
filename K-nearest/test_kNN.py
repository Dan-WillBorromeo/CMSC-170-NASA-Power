# Module: Test Runner for K-Nearest Neighbors
# > K-nearest/test_kNN.py
#
# Disclosures:
#   - Gemini 3.1 Pro assisted in debugging and code cleanup/shortening.
#
# Author: Lagman, Johann Gabriel S.
# Date Created  : 18 November 2026
# Date Modified : 21 November 2026
# Version       : 2.2.0

from pathlib import Path
import json
import math
import random

from knn_model import (
    getNearestNeighbors,
    classifyVector
)


# =================================================================================================|
# [1] - CLIMATE PROFILE FEATURE EXTRACTION
# =================================================================================================|

FEATURES = ['ghi', 'ghi_iav', 'ghi_season', 'ghi_low', 't', 'trange', 'rh', 'ps', 'ws', 'ws99', 'rain', 'wet']


def readPowerRaw(filepath):
    # Read NASA POWER CSV rows (YEAR < 2026) and filter out -999 missing values.

    lines = Path(filepath).read_text().splitlines()
    startIdx = next(i for i, line in enumerate(lines) if line.startswith("-END HEADER-")) + 1
    cols = lines[startIdx].split(",")

    rows = []
    for line in lines[startIdx + 1:]:
        if not line.strip():
            continue
        rowDict = dict(zip(cols, map(float, line.split(","))))
        if rowDict["YEAR"] < 2026 and all(val != -999 for val in rowDict.values()):
            rows.append(rowDict)

    return rows


def _mean(vals):
    return sum(vals) / len(vals)


def _std(vals):
    m = _mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def _quantile(vals, q):
    s = sorted(vals)
    return s[min(int(q * len(s)), len(s) - 1)]


def extractClimateProfile(filepath):
    # Compute the 12-feature climate summary vector for a single grid cell CSV.

    d = readPowerRaw(filepath)
    n = len(d)
    g = [r["ALLSKY_SFC_SW_DWN"] for r in d]

    yearly, monthly, yearlyRain = {}, {}, {}
    for r in d:
        yearly.setdefault(r["YEAR"], []).append(r["ALLSKY_SFC_SW_DWN"])
        monthly.setdefault(r["MO"], []).append(r["ALLSKY_SFC_SW_DWN"])
        yearlyRain[r["YEAR"]] = yearlyRain.get(r["YEAR"], 0.0) + r["PRECTOTCORR"]

    ym = [_mean(v) for v in yearly.values()]
    mm = [_mean(v) for v in monthly.values()]

    return [
        _mean(g),
        _std(ym),
        (max(mm) - min(mm)) / _mean(mm),
        sum(x < 2 for x in g) / n,
        _mean([r["T2M"] for r in d]),
        _mean([r["T2M_RANGE"] for r in d]),
        _mean([r["RH2M"] for r in d]),
        _mean([r["PS"] for r in d]),
        _mean([r["WS10M"] for r in d]),
        _quantile([r["WS10M"] for r in d], 0.99),
        _mean(list(yearlyRain.values())),
        sum(r["PRECTOTCORR"] > 1 for r in d) / n
    ]


def loadClimateFolder(folderPath):
    # Load all grid cell CSVs inside a folder sorted by filename, returning (cellKey, features).

    csvFiles = sorted(Path(folderPath).glob("*.csv"))
    return [(p.stem, extractClimateProfile(p)) for p in csvFiles]


def computeFoldScaler(trainX):
    # Compute column-wise min and max bounds from training fold only.

    lo = [min(col) for col in zip(*trainX)]
    hi = [max(col) for col in zip(*trainX)]
    return lo, hi


def scaleVector(x, lo, hi):
    # Scale features to [0.1, 0.9] using training fold bounds (matches train.py).

    return [
        min(1.0, max(0.0, 0.1 + 0.8 * (v - l) / (h - l))) if h > l else 0.5
        for v, l, h in zip(x, lo, hi)
    ]





# =================================================================================================|
# [2] - SOFT PROBABILITY & METRICS (ROC-AUC & Accuracy)
# =================================================================================================|

def predictPositiveScore(trainX, trainY, testX, k, p, weighted=True):
    # Return continuous score in [0, 1] for ROC-AUC computation based on k nearest neighbors.

    neighbors = getNearestNeighbors(trainX, trainY, testX, k, p)

    if not weighted:
        return sum(label for _, label in neighbors) / len(neighbors)

    posWeight = 0.0
    totalWeight = 0.0
    for dist, label in neighbors:
        w = 1.0 / (dist + 1e-8)
        if label == 1:
            posWeight += w
        totalWeight += w

    return posWeight / totalWeight


def computeAUC(scores, labels):
    # Compute ROC-AUC using Mann-Whitney U statistic (matches train.py).

    posScores = [s for s, l in zip(scores, labels) if l == 1]
    negScores = [s for s, l in zip(scores, labels) if l == 0]
    if not posScores or not negScores:
        return 0.0
    return sum((p > n) + 0.5 * (p == n) for p in posScores for n in negScores) / (len(posScores) * len(negScores))





# =================================================================================================|
# [3] - 5-FOLD STRATIFIED CROSS-VALIDATION BENCHMARK & MODEL ARTIFACT EXPORT
# =================================================================================================|

def runBenchmark(posDir, negDir, outputsDir, foldsCount=5, seed=170):
    # Evaluate k-NN on the exact same 5-fold stratified split (seed=170) as Neural Network.

    posSamples = loadClimateFolder(posDir)
    negSamples = loadClimateFolder(negDir)

    print("=" * 76)
    print("5-FOLD STRATIFIED CV: SOLAR FARM (1) vs. NON-SOLAR FARM (0)")
    print("=" * 76)
    print(f"Positive Cells (SF)  : {len(posSamples)}")
    print(f"Negative Cells (NSF) : {len(negSamples)}")
    print(f"Features             : {len(FEATURES)} ({', '.join(FEATURES)})\n")

    rng = random.Random(seed)
    data = [(cell, x, 1) for cell, x in posSamples] + [(cell, x, 0) for cell, x in negSamples]
    rng.shuffle(data)

    idx = list(range(len(data)))
    posIdx = [i for i in idx if data[i][2] == 1]
    negIdx = [i for i in idx if data[i][2] == 0]
    folds = [posIdx[i::foldsCount] + negIdx[i::foldsCount] for i in range(foldsCount)]

    print(f"{'k':<4} | {'p':<3} | {'Weighted':<9} | {'5-Fold AUC':<20} | {'5-Fold Accuracy':<20}")
    print("-" * 68)

    summaryRows = []
    bestKey = (-1.0, -1.0)
    bestParams = {}
    bestOofRows = []

    for p in [1, 2]:
        for k in [3, 5, 7, 9]:
            for weighted in [False, True]:
                foldAucs = []
                foldAccs = []
                oofRows = []

                for foldNum, f in enumerate(folds, start=1):
                    foldSet = set(f)
                    trainSplit = [data[i] for i in idx if i not in foldSet]
                    testSplit = [data[i] for i in f]

                    lo, hi = computeFoldScaler([x for _, x, _ in trainSplit])
                    trainX = [scaleVector(x, lo, hi) for _, x, _ in trainSplit]
                    trainY = [y for _, _, y in trainSplit]

                    testX = [scaleVector(x, lo, hi) for _, x, _ in testSplit]
                    testY = [y for _, _, y in testSplit]

                    scores = [predictPositiveScore(trainX, trainY, x, k, p, weighted=weighted) for x in testX]
                    if weighted:
                        preds = [1 if s > 0.5 else 0 for s in scores]
                    else:
                        preds = [classifyVector(trainX, trainY, x, k, p) for x in testX]

                    for (cell, _, actual), score, pred in zip(testSplit, scores, preds):
                        oofRows.append((foldNum, cell, actual, round(score, 6), pred))

                    foldAucs.append(computeAUC(scores, testY))
                    foldAccs.append(sum(pred == actual for pred, actual in zip(preds, testY)) / len(testY))

                aucMean, aucStd = _mean(foldAucs), _std(foldAucs)
                accMean, accStd = _mean(foldAccs), _std(foldAccs)

                summaryRows.append({
                    "k": k,
                    "p": p,
                    "weighted": weighted,
                    "auc_mean": aucMean,
                    "auc_std": aucStd,
                    "acc_mean": accMean,
                    "acc_std": accStd
                })

                if (aucMean, accMean) > bestKey:
                    bestKey = (aucMean, accMean)
                    bestParams = {"k": k, "p": p, "weighted": weighted}
                    bestOofRows = oofRows

                aucStr = f"{aucMean:.3f} ± {aucStd:.3f}"
                accStr = f"{accMean:.3f} ± {accStd:.3f}"
                print(f"{k:<4} | {p:<3} | {str(weighted):<9} | {aucStr:<20} | {accStr:<20}")

    # Gemini changed: added standardized CSV export (knn_cv_summary.csv & knn_oof_predictions.csv) to Outputs/metrics/
    metricsDir = Path(outputsDir) / "metrics"
    metricsDir.mkdir(parents=True, exist_ok=True)

    summaryPath = metricsDir / "knn_cv_summary.csv"
    with summaryPath.open("w") as fh:
        fh.write("model,params,auc_mean,auc_std,acc_mean,acc_std\n")
        for r in summaryRows:
            fh.write(
                f"kNN,\"k={r['k']},p={r['p']},weighted={r['weighted']}\","
                f"{r['auc_mean']:.6f},{r['auc_std']:.6f},{r['acc_mean']:.6f},{r['acc_std']:.6f}\n"
            )

    oofPath = metricsDir / "knn_oof_predictions.csv"
    with oofPath.open("w") as fh:
        fh.write("fold,cell_key,actual,pred_score,pred_label\n")
        for row in bestOofRows:
            fh.write(",".join(map(str, row)) + "\n")

    # Gemini changed: added JSON reference artifact export (knn_model.json) to Outputs/models/ for standalone inference
    modelsDir = Path(outputsDir) / "models"
    modelsDir.mkdir(parents=True, exist_ok=True)

    fullLo, fullHi = computeFoldScaler([x for _, x, _ in data])
    fullX = [scaleVector(x, fullLo, fullHi) for _, x, _ in data]
    fullY = [y for _, _, y in data]
    fullCells = [cell for cell, _, _ in data]

    modelPath = modelsDir / "knn_model.json"
    with modelPath.open("w") as fh:
        json.dump(
            {
                "model": "kNN",
                "params": bestParams,
                "cv_metrics": {
                    "auc_mean": round(bestKey[0], 6),
                    "acc_mean": round(bestKey[1], 6),
                    "folds": foldsCount,
                    "seed": seed
                },
                "features": FEATURES,
                "lo": fullLo,
                "hi": fullHi,
                "reference_cells": fullCells,
                "reference_X": fullX,
                "reference_Y": fullY
            },
            fh,
            indent=2
        )

    bestDesc = f"k={bestParams['k']}, p={bestParams['p']}, weighted={bestParams['weighted']}"
    print(f"\nBest k-NN Config : {bestDesc} (AUC {bestKey[0]:.3f}, Acc {bestKey[1]:.3f})")
    print(f"Saved model      : {modelPath}")
    print(f"Saved metrics    : {summaryPath}")
    print(f"Saved OOF preds  : {oofPath}")





# =================================================================================================|
# [4] - MAIN EXECUTION
# =================================================================================================|

if __name__ == "__main__":
    # Gemini changed: updated dataset and output paths to resolve relative to root Datasets/ and Outputs/
    repoRoot = Path(__file__).resolve().parent.parent
    datasetsDir = repoRoot / "Datasets"
    outputsDir = repoRoot / "Outputs"

    posFolder = datasetsDir / "Weather (SF)"
    negFolder = datasetsDir / "Weather (NSF)"

    if not posFolder.exists() or not negFolder.exists():
        print("Missing 'Weather (SF)' or 'Weather (NSF)' folder inside Datasets/.")
    else:
        runBenchmark(posFolder, negFolder, outputsDir, foldsCount=5, seed=170)


