# Bruce Li, Evan Yukevich, and Preston Vander Vos
# Carnegie Mellon University
# 70-339 FinTech
# Final Project

# data: https://coinmetrics.io/data-downloads/

import os
import numpy as np
import random

random.seed(123)

def preprocessData(path):
    files = os.listdir(path)
    allCols = []
    colDict = dict()
    for file in files:
        f = open(path + file, "r").read()
        curCols = f.splitlines()[0].split(",")
        for col in curCols:
            colDict[col] = colDict.get(col, 0) + 1
        allCols.append(curCols)
    colCounts = []
    for i in range(len(allCols)):
        temp = []
        for col in allCols[i]:
            # keep col if it appears in a majority of files
            if(col != 'date' and colDict[col] > len(files) / 2):
                temp.append(col)
        allCols[i] = temp
        colCounts.append(len(allCols[i]))
    numInputs = max(colCounts)
    inputNames = allCols[colCounts.index(numInputs)]
    labelIndex = inputNames.index('price(USD)')
    allData = []
    labels = []
    for i in range(len(files)):
        if(colCounts[i] != numInputs):
            continue
        filename = path + files[i]
        f = open(filename, "r").read().splitlines()
        curCols = f[0].split(",")
        colPos = []
        for col in inputNames:
            colPos.append(curCols.index(col))
        for line in f[1:]:
            allEntries = line.split(",")
            temp = [None] * numInputs
            for j in range(len(colPos)):
                temp[j] = allEntries[colPos[j]]
            goodRow = True
            for entry in temp:
                if(entry == ""):
                    goodRow = False
                    break
            if(goodRow):
                labels.append(temp[labelIndex])
                bye = temp.pop(labelIndex)
                allData.append(temp)
    labels = np.matrix(labels).transpose()
    allData = np.matrix(allData)
    return(allData, labels)

def normalizeData(data, label):
    data = data.astype(np.float)
    label = label.astype(np.float)
    dataMean = data.mean(0)
    labelMean = label.mean(0)
    dataSD = data.std(0)
    labelSD = label.std(0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data.itemset((i, j), (data.item(i, j) - dataMean.item(j)) / dataSD.item(j))
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            label.itemset((i, j), (label.item(i, j) - labelMean.item(j)) / labelSD.item(j))
    return(data, label)

def splitData(data, label):
    validateSize = data.shape[0] // 3
    validateIndexes = random.sample(range(data.shape[0]), validateSize)
    validateDataMatrix = data[validateIndexes, :]
    validateLabelMatrix = label[validateIndexes, :]
    trainDataMatrix = np.delete(data, validateIndexes, 0)
    trainLabelMatrix = np.delete(label, validateIndexes, 0)
    return(trainDataMatrix, trainLabelMatrix, validateDataMatrix, validateLabelMatrix)

def holdConstantData(data1, data2, index):
    data1[:, index] = 1
    data2[:, index] = 1
    return(data1, data2)
