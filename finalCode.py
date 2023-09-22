import random
from random import randint, random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from statistics import mean    



boothiPath = "./imgs/boothiGray.jpg"
groupImgPath = "./imgs/groupGray.jpg"

groupImg = mpimg.imread(groupImgPath)
boothi = mpimg.imread(boothiPath)
groupRows, groupCols =  np.shape(groupImg)
boothiRows, boothiCols =  np.shape(boothi)
maxCorelations = []
meanOfEveryGeneration = []


populationSize = 1000
thresholdValue = 0.85
generations = 0
maxGenerations = 1000


def giveRandomNumber(maxRange):
    return int(random() * maxRange + 1)


def populationInitialization(groupCols, groupRows, populationSize):
    currentGeneration = []
    while (len(currentGeneration) != populationSize):
        x, y = (giveRandomNumber(groupRows), giveRandomNumber(groupCols))
        if (x + 35 ) < 512 and (y + 29) < 1024:
            currentGeneration.append((x, y))
     
    return currentGeneration    



def findCorelationOfTwoMatrixes(slicedMatrixFromBarriImage, boothi):
    x = slicedMatrixFromBarriImage
    y = boothi
    return ((x - x.mean()) * (y - y.mean())).mean()/ (np.std(x) * np.std(y))


def fitnessEvaluation(currentGeneration, barriImage, boothi):
    fitnessValues = []
    for points in currentGeneration:
        x, y = points
        slicedImageFromBarriImage = barriImage[x:x +boothiRows, y: y+boothiCols]
        fitnessValues.append(findCorelationOfTwoMatrixes(slicedImageFromBarriImage, boothi))
    return fitnessValues
  

def selection(currentGeneration, fitnessValues):
    tempArray = []
    for i in range(len(currentGeneration)):
        tempArray.append([fitnessValues[i], currentGeneration[i]])
    tempArray.sort(key=lambda row: (row[0]), reverse=True)
    return tempArray


# Utility functions 

def giveMeCoordinates(selectedGeneration):
    onlyCoordinates = []
    for i in selectedGeneration:
        onlyCoordinates.append(i[1])
    return onlyCoordinates


def binaryOF(num, bits=10):
    return np.binary_repr(num, bits)


def decimalOf(num):
    return int(num, 2)


def giveDecimalValues(listOfBits):
    temp = ""
    for i in listOfBits:
        temp += str(i)
        
    return int(temp, 2)




def giveMeFitnessValues(selectedGeneration):
    fitnessValues = [] 
    for i in selectedGeneration:
        fitnessValues.append(i[0])
    return fitnessValues

# ======================= End Utility Functions >



# Cross Over  

def crossOver(parent1, parent2):
    x1, y1 = parent1
    x2, y2 = parent2
    condition = True
    while (condition):
        x1y1 =  binaryOF(x1, 9)+binaryOF(y1, 10)
        x2y2 = binaryOF(x2, 9)+binaryOF(y2, 10)
        randomNumberToMakeACut = np.random.randint(0, 9)
        randomNumberToMakeCutTwo = np.random.randint(9, 18)
        x1y1New = x1y1[:randomNumberToMakeACut] + x2y2[randomNumberToMakeACut:randomNumberToMakeCutTwo] + x1y1[randomNumberToMakeCutTwo:]
        x2y2New = x2y2[:randomNumberToMakeACut] + x1y1[randomNumberToMakeACut:randomNumberToMakeCutTwo] + x2y2[randomNumberToMakeCutTwo:]
      
        x1 = decimalOf(x1y1New[:9])
        y1 = decimalOf(x1y1New[9:])
        x2 = decimalOf(x2y2New[:9])
        y2 = decimalOf(x2y2New[9:])
        # if crossover is good then exit the loop meaning make the condition false
        condition = not((x1 + 35 < 512 and y1 + 29 < 1024) and (x2 + 35 < 512 and y2 + 29 < 1024))
    return [(x1, y1), (x2, y2)]


def mutation(point):
    xO, yO = point # x_original and y_original
    x = binaryOF(xO, 9)
    y = binaryOF(yO, 10)
    combination = x + y
    combination = list(combination)
    temp = giveRandomNumber(len(combination) - 1)
    condition = True
    while (condition):
        if (combination[temp] == '0'):
            combination[temp] = '1'
        else:
            combination[temp] = '0'
              
        xO = giveDecimalValues(combination[:9])
        yO = giveDecimalValues(combination[9:])
        condition = ( xO + 35 > 512 or yO + 29 > 1024)
 
    return xO, yO


def mutationOfPoints(generation):
    generationLength = len(generation) * 0.02
    count = 0
    while (count != generationLength):
        # mid = int(len(generation) / 2)
        randomNumber1 = randint((len(generation) // 2), len(generation) -1)
        randomNumber2 = randint(len(generation) // 2, len(generation)-1)
        generation[randomNumber1] = mutation(generation[randomNumber1])
        generation[randomNumber2] = mutation(generation[randomNumber2])
        count += 2
    # generation[-1] = mutation(generation[-1])
    return generation

def newGeneration(selectedGeneration, groupRows, groupCols):
    newGeneration = []
    newGeneration.append(selectedGeneration[0][1])
    lastChild = selectedGeneration[-1][1]
    selectedGeneration = giveMeCoordinates(selectedGeneration)
    for i in range(1, len(selectedGeneration) - 1, 2):
        newPoints = crossOver(selectedGeneration[i], selectedGeneration[i+1])
        newGeneration.append(newPoints[0])
        newGeneration.append(newPoints[1])
    newGeneration.append(lastChild)
    newGeneration = mutationOfPoints(newGeneration)
    return newGeneration


# Graph stuff

maxCorelations = []
avgFitness = []

def plotFitness(maxCorelations, avgFitness):
    plt.figure()
    plot1, = plt.plot(maxCorelations)
    plot2, = plt.plot(avgFitness)
    plt.title("Plot of Max Corelations and Average Fitnesses")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend([plot1,plot2],["Max Fitess per Generation", "Avg Fitness per Generation"])
    plt.show()


# Program will start from here

currentGeneration = populationInitialization(groupCols, groupRows, 500)
fitnessValues = fitnessEvaluation(currentGeneration, groupImg, boothi)
selectedGeneration = selection(currentGeneration, fitnessValues)

for i in range(maxGenerations):
    print(f"Generation No. {i + 1}")
    newGenerationCoordinates = newGeneration(selectedGeneration, groupRows, groupCols)
    fitnessValues = fitnessEvaluation(newGenerationCoordinates, groupImg, boothi)
    selectedGeneration = selection(newGenerationCoordinates, fitnessValues)
    sortedFitnessValues = giveMeFitnessValues(selectedGeneration)
    maxCorelations.append(max(sortedFitnessValues))
    avgFitness.append(mean(sortedFitnessValues))
    temp = round(selectedGeneration[0][0], 2)
    x, y = selectedGeneration[0][1]
    
    if temp>=thresholdValue:
        print("I am boothi", (x, y), temp)
        plt.imshow(Image.open("./imgs/groupGray.jpg"), cmap=plt.get_cmap('gray'))
        plt.gca().add_patch(Rectangle((y,x),29,35,
                        edgecolor='red',
                        facecolor='none',
                        lw=2))
        plt.show()
        break
        

# print(max(maxCorelations))
plotFitness(maxCorelations, avgFitness)
