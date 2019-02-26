import csv
import numpy as np
import math
import matplotlib.pyplot as plt

TrainingFilename = "train.csv"
TestingFilename = "test.csv"
#TrainingFilename = "zipcode_train.csv"
#TestingFilename = "zipcode_test.csv"

# The following is used to plot the class boundaries
# .....................................................................................
# def plot_by_class_better(dataset, classes, title):
#     dataset[:, 2] = classes
#     x0 = [row[0] for row in dataset if 0 in row]
#     y0 = [row[1] for row in dataset if 0 in row]
#     x1 = [row[0] for row in dataset if 1 in row]
#     y1 = [row[1] for row in dataset if 1 in row]
#     plt.scatter(x0, y0, label='0', color="red", marker="o", s=30)
#     plt.scatter(x1, y1, label='1', color="blue", marker="o", s=30)
#     plt.xlabel('x - axis')
#     plt.ylabel('y - axis')
#     plt.title(title)
#     plt.show()
#     return
#
# def createDataPoint():
#     x = -3
#     y = -8
#     data = []
#     res = 20
#     for i in range(9*res):
#         for j in range(20*res):
#             data.append([(x*res+i+0.1)/res, (y*res+j+0.1)/res, 0])
#     data = np.asarray(data)
#     return data
# .....................................................................................

# Load data from csv file
def load_csv(filename):
    lines = csv.reader(open(filename, "r", encoding='utf-8-sig'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    dataset = np.asarray(dataset, dtype=np.float32)
    return dataset


# Seperate data by class
def class_sorted_data(dataset):
    classes = np.unique(dataset[:, np.size(dataset, 1) - 1])
    sortedclassdata = []
    for i in range(len(classes)):
        item = classes[i]
        itemindex = np.where(dataset[:, np.size(dataset, 1) - 1] == item)   # index  of rows with label class[i]
        singleclassdataset = dataset[itemindex, 0:np.size(dataset, 1) - 1]  # array  of data for class[i]
        sortedclassdata.append(np.matrix(singleclassdataset))               # matrix of data for class[i]
    return sortedclassdata, classes


# posterior prob = likelihood * prior probability
# first prior probability
def prior_prob(dataset, sortedclassdata):
    priorprobability = []
    for i in range(len(sortedclassdata)):
        priorprobability.append(len(sortedclassdata[i])/len(dataset))
    return priorprobability


# to find likelihood
# we find mean and covariance first
def find_mean(sortedclassdata):
    classmeans = []
    for i in range(len(sortedclassdata)):
        classmeans.append(sortedclassdata[i].mean(0))
    return classmeans


def find_covariance(sortedclassdata, classmeans):
    covariance = []
    ndpc = len(sortedclassdata[0])      # total number of data points (rows) per class
    for i in range(len(classmeans)):
        xn = np.transpose(sortedclassdata[i])
        mean_class = np.transpose(classmeans[i])
        tempvariance = sum([(xn[:, x] - mean_class) * np.transpose(xn[:, x] - mean_class) for x in range(int(ndpc))])
        tempvariance = tempvariance / (ndpc - 1)
        covariance.append(tempvariance)
    return covariance


# find likelihood, given a gaussian distribuition
# and knowing the mean and variance, or in this case, the covariance
# see eq(1.52) pg.25 Pattern Recognition and Machine Leraning....
#
def find_n_class_probability(dataset, classmeans, covariance, priorProb, classes):
    expo = []
    nclassprob = []
    probabilityofclass = []
    datasetDimensions = len(covariance[0])
    testdatasetMatrix = np.matrix(dataset)
    datasetTranspose = np.transpose(testdatasetMatrix[:,0:len(dataset[0])-1])
    for i in range(len(dataset)):
        x = datasetTranspose[:, i]
        for j in range(len(classmeans)):
            determinate = np.linalg.det(covariance[j])
            if determinate == 0:
                addValue = 0.006*np.identity(datasetDimensions)
                covariance[j] = addValue + covariance[j]
                determinate = np.linalg.det(covariance[j])
                #print("Changed Determinate")
            exponent = (-0.5)*np.transpose(x-np.transpose(classmeans[j]))*np.linalg.inv(covariance[j])*(x-np.transpose(classmeans[j]))
            expo.append(exponent)
            nprobabilityofclass = priorProb[j]*(1/((2*math.pi)**(datasetDimensions/2)))*(1/(determinate**0.5))*math.exp(expo[j])
            probabilityofclass.append(nprobabilityofclass)
        arrayprob = np.array(probabilityofclass)
        nclassprob.append(classes[np.argmax(arrayprob)])
        probabilityofclass = []
        expo = []
    return nclassprob


def get_accuracy(nclassprob, dataset):
    Classes = np.transpose([np.asarray(nclassprob, dtype=np.float32)])
    Truth = np.transpose([np.asarray(dataset[:, dataset.shape[1]-1])])
    validate = np.equal(Classes, Truth)
    accuracy = 100 * (np.sum(validate) / dataset.shape[0])
    return accuracy

def convert_covariance_to_naive(matrix):
    numofclasses = len(matrix)
    numoffeatures = len(matrix[0])
    for i in range(numofclasses):
        for j in range(numoffeatures):
            for k in range(numoffeatures):
                if j != k:
                    matrix[i][j, k] = 0
    print("Converted covariance to Naive Bayes")
    return matrix

trainingData = load_csv(TrainingFilename)

testingData = load_csv(TestingFilename)

# The following is used to plot the class boundaries
# .....................................................................................
#testingData = createDataPoint()
# .....................................................................................

sortclassdata, classes = class_sorted_data(trainingData)

priorProb = prior_prob(trainingData, sortclassdata)

meansbyclass = find_mean(sortclassdata)

covariance = find_covariance(sortclassdata, meansbyclass)

# The following is used to find naive bayes covariance
# .....................................................................................
#covariance = convert_covariance_to_naive(covariance)
# .....................................................................................

nclassprob = find_n_class_probability(trainingData, meansbyclass, covariance, priorProb, classes)
accuracy = get_accuracy(nclassprob, trainingData)
print(f"{accuracy}% Correct on Training Data using Bayes Classifier")

nclassprob = find_n_class_probability(testingData, meansbyclass, covariance, priorProb, classes)

# The following is used to plot the class boundaries
# .....................................................................................
#plot_by_class_better(testingData , nclassprob, 'Non Naive Bayes Class Boundaries')
# .....................................................................................

accuracy = get_accuracy(nclassprob, testingData)
print(f"{accuracy}% Correct on Testing Data using Bayes Classifier")