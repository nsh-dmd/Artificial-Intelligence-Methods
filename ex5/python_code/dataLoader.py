__author__ = 'kaiolae'
__author__ = 'kaiolae'
""" The given code skeleton is editet by Neshat Naderi"""

import Backprop_skeleton as Bp
import operator, pdb, sys
import matplotlib.pyplot as plt
import numpy as np
#Class for holding your data - one object for each line in the dataset
class dataInstance:

    def __init__(self,qid,rating,features):
        self.qid = qid #ID of the query
        self.rating = rating #Rating of this site for this query
        self.features = features #The features of this query-site pair.

    def __str__(self):
        return "Datainstance - qid: "+ str(self.qid)+ ". rating: "+ str(self.rating)+ ". features: "+ str(self.features)


#A class that holds all the data in one of our sets (the training set or the testset)
class dataHolder:

    def __init__(self, dataset):
        self.dataset = self.loadData(dataset)

    def loadData(self,file):
        #Input: A file with the data.
        #Output: A dict mapping each query ID to the relevant documents, like this: dataset[queryID] = [dataInstance1, dataInstance2, ...]
        data = open("datasets/"+file+".txt")
        dataset = {}
        for line in data:
            #Extracting all the useful info from the line of data
            lineData = line.split()
            rating = int(lineData[0])
            qid = int(lineData[1].split(':')[1])
            features = []
            for elem in lineData[2:]:
                if '#docid' in elem: #We reached a comment. Line done.
                    break
                features.append(float(elem.split(':')[1]))
            #Creating a new data instance, inserting in the dict.
            di = dataInstance(qid,rating,features)
            if qid in dataset.keys():
                dataset[qid].append(    di)
            else:
                dataset[qid]=[di]
        return dataset


def runRanker(dhTraining, dhTesting, epochs=20, err=True):
    #TODO: Insert the code for training and testing your ranker here.
    #Dataholders for training and testset
    #Creating an ANN instance - feel free to experiment with the learning rate (the third parameter).
    nn = Bp.NN(46,10,0.001)

    #TODO: The lists below should hold training patterns in this format: [(data1Features,data2Features), (data1Features,data3Features), ... , (dataNFeatures,dataMFeatures)]
    #TODO: The training set needs to have pairs ordered so the first item of the pair has a higher rating.
    trainingPatterns = [] #For holding all the training patterns we will feed the network
    testPatterns = [] #For holding all the test patterns we will feed the network
    for qid in dhTraining.dataset.keys():
        #This iterates through every query ID in our training set
        dataInstance = dhTraining.dataset[qid] #All data instances (query, features, rating) for query qid
        # store them as pairs, where the first item is rated higher than the second.
        dataInstance.sort(key=operator.attrgetter('rating'))
        for i in range(len(dataInstance)-1):
            for j in range(i+1, len(dataInstance)):
                a = dataInstance[i]
                b = dataInstance[j]
                if  a.rating > b.rating :
                    trainingPatterns.append( (a, b) )
                elif b.rating > a.rating:
                    trainingPatterns.append( (b, a) )
                else: continue
    # pdb.set_trace()
    
    # sort trainingPattern by highest pair
    # trainingPatterns.sort()

    for qid in dhTesting.dataset.keys():
        #This iterates through every query ID in our test set
        dataInstance = dhTesting.dataset[qid]
        dataInstance.sort(key=operator.attrgetter('rating'))
        for i in range(len(dataInstance)-1):
            for j in range(i+1, len(dataInstance)):
                a = dataInstance[i]
                b = dataInstance[j]
                if a.rating > b.rating :
                    testPatterns.append( (a, b) )
                elif b.rating > a.rating:
                    testPatterns.append( (b, a) )
                else: continue
    # sort testPattern by highest pair

    #Check ANN performance before training
    testErrorRates = [nn.countMisorderedPairs(testPatterns, err)] 
    trainErrorRates = [nn.countMisorderedPairs(trainingPatterns, err)] 
    # pdb.set_trace()

    for i in range(epochs):
        #Training
        nn.train(trainingPatterns, iterations=1)
        #Check ANN performance after training.

        testErrorRates.append( nn.countMisorderedPairs(testPatterns, err) )
        trainErrorRates.append( nn.countMisorderedPairs(trainingPatterns, err) )
        # pdb.set_trace()

    # pdb.set_trace()

    return testErrorRates, trainErrorRates, nn

    # pdb.set_trace()
  
def main(epochs):

    dhTraining = dataHolder('train')
    dhTesting = dataHolder('test')

    #Store the data returned by countMisorderedPairs and plot it, showing how training and testing errors develop.

    if epochs > 25 :
        avgTest, avgTrain, nn = runRanker(dhTraining, dhTesting, epochs, err=False)
        plotGraph( avgTrain, avgTest, "Performance", "Accuracy", epochs)
        return 

    avgTest, avgTrain, nn = runRanker(dhTraining, dhTesting, epochs)
    # pdb.set_trace()

    # calculate average 
    for it in range(1, 5):
        testErr, trainErr, nn = runRanker(dhTraining, dhTesting, epochs)
        avgTest = np.add(avgTest, testErr)
        avgTrain = np.add(avgTrain, trainErr)
        plotGraph(trainErr, testErr, "Error measurement in iteration " + str(it), "Error rate", epochs)
    # pdb.set_trace()

    avgTest /= 5.
    avgTrain /= 5.
    # Plot the averaged results
    plotGraph(avgTrain, avgTest, "Average Performance", "Error rate", epochs)

    return 0


def plotGraph(trainErr, testErr, title, yLabel, n):
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    ax.plot(range(0, n+1), trainErr, color='r')
    ax.plot(range(0, n+1), testErr, color='b')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(yLabel)
    ax.grid(True)
    fig.show()

if __name__ == "__main__":
    main(20)