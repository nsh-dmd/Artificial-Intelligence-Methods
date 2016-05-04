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


def runRanker(dhTraining, dhTesting, epochs=20):
    #TODO: Insert the code for training and testing your ranker here.
    #Dataholders for training and testset
    # dhTraining = dataHolder(trainingset)
    # dhTesting = dataHolder(testset)

    #Creating an ANN instance - feel free to experiment with the learning rate (the third parameter).
    nn = Bp.NN(46,10,0.001)

    #TODO: The lists below should hold training patterns in this format: [(data1Features,data2Features), (data1Features,data3Features), ... , (dataNFeatures,dataMFeatures)]
    #TODO: The training set needs to have pairs ordered so the first item of the pair has a higher rating.
    trainingPatterns = [] #For holding all the training patterns we will feed the network
    testPatterns = [] #For holding all the test patterns we will feed the network
    for qid in dhTraining.dataset.keys():
        #This iterates through every query ID in our training set
        dataInstance = dhTraining.dataset[qid] #All data instances (query, features, rating) for query qid
        #TODO: Store the training instances into the trainingPatterns array. 
            #  Remember to store them as pairs, where the first item is rated higher than the second.
        dataInstance.sort(key=operator.attrgetter('rating'))
        for i in range(len(dataInstance)-1):
            for j in range(i+1, len(dataInstance)):
                a = dataInstance[i]
                b = dataInstance[j]
                if a.rating > b.rating :
                    trainingPatterns.append( (a, b) )
                elif b.rating > a.rating:
                    trainingPatterns.append( (b, a) )
                else: continue
        #TODO: Hint: A good first step to get the pair ordering right, is to sort the instances based on 
                #    their rating for this query. (sort by x.rating for each x in dataInstance)
    # pdb.set_trace()
    
    # sort trainingPattern by highest pair
    # trainingPatterns.sort()



    for qid in dhTesting.dataset.keys():
        #This iterates through every query ID in our test set
        dataInstance = dhTesting.dataset[qid]
        #TODO: Store the test instances into the testPatterns array, once again as pairs.
        #TODO: Hint: The testing will be easier for you if you also now order the pairs -
             # it will make it easy to see if the ANN agrees with your ordering.
        dataInstance.sort(key=operator.attrgetter('rating'))
        # for i in range(0, len(dataInstance)-1, 2):
        #     a = dataInstance[i]
        #     b = dataInstance[i+1]
        #     # testPatterns.append( (a, b) if a.rating > b.rating else (b, a) )
        #     if a.rating > b.rating :
        #         testPatterns.append( (a, b) )
        #     elif b.rating > a.rating :
        #         testPatterns.append( (b, a) )
        #     else: continue
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
    testErrorRates = [nn.countMisorderedPairs(testPatterns)] 
    trainErrorRates = [nn.countMisorderedPairs(trainingPatterns)] 
    # pdb.set_trace()

    for i in range(epochs):
        #Training
        nn.train(trainingPatterns, iterations=1)
        #Check ANN performance after training.

        testErrorRates.append( nn.countMisorderedPairs(testPatterns) )
        trainErrorRates.append( nn.countMisorderedPairs(trainingPatterns) )
        # pdb.set_trace()

    # pdb.set_trace()

    return testErrorRates, trainErrorRates, nn

    # pdb.set_trace()
  
def main(epochs):

    dhTraining = dataHolder('train')
    dhTesting = dataHolder('test')

    # testErr, trainErr, nn = runRanker('train', 'test', epochs)
      #TODO: Store the data returned by countMisorderedPairs and plot it, showing how training and testing errors develop.

    avgTest, avgTrain, nn = runRanker(dhTraining, dhTesting, epochs)
    # print len(avgTest.tolist()), len(avgTrain.tolist()), avgTest[-1], avgTrain[-1]
    # exit()
    # pdb.set_trace()
    avgTest = np.array(avgTest)
    avgTrain = np.array(avgTrain)
    # pdb.set_trace()

    # calculate average 
    for it in range(1, 5):
        testErr, trainErr, nn = runRanker(dhTraining, dhTesting, epochs)

        avgTest = np.add(avgTest, testErr)
        avgTrain = np.add(avgTrain, trainErr)
        print avgTest.size, avgTrain.size, avgTest[-1], avgTrain[-1]
    # pdb.set_trace()


    avgTest /= 5.
    avgTrain /= 5.
    print avgTest.shape(), avgTrain.shape(), '\n',avgTest, '\n',avgTrain
    # print len(avgTest), len(avgTrain)
    # print avgTest[0], avgTrain[0]
    pdb.set_trace()
    

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Performance measurment")
    ax.plot(range(0, epochs+1), avgTrain, color='r')
    ax.plot(range(0, epochs+1), avgTest, color='b')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error rate')
    ax.grid(True)
    plt.show()


    # output = open('output.txt', 'a')
    # print >> output, "\nTests \n*******************************"
    # print >> output, "numInputs = ", nn.numInputs, "\tnumHidden = ", nn.numHidden
    # print >> output, "epochs = ", epochs, "\tlearning rate = ", nn.learningRate
    

    return 0

if __name__ == "__main__":
    main(25)