__author__ = 'kaiolae'
__author__ = 'kaiolae'
import Backprop_skeleton as Bp
import matplotlib.pyplot as plt
import operator, pdb
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


def runRanker(trainingset, testset):
    #TODO: Insert the code for training and testing your ranker here.
    #Dataholders for training and testset
    dhTraining = dataHolder(trainingset)
    dhTesting = dataHolder(testset)

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
    # testPatterns.sort() 
    

    #Check ANN performance before training
    testErrorRates = [ nn.countMisorderedPairs(testPatterns) ]
    epochs = 50
    for i in range(epochs):
        #Running 25 iterations, measuring testing performance after each round of training.
        #Training
        nn.train(trainingPatterns, iterations=1)
        #Check ANN performance after training.
        testErrorRates.append( nn.countMisorderedPairs(testPatterns) )

        test_error_percent.append(nn.countMisorderedPairs(testPatterns))
        training_error_percent.append(nn.countMisorderedPairs(trainingPatterns))

    #     if i==3:
    #         break
    # pdb.set_trace()
    #TODO: Store the data returned by countMisorderedPairs and plot it, showing how training and testing errors develop.
    output = open('output.txt', 'a')
    print >> output, "\nTests \n*******************************"
    # print >> output, "Error rates = ", errorRates
    print >> output, "numInputs = ", nn.numInputs, "\tnumHidden = ", nn.numHidden
    print >> output, "epochs = ", epochs, "\tlearning rate = ", nn.learningRate

    plt.plot(range(0, epochs+1), errorRates)
    plt.ylabel('Error rate')
    plt.xlabel('Training epochs')
    plt.grid(True)
    plt.show()
runRanker('train', 'test')