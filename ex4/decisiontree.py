import itertools, random, copy, sys
import numpy as np
import pdb

   
class Node():
    """nodes are """
    def __init__(self, data):
        # self.parent = parent
        self.children = {}
        self.data = data

    def addSubtree(self, sub, data):
        self.children[data] = sub

    def printNode(self):
        if not self.children : print "leaf node->", self.data
        else:
            for key, value in self.children.items():
                print "node->", self.data
                self.children[key].printNode()

def main(importance='argmax'):
    
    examples = getData('training') 
    attributes = initAttributes(examples)
    # pdb.set_trace()

    dtl = decisionTreeLearning(examples, attributes, importance, None)

    testData = getData('test')
    # print len(testData)

    accuracy = calculateAccuracy(dtl, testData)
    printResults(dtl, accuracy, importance)
    return 0

    
def decisionTreeLearning(examples, attributes, selection, parentExample=None ):
    # attCopy = attributes.copy()
    if not attributes : return Node( pluralityValue(examples) )
    elif not examples : return Node( pluralityValue(parentExample) )
    elif hasEqualClassification(examples) : return Node( examples[0][0] )
    else:
    # " Att is 'best' decision attribute for next node "
        A = selectAttribute(attributes, examples, selection)
        # print A
        tree = Node(A)
        # pdb.set_trace()        
        exs = []
        # print set(attributes[A])
        # pdb.set_trace()
        for value in set(attributes[A]):
            for e in examples:
                # pdb.set_trace()
                if e[int(A)] == value :
                    exs.append(e)
            # pdb.set_trace()
            if attributes : del attributes[A] #(attributes - A)
            subtree = decisionTreeLearning(exs, attributes, selection, examples)
            tree.addSubtree(subtree, value)
        return tree

def getData(fileName) :
    data = [ ] #2D
    file = open( 'data/'+fileName+'.txt', 'r' )
    for example in file : # example is a list of numbers 
        data.append( example.strip().split('\t') )
    # data.append( example.strip().split('\t') for example in file ) # 2D array
    file.close()
    # examples = []
    # n = len(data[0]) - 1
    # for d in data:
    #   ex = {'label' : d[n]}
    #   for i in range(n):
    #       ex[ str(i) ] = d[i]
    #   examples.append(ex)
    # print ('type(examples), length: ' , type(examples), ' ', len(examples) )
    return data

# dictionary
def initAttributes(examples) :
    attributes = { str(i+1):[a] for i, a in enumerate(examples[0][:-1]) }
    for i in range(len(examples)):
        for j in range(1,len(examples[i])):
            if(examples[i][j] == "1"):
                #add examples value to attributes
                attributes[str(j)].append(examples[i][j-1]) 
    return attributes

def getLabels(examples):
    # labels = []
    # for ex in examples :
    #   labels.append( ex[-1] )
    # return labels
    # print ('type(lables), length: ' , type(labels), ' ', len(labels) )
    return [ ex[-1] for ex in examples ]
    


def hasEqualClassification(examples):
    # classificatin = examples[0]
    for ex in examples[1:] :
        if ex != examples[0] : return False 
    return True

#plurality value
""" @return most common output value among a set of examples """
def pluralityValue(parentExample):
    if not parentExample: return None
    #element count
    labelsFreq = {}
    # print parentExample
    for p in parentExample:
        if not p[-1] in labelsFreq:
            labelsFreq[p[-1]] = 1
        else:
            labelsFreq[p[-1]] += 1
    maxFreq = max( labelsFreq.values() )
    # print 'max', maxFreq
    # find mostFreqLabel
    if len(labelsFreq) == 0 : return 0
    # print labelsFreq
    for key in labelsFreq.keys():
        if labelsFreq[key] == maxFreq :
            # print 'k: ',key
            return key


def selectAttribute(attributes, examples, selection):
    if selection == 'random':
        # return randImportance(attributes)
        return random.choice( attributes.keys() ) 
    if selection == 'argmax':
        return argmaxImportance(attributes, examples)

# def randImportance(attributes):
#     return random.choice( attributes.keys() ) 
    
def argmaxImportance(attributes, examples):
    # A good attribute splits the examples into subsets that are (ideally) all positive or all negative
    # labels = getLabels(examples) #outcomes
    labels = [ex[-1] for ex in examples]
    # print labels
    diffs = []
    # map(values.extend, attributes.values())
    for att in attributes.values() :
        diff = copy.deepcopy(labels)
        for v in att:
            if v in diff : diff.remove(v)
        diffs.append(diff)
    b = calculateBvalues(labels, diffs, attributes)  
    # print 'max',max(b.values())
    # exit()
    bestAttributes = [ k for k in b.keys() if b[k] == max(b.values()) ]
    # print 'best:    ',bestAttributes
    # print random.choice(bestAttributes)
    return random.choice(bestAttributes)

def gainInfo(data, parent):
    n = float(len(parent))
    remainder = 0.0
    for d in data:
        #if there are no elements in a return 0
        if not len(d): return 0
        remainder += (float(len(d))/n) * getB( d.count(2) / float(len(d)) )

    return 1-remainder

def calculateBvalues(labels, diffs, attributes):
    b = {}
    i=0
    values = attributes.values()
    for i, (key, value) in enumerate(attributes.items()):
        # print (key), i

        b[key] = gainInfo([values[i], diffs[i]], labels)
        # print [values[i], diffs[i]]
        i+=1
    return b

def getB(q):
    # calculate B value: 
    if q >= 1 or q <= 0: return 0
    #Entropy formula from the book 
    return -(q * np.log2(q) + ((1.0-q) * np.log2(1.0-q)))

def infoGain(examples, parent):
    n = len(parent)
    remainder = 0.
    for ex in examples :
        if not ex: return 0
        remainder += ( len(ex) / float(n) ) * getB(ex.count('1') / float(len(ex)))
    return 1. - remainder



def isMatch(tree, test):
    # pdb.set_trace()

    if len(tree.children) == 0:
        if test[-1] == tree.data :
            return True
        # print 'False'
        return False
    else:
        child = test[int(tree.data)]
        # print child
        # pdb.set_trace()
        return isMatch(tree.children[child], test)

def calculateAccuracy(tree, testData):
    matches = 0
    # print testData
    for test in testData:
        if isMatch(tree, test) : 
            # print test
            matches += 1
    # print 'matches= ', matches
    return float(matches) / len(testData)



def printResults(tree, accuracy, importance):
    output = open('out.txt','a') # open the result file in write mode
    tree.printNode()
    print >> output, "\nResults \n*******************************"
    print >>  output, "Importance: ", importance, "\t Accuracy = ", accuracy
    print "Importance: ", importance, " \t Accuracy = " , accuracy
    output.close()
 




if __name__ == "__main__":
    main('argmax')
    main('random')
