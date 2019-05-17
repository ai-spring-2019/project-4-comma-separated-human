"""
Trish Gagnon

Usage: python3 project3.py DATASET.csv
Only the increment data set works right now, and "works" is a strong word - 
all I adjust turns to nan values. Print statements have been left in if you
would like to see for yourself
"""

import csv, sys, random, math

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        nn.forward_propagate(x)
        class_prediction = nn.predict_class()
        if class_prediction != y[0]:
            true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here

class NeuralNode:

    def __init__(self, inweights):
        self._weights = inweights
        self._total = 0
        self._fault = 0
        if not self._weights == []:
            dummyWeight = random.uniform(0,1)
            self._weights = [dummyWeight] + self._weights
            self._total = dummyWeight

    def __str__(self):
        result = "I'm a node! "
        for weight in self._weights:
            result += str(weight) + ' '
        return result[:len(result)-1] + " value: " + str(self._total)

class NeuralNetwork:

    def __init__(self, layers, threshold):
        self._layers = layers
        self._threshold = threshold
        self._nodes = self.assign_random_weights()
        #for node in self._nodes:
        #    print(node)

    def assign_random_weights(self):
        """Each weight is in reference to the incoming weight"""
        
        allNodes = []
        currentLayer = []
        #first layer has no incoming weights
        for x in range(self._layers[0]):
            currentLayer.append(NeuralNode([]))
        allNodes.append(currentLayer)
        
        #for each non-input layer
        for i in range(1, len(self._layers)):
            currentLayer = []
            for j in range(self._layers[i]):
                nodeWeights = [] 
                #adds the weights for each node in the previous layer
                for _ in range(self._layers[i-1]):
                    nodeWeights.append(random.uniform(0,1))
                node = (NeuralNode(nodeWeights))
                currentLayer.append(node)
            allNodes.append(currentLayer)
        return allNodes
            

    def forward_propogate(self, x):
        """Takes the inputs, x, and propogates them forward using the weights to
        determine the outputs"""
        
        #assigns the value of the first layer of nodes
        for i in range(len(self._nodes[0])):
            self._nodes[0][i]._total = x[0][i]

        #goes through the rest of the layers and moves the values forward
        for i in range(1, len(self._layers)):
            for j in range(self._layers[i]):
                currentNode = self._nodes[i][j]
                currentNode._total = currentNode._weights[0]
                for x in range(len(self._nodes[i-1])):
                    currentNode._total += currentNode._weights[x] * self._nodes[i-1][x]._total

                    
    def back_propogation_learning(self, training):
        """Goes backward through the nodes and readjusts the weights of all the
        nodes"""
        
        #was going to increase this, but the errors were happening much earlier
        for i in range(1):
            for example in training:
                #check each example
                self.forward_propogate(example)
                check = increment_check(example[1], self._nodes[-1])
                if not check:
                    for i in range(len(self._nodes[-1])):
                        node = self._nodes[-1][i]
                        node._fault = node._total * (1-node._total) * (example[1][i] - node._total)
                    #each layer
                    for i in range(len(self._nodes) - 1, 0, -1):
                        #previous layer's nodes, this layer's weights
                        for j in range(len(self._nodes[i-1])):
                            prevNode = self._nodes[i-1][j]
                           # prevNode._fault = prevNode._total * (1-prevNode._total)
                            sum = 0
                            
                            #this layer's nodes
                            for node in self._nodes[i]:
                                prevNode._fault += (node._total) * (1-prevNode._total)
                                sum += node._weights[j] * node._fault
                            prevNode._fault *= sum

                    #all layers but the last
                    for i in range(len(self._nodes) - 1):
                        #next layer's nodes
                        for node in self._nodes[i + 1]:
                            print("before:", node)
                            for weight in range(1, len(node._weights)):
                                print(node._fault)
                                node._weights[weight] += self._nodes[i][weight - 1]._total * node._fault
                            print("after:", node)

                    self.reset_all_faults()

    def reset_all_faults(self):
        """Resets all faults in the nodes to 0"""
        
        for layer in self._nodes:
            for node in layer:
                node._fault = 0

def increment_check(y, nodes):
    """This checks to see if the output of a forward propogation matches the 
    expected value"""
    
    vals = []
    for node in nodes:
        if node._total > .5:
            vals.append(1.0)
        else:
            vals.append(0.0)
    if vals == y:
        return True
    else:
        return False
    
def main():
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]

    # Check out the data:
    for example in training:
        print(example)

    ### I expect the running of your program will work something like this;
    ### this is not mandatory and you could have something else below entirely.
    nn = NeuralNetwork([3, 6, 3], 0.5)
    nn.back_propogation_learning(training)

if __name__ == "__main__":
    main()
