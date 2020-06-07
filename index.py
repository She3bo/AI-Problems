from math import sqrt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import random
import math
from collections import deque

# region SearchAlgorithms

listrowmaze = []
listcolmaze = []
fullmaze=[]
map={}
mappath={}
col = 0
rw = 0
class Node:
    id = None
    up = None
    down = None
    left = None
    right = None
    previousNode = None
    visit = False

    def __init__(self, value):
        self.value = value


class SearchAlgorithms:
    ''' * DON'T change Class, Function or Parameters Names and Order
        * You can add ANY extra functions,
          classes you need as long as the main
          structure is left as is '''
    path = []  # Represents the correct path from start node to the goal node.
    fullPath = []  # Represents all visited nodes from the start node to the goal node.
    maze = ''

    def __init__(self, mazeStr):
        ''' mazeStr contains the full board
         The board is read row wise,
        the nodes are numbered 0-based starting
        the leftmost node'''
        self.maze = mazeStr

    def load(self):
        global listrowmaze
        global listcolmaze
        global fullmaze
        global map

        count = 0
        listrowmaze.clear()
        listcolmaze.clear()
        listnode = []
        map.clear()
        fullmaze.clear()

        size = len(self.maze)

        for i in range(0, size):
            if (self.maze[i] == ','):
                continue
            elif (self.maze[i] == ' '):
                listrowmaze.append(listcolmaze.copy())
                listcolmaze.clear()
            else:
                listcolmaze.append(self.maze[i])
        listrowmaze.append(listcolmaze.copy())

        for r in range(0, len(listrowmaze)):
            listnode.clear()
            for c in range(0, len(listrowmaze[r])):
                nnode = Node(listrowmaze[r][c])
                nnode.id = count

                if r + 1 < len(listrowmaze):
                    nnode.down = count + len(listrowmaze[r])
                else:
                    nnode.down = None
                if r - 1 >= 0:
                    nnode.top = count - len(listrowmaze[r])
                else:
                    nnode.top = None
                if c + 1 < len(listrowmaze[r]):
                    nnode.right = count + 1

                else:
                    nnode.right = None
                if c - 1 >= 0:
                    nnode.left = count - 1

                else:
                    nnode.left = None
                map[count] = nnode
                listnode.append(nnode)
                count += 1
            fullmaze.append(listnode.copy())
    def BFS(self):
        '''Implement Here'''
        self.load()
        mappath.clear()
        self.fullPath.clear()
        self.path.clear()
        opened=deque()
        listnode=[]
        for  i in range(0,len(fullmaze)):
            for j in range(0,len(fullmaze[i])):
                if fullmaze[i][j].value=='S':
                    start = fullmaze[i][j]
                    fullmaze[i][j].visit=True
                    break

        opened.append(start)

        while True:
            if len(opened)>0:
                start = opened.popleft()
            else:
                break
            if start.value=='E':
                self.fullPath.append(start.id)
                break;
            else:
                global il, ir, it, id, jl, jr, jt, jd
                for i in range(0, len(fullmaze)):
                    for j in range(0, len(fullmaze[i])):
                        if fullmaze[i][j].id == start.left:
                            il=i
                            jl=j
                        if fullmaze[i][j].id == start.right:
                            ir=i
                            jr=j
                        if fullmaze[i][j].id == start.down:
                            id=i
                            jd=j
                        if fullmaze[i][j].id == start.top:
                            it=i
                            jt=j
                if start.top != None:
                    if fullmaze[it][jt].visit == False:
                        if fullmaze[it][jt].value == '.' or fullmaze[it][jt].value == 'E':
                            opened.append(fullmaze[it][jt])
                            fullmaze[it][jt].visit = True
                if start.down != None:
                    if fullmaze[id][jd].visit == False:
                        if fullmaze[id][jd].value == '.' or fullmaze[id][jd].value == 'E':
                            opened.append(fullmaze[id][jd])
                            fullmaze[id][jd].visit = True
                if start.right != None:
                    if fullmaze[ir][jr].visit == False:
                        if fullmaze[ir][jr].value == '.' or fullmaze[ir][jr].value == 'E':
                            opened.append(fullmaze[ir][jr])
                            fullmaze[ir][jr].visit = True
                if start.left != None:
                    if fullmaze[il][jl].visit == False:
                        if fullmaze[il][jl].value == '.' or fullmaze[il][jl].value == 'E':
                            opened.append(fullmaze[il][jl])
                            fullmaze[il][jl].visit = True


                if start.right != None:
                   listnode.append(map.get(start.right))
                if start.left != None:
                   listnode.append(map.get(start.left))
                if start.top!=None:
                   listnode.append(map.get(start.top))
                if start.down!=None:
                   listnode.append(map.get(start.down))


                mappath[start]=listnode.copy()
                listnode.clear()
                self.fullPath.append(start.id)

        queue = []
        queue.append([map[0]])
        path2 = []
        while queue:
            path2 = queue.pop(0)
            node = path2[-1]
            if node.value == 'E':
                break
            for adjacent in mappath.get(node, []):
                new_path = list(path2)
                new_path.append(adjacent)
                queue.append(new_path)
        for i in range(0, len(path2)):
            self.path.append(path2[i].id)
        return self.fullPath, self.path

# endregion

# region NeuralNetwork
class NeuralNetwork():

    def __init__(self, learning_rate, threshold):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.synaptic_weights = 2 * np.random.random((2, 1)) - 1

    def step(self, x):
        if x > float(self.threshold):
            return 1
        return 0

    def train(self, training_inputs, training_outputs, training_iterations):
        
        for iteration in range(training_iterations):
            
            output = self.think(training_inputs)

            error = training_outputs - output

            
            adjustments = np.dot(training_inputs.T, error * self.learning_rate)

            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output_in = np.sum(np.dot(inputs, self.synaptic_weights))
        output=self.step(output_in)
        return output


# endregion


# region ID3
class item:
    def __init__(self, age, prescription, astigmatic, tearRate, diabetic, needLense):
        self.age = age
        self.prescription = prescription
        self.astigmatic = astigmatic
        self.tearRate = tearRate
        self.diabetic = diabetic
        self.needLense = needLense



    def getDataset():
        data = []
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
        data.append(item(0, 0, 0, 0, 1, labels[0]))
        data.append(item(0, 0, 0, 1, 1, labels[1]))
        data.append(item(0, 0, 1, 0, 1, labels[2]))
        data.append(item(0, 0, 1, 1, 1, labels[3]))
        data.append(item(0, 1, 0, 0, 1, labels[4]))
        data.append(item(0, 1, 0, 1, 1, labels[5]))
        data.append(item(0, 1, 1, 0, 1, labels[6]))
        data.append(item(0, 1, 1, 1, 1, labels[7]))
        data.append(item(1, 0, 0, 0, 1, labels[8]))
        data.append(item(1, 0, 0, 1, 1, labels[9]))
        data.append(item(1, 0, 1, 0, 1, labels[10]))
        data.append(item(1, 0, 1, 1, 0, labels[11]))
        data.append(item(1, 1, 0, 0, 0, labels[12]))
        data.append(item(1, 1, 0, 1, 0, labels[13]))
        data.append(item(1, 1, 1, 0, 0, labels[14]))
        data.append(item(1, 1, 1, 1, 0, labels[15]))
        data.append(item(1, 0, 0, 0, 0, labels[16]))
        data.append(item(1, 0, 0, 1, 0, labels[17]))
        data.append(item(1, 0, 1, 0, 0, labels[18]))
        data.append(item(1, 0, 1, 1, 0, labels[19]))
        data.append(item(1, 1, 0, 0, 0, labels[20]))
        return data

class Feature:
    def __init__(self, name):
        self.name = name
        self.visited = -1
        self.infoGain = -1
tree=[]
class ID3:
    def __init__(self, features):
        self.features = features

    def entropy(self,target_col):
        # print(target_col)
        elements, counts = np.unique(target_col, return_counts=True)
        #print(elements,"  " , counts)
        entropy = np.sum([(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
        #print(entropy)
        return entropy

    def getItems(self,data,val,label):
        a = []
        for i in range(len(data)):
            if data[i] == val:
              a.append(label[i])
        # print("a =",a)
        return a

    def InfoGain(self, data,label):
        total_entropy = self.entropy(label)    # lable or needlese
        # print("total_entropy = ",total_entropy)
        vals, counts = np.unique(data, return_counts=True)
        #print("from Info gain : ",vals,counts)
        Weighted_Entropy = np.sum([(counts[i] / np.sum(counts)) * self.entropy(self.getItems(data,vals[i],label)) for i in range(len(counts))])
        # print("Weighted_Entropy : ",Weighted_Entropy)
        Information_Gain = total_entropy - Weighted_Entropy
        #print("Information_Gain : ",Information_Gain)
        return Information_Gain

    def start(self,dataset):
        label = [dataset[i].needLense for i in range(len(dataset))]
        data = []
        features = []
        for f in self.features:
            fet = []
            for dd in dataset:
                fet.append(dd.__getattribute__(f.name))
            data.append(fet)
            features.append(f.name)
        data.append(label)
        # print(feature[5])
        self.build_tree(data,features)

    def split_feature(self,feature,indexs):
        newFeature = []
        for f in feature:
            data = []
            for i in indexs:
                data.append(f[i])
            newFeature.append(data)
        return newFeature

    def build_tree(self,data,features):
        v, c = np.unique(data[len(data) - 1], return_counts=True)
        if len(c)==1:
            tree.append(v[0])
            return v;
        infoGain = []
        for f in data[0:len(data)-1]:
            infoGain.append(self.InfoGain(f,data[len(data)-1]))
        # print("infoGain : ",infoGain)
        best_feature_index = np.argmax(infoGain)
        best_feature = features[best_feature_index]
        self.features[best_feature_index].visited = 1;
        tree.append(best_feature)
        # print(tree)
        features.remove(features[best_feature_index])
        for val in np.unique(data[best_feature_index]):
            ind = np.where(data[best_feature_index] == val)
            sub_feature = self.split_feature(data,ind[0])
            sub_feature.remove(sub_feature[best_feature_index])
            self.build_tree(sub_feature,features)
            #print(sub_feature)
        # print(best_feature)
    def classify(self, input):
        # takes an array for the features ex. [0, 0, 1, 1, 1]
        # should return 0 or 1 based on the classification
        for i in range(len(tree)):
            for f in range(len(self.features)):
                if tree[i] == self.features[f].name:
                    if input[f] == 0:
                        # print(i)
                        return tree[i+1]
        return tree[len(tree)-1]

# endregion

def print_tree():
    print("\t\t\t\t\t\t\t", tree[0], "\n")
    print("\t\t\t", tree[1], "\t\t\t\t\t\t\t\t", tree[2], "\n")
    print("\t\t\t\t\t\t\t\t", tree[3], "\t\t\t\t\t\t\t", tree[4], "\n")
    print("\t\t\t\t\t\t\t\t\t\t\t\t", tree[5], "\t\t\t\t\t\t\t", tree[6], "\n")
    print("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t", tree[7], "\t\t\t\t\t\t\t", tree[8], "\n")

# region ID3_Main_Fn        pass
# endregion

#################################### Algorithms Main Functions #####################################
# region Search_Algorithms_Main_Fn

def SearchAlgorithm_Main():
    searchAlgo = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.')
    fullPath, path = searchAlgo.BFS()
    print('**BFS**\n Full Path is: ' + str(fullPath) + "\n Path: " + str(path))


# endregion

# region Neural_Network_Main_Fn
def NN_Main():
    learning_rate = 0.1
    threshold = -0.2
    neural_network = NeuralNetwork(learning_rate, threshold)

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1]])

    training_outputs = np.array([[0, 0, 0, 1]]).T

    neural_network.train(training_inputs, training_outputs, 100)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    inputTestCase = [1, 1]

    print("Considering New Situation: ", inputTestCase[0], inputTestCase[1], end=" ")
    print("New Output data: ", end=" ")
    print(neural_network.think(np.array(inputTestCase)))
    print("Wow, we did it!")


# endregion
# region ID3_Main_Fn
def ID3_Main():
    dataset = item.getDataset()
    features = [Feature('age'), Feature('prescription'), Feature('astigmatic'), Feature('tearRate'),
                Feature('diabetic')]
    id3 = ID3(features)
    ID3(features).start(dataset)
    # print_tree()
    
    cls = id3.classify([0, 0, 1, 1, 1])
    print('testcase 1: ', cls)
    cls = id3.classify([1, 1, 0, 0, 0])
    print('testcase 2: ', cls)
    cls = id3.classify([1, 1, 1, 0, 0])
    print('testcase 3: ', cls)
    cls = id3.classify([1, 1, 0, 1, 0])
    print('testcase 4: ', cls)


# endregion

######################## MAIN ###########################33
if __name__ == '__main__':

    SearchAlgorithm_Main()
    NN_Main()
    ID3_Main()
