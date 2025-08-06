import csv
import pandas as pd
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def read_file(file_name: str):
    """return column name and other datas"""
    with open(file_name) as file:
        csv_file = csv.reader(file, delimiter=",")

        header = next(csv_file)
        data = []

        for row in csv_file:
            data.append(row)

    return header, data


class Node:

    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""

    def __str__(self):
        return self.attribute
    
def subtables(data: np.ndarray, col: int, delete: bool):
    dic = {}
    items = np.unique(data[:, col]) # this returns only the unique elements in that particular column
    count = np.zeros((items.shape[0], 1), dtype=np.int32)
    
    for x in range(items.shape[0]): # similar to for x in range(len(items)) 
        for y in range(data.shape[0]):
            """this loops check the column no: of [features] in the column"""
            if data[y, col] == items[x]:
                count[x] += 1

    for x in range(items.shape[0]):
        dic[items[x]] = np.empty((int(count[x]), data.shape[1]), dtype="|S32")
        pos = 0
        for y in range(data.shape[0]):
            if data[y, col] == items[x]:
                dic[items[x]][pos] = data[y]
                pos += 1
        if delete:
            dic[items[x]] = np.delete(dic[items[x]], col, 1)
    
    return items, dic


def entropy(s):
    items = np.unique(s)

    if items.size == 1:
        return 0
    
    counts = np.zeros((items.shape[0], 1))
    sums = 0

    for x in range(items.shape[0]):
        counts[x] = sum(s == items[x]) / (s.size * 1.0)
    for count in counts:
        sums += -1 * count * math.log(count, 2)
    return sums



def gain_ratio(data: np.ndarray, col: int):
    items, dic = subtables(data, col, delete = False)

    total_size = data.shape[0]
    entropies = np.zeros((items.shape[0], 1))
    intrinsic = np.zeros((items.shape[0], 1))
    for x in range(items.shape[0]):
        ratio = dic[items[x]].shape[0] / (total_size * 1.0)
        entropies[x] = ratio * entropy(dic[items[x]][:, -1])
        intrinsic[x] = ratio * math.log(ratio, 2)
        
    total_entropy = entropy(data[:, -1])

    iv = -1 * sum(intrinsic)
    for x in range(entropies.shape[0]):
        total_entropy -= entropies[x]

    return total_entropy / iv



def create_node(data: np.ndarray, column_names):

    if (len(np.unique(data[:, -1])) == 1):
        """if the target column has only one unique element"""
        node = Node("")
        node.answer = np.unique(data[:,-1])[0]
        return node
    
    gains = np.zeros((data.shape[1] - 1, 1))
    """other wise create a zero filled array with size of heading - 1 (without the target column) ex: [[0], [0], [0], [0]]"""

    for col in range(data.shape[1] -1):
        gains[col] = gain_ratio(data, col)
    
    split = np.argmax(gains)
    node = Node(data[split])
    data = np.delete(data, split, 0)
    items, dic = subtables(data, split, delete=True)


    for x in range(items.shape[0]):
        child = create_node(dic[items[x]], data)
        node.children.append((items[x], child))

    return node

def empty(size):
    return " " * size


def print_tree(node, level):
    if node.answer != "":
        print(empty(level), node.answer)
        return 
    
    print(empty(level), node.attribute)
    for value, n in node.children:
        print(empty(level+1), value)
        print_tree(n, level + 2)

heading, data = read_file("data_ex2.csv")
data = np.array(data)
node = create_node(data, heading)
print_tree(node, 0)