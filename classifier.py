import csv
import numpy as np
import random
import scipy.stats as st

NUM_FEATS = 274
num_nodes = 0
num_leaves = 0
p_value = 0.5
output_file = 'test_output.csv'

class treeNode():
    def __init__(self, data='T', children=[-1] * 5):
        self.nodes = list(children)
        self.data = data

def read_feature_files(data_file, df):
    with open(data_file, 'r') as tf:
        reader = csv.reader(tf)
        for row in reader:
            row = map(int, row[0].split())
            df.append(row)

def read_labels(label_file, df):
    with open(label_file, 'r') as tf:
        reader = csv.reader(tf)
        for row in reader:
            df.append(int(row[0]))


def entropy(col):
    value, counts = np.unique(col, return_counts=True)
    freqs = counts.astype('float')/len(col)
    return -freqs.dot(np.log2(freqs))


def info_gain(base_entropy, labels, attr_value):
    result = base_entropy

    value, counts = np.unique(attr_value, return_counts=True)
    freqs = counts.astype('float')/len(attr_value)

    for k, v in zip(freqs, value):
        result -= k * entropy(labels[attr_value == v])

    return result


def split(data, labels, attr):
    new_df, new_labels = [], []

    for i in range(1, 6):
        new_df.append(dataset[dataset[:, attr] == i])
        new_labels.append(labels[labels[:, attr] == i])

    return new_df, new_labels

def chi_square_test(labels, new_labels, p_thres):
    N = len(labels)
    pos = np.sum(labels)
    n = N - pos

    S = 0.0
    m = 0

    for i in range(5):
        if new_labels[i].size == 0:
            continue

        m += 1
        pi_bar = float(pos)*new_labels[i].size/N
        ni_bar = float(neg)*new_labels[i].size/N

        pi = len(new_labels[i][:] == 1)
        ni = len(new_labels[i][:] == 0)

        score = ((pi_bar - pi)**2 / pi_bar) + ((ni_bar - ni)**2 / ni_bar)

    p_value = st.chi2.sf(score, df=(m-1))
    return isnan(p_value) or p_value >= p_thres

def create_tree(data, labels, pvalue):
    global num_nodes
    global num_leaves

    if labels.size == 0:
        num_leaves += 1

        if (random.randint(0,1) == 0):
            return treeNode('T', [])
        else:
            return treeNode('F', [])

    if len(np.unique(labels)) == 1:
        num_leaves += 1
        return treeNode(data='T' if labels[0] else 'F')

    base_entropy = entropy(labels)
    selected_attribute = np.argmax([info_gain(base_entropy, labels, attribute) for attribute in data.T])

    new_datasets, new_labels = split(data, labels, selected_attribute)
        
    if chi_square_test(labels, new_labels, pvalue):
        node_label = st.mode(labels)[0][0]
        curr_label = 'T' if node_label == 1 else 'F'
        num_leaves += 1
        root = TreeNode(data=curr_label)
    else:
        num_nodes += 1
        root = TreeNode(data=str(selected_attribute + 1))
        for i in range(5):
            root.nodes[i] = create_tree(new_datasets[i], new_labels[i], pvalue)
    
    return root

def evaluate(root, datapt):
    if root.data == 'T':
        return 1
    if root.data == 'F':
        return 0

    return evaluate(root.nodes[datapt[int(root.data)-1]-1], datapt)

if __name__ == '__main__':
    X_train, Y_train, X_test = [], [], []

    read_feature_files('trainfeat.csv', X_train)
    read_feature_files('testfeat.csv', X_test)
    read_labels('trainlabs.csv', Y_train)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)

    print("Training\n")

    s = create_tree(X_train, Y_train, p_value)
    
    print("Done training\n")


    print("Testing\n")

    Y_test = []
    for i in range(0, len(X_test)):
        Y_test.append([evaluate(s, X_test[i])])

    with open(output_file, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(Y_test)

    print("Done\n")
