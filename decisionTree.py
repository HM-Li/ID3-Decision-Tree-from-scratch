import numpy as np
import os
import sys
# import pickle
class Node:
    '''# a binary tree to store results after each split
    '''
    def __init__(self, index):
        self.index = str(index)
        self.split_attr = None
        self.links = {}
        self.current_dist = {}
        self.node_depth = None
        self.prediction = None
    


class decision_tree_classifier:
    '''# a classifier class to fit, transform, and predict
    X is an array with attribute name at the first line
    '''
    def __init__(self):
        self.X = None
        self.y = None
        self.max_depth = None
        self.current_depth = 0
        self.original_dist = None
        self.tree = {}
        self.included_attributes = []
        # labels conclude unique labels for this training data
        self.labels = None
    ## include a tree class in attributes
    # initialize the instance
    def fit(self, X, y, max_depth):
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.labels = np.unique(y)

    # check whether the set is pure enough or not
    def check_pure(self, data):
        return False
    
    # recursively partition data and load to the tree
    def get_partition(self, data):
        results = {element: (data==element).nonzero()[0] for element in np.unique(data)}
        return results

    def get_distribution(self, data):
        values, counts = np.unique(data,return_counts=True)
        value_counts = {value: count for value, count in zip(values, counts)}
        return value_counts

    # calculate entropy for a list and return the number
    def cal_entropy(self, attribute_list):
        counts = np.unique(attribute_list, return_counts=True)
        entropy = 0
        for count in counts[1]:
            total = sum(counts[1])
            if count.astype('float') != 0.0:
                entropy -= ((count.astype('float')/total)*np.log2(count.astype('float')/total))
        return entropy
    
    # calculate mutual information H(Y|X)
    def cal_information_gain(self,x,y):
        ig = self.cal_entropy(y)
        values, counts = np.unique(x, return_counts=True)
        freqs = counts.astype('float')/len(x)
        for freq, value in zip(freqs, values):
            ig -= freq*self.cal_entropy(y[x==value])
        return ig
    # selet position of attributes from x and y, return None if mutual information is <= 0
    def select_attributes(self, X,y):
        record = []
        for x in X:
            ig = self.cal_information_gain(x[1:],y)
            record.append(ig)
        if max(record)>0:
            return record.index(max(record))
        return None

    # check if over deep
    def check_stop(self):
        check = ((self.current_depth > self.max_depth)|(len(self.included_attributes)==self.X.shape[0]))
        # print(self.current_depth , self.max_depth)
        # print(len(self.included_attributes),self.X.shape[0])
        return check

    # print distribution in the required format
    def print_dist(self, dist):
        str_print=''
        keys = list(dist.keys())
        # in order to print the distribution in the correct order
        keys.sort()
        for key in keys:
            count = dist.get(key)
            str_print+='{0} {1}/'.format(count, key)
        return '['+str_print[:-1]+']'

    # predict using majority vote algorithm and return the prediction result list
    def majority_predict(self, attribute_list):
        counts = np.unique(attribute_list, return_counts=True)
        idx = np.argmax(counts[1])
        prediction = counts[0][idx]
        return prediction
    
    # split function
    def recursively_split(self, X, y):
        self.current_depth +=1
        # create a node
        current_node = str(len(self.tree))
        node = Node(str(len(self.tree)))
        node.current_dist = self.get_distribution(y)
        # add label to the dist if the node is a leaf, containing only one label
        for label in self.labels:
            if node.current_dist.get(label) == None:
                node.current_dist = dict(node.current_dist, **{label:0})
        ## check stop function
        if self.check_stop():
            node.prediction = self.majority_predict(y)
            ## load the new node into the tree
            self.tree = dict(self.tree, **{node.index: node})
            return
        ## select attributes and get the index of that attribute
        ### if returns none then break
        pos_selected_attr = self.select_attributes(X,y)
        if pos_selected_attr == None:
            node.prediction = self.majority_predict(y)
            ## load the new node into the tree
            self.tree = dict(self.tree, **{node.index: node})
            return
        selected_attr = X[pos_selected_attr][0]
        self.included_attributes.append(selected_attr)
        node.node_depth = self.current_depth
        node.split_attr = X[pos_selected_attr][0].strip()
        ## partition using that attribute
        partitions = self.get_partition(X[pos_selected_attr][1:])
        ## load the new node into the tree
        self.tree = dict(self.tree, **{node.index: node})
        # remove partitioned attribute from X
        X = np.delete(X, pos_selected_attr, axis=0)
        ## get split result and recurse until
        for partition, index in partitions.items():
            # add the link to the child to the node on the tree
            self.tree.get(current_node).links = dict(self.tree.get(current_node).links, **{partition:str(len(self.tree))})
            y_sub = y.take(index, axis=0)
            # add one position for titles for X
            index = index+1
            # include title with X_sub
            pos = [0]
            pos.extend(index)
            X_sub = X.take(pos, axis=1)
            self.recursively_split(X_sub, y_sub)
            self.current_depth -=1
        # before going to the upper level, remove the current attr from the included list
        self.included_attributes.remove(selected_attr)

    # recursively print out nodes
    def recursively_print_node(self,node):
#     if node is a leaf, return
        if node.split_attr == None:
            return
        for splitter, link in node.links.items():
            next_node = self.tree.get(link)
            print_str = '| '*node.node_depth+node.split_attr+' = '+splitter+': '+self.print_dist(next_node.current_dist)
            print(print_str)
            self.recursively_print_node(next_node)

    # print out the tree
    def print_tree(self):
        root = self.tree.get('0')
        print(self.print_dist(root.current_dist))
        self.recursively_print_node(root)
    
    # recursively check nodes for an entry
    def check(self, node, x_dict):
        if(node.prediction!=None):
            return node.prediction
        value = x_dict.get(node.split_attr)
        link = node.links.get(str(value))
        next_node = self.tree.get(link)
        return self.check(next_node, x_dict)
    
    # predict
    def predict(self, X):
        attrs = [attr.strip() for attr in X[0]]
        predictions = []
        for x in X[1:]:
            x_dict = dict(zip(attrs, x))
            label = self.check(self.tree.get('0'), x_dict)
            predictions.append(label)
        return predictions

    # do things
    def transform(self):
        
        # get distribution
        self.original_dist = self.get_distribution(y)
        self.print_dist(self.original_dist)

        # recursive split
        self.recursively_split(X,y)

        # print the tree
        self.print_tree()
        
    def fit_transform(self, X, y, max_depth):
        self.fit(X, y, max_depth)
        self.transform()
    
    # calculate the error rate for two lists and return the number
    def cal_error_rate(self, attribute_list, prediction_list):
        assert((type(attribute_list) is np.ndarray) 
            & (type(prediction_list) is np.ndarray) 
            & (attribute_list.size==prediction_list.size)),"Input list format problem."
        right_count = (attribute_list==prediction_list).sum()
        error_rate = (len(attribute_list)-right_count)/len(attribute_list)
        return error_rate
         
# read a file and return a list
def read_file(path):
    with open(path,'r') as file:
        lines = file.read().splitlines()
        contents = np.array([line.split(',') for line in lines])
    return contents

#  write to a file
def write_file(list, path):
    with open(path, 'w') as file:
        for line in list:
            file.write(line+'\n')


if __name__ == '__main__':
    # define paths
    train_input_path = os.path.join('./',sys.argv[1])
    test_input_path = os.path.join('./',sys.argv[2])
    max_depth = int(sys.argv[3])
    train_output_path = os.path.join('./',sys.argv[4])
    test_output_path = os.path.join('./',sys.argv[5])
    metrics_output_path = os.path.join('./',sys.argv[6])

    # train_input_path = os.path.join('./','education_train.csv')
    # test_input_path = os.path.join('./','education_test.csv')
    # max_depth = 3   
    # train_output_path = os.path.join('./','edu_3_train.labels')
    # test_output_path = os.path.join('./','edu_3_test.labels')
    # metrics_output_path = os.path.join('./','edu_3_metrics.txt')

    # load data
    train_data = read_file(train_input_path)
    test_data = read_file(test_input_path)
    # get X and y from the dataset
    X = train_data.T[:-1]
    y = train_data.T[-1][1:]
    # fit classifier using the data with X and y
    clf = decision_tree_classifier()
    # transform the classifier to train the model
    clf.fit_transform(X,y,max_depth)
    
    # predict for train data
    prediction = clf.predict(X.T)
    # write predictions
    write_file(prediction, train_output_path)
    # predict for test data
    test_prediction = clf.predict(test_data.T[:-1].T)
    # write predictions
    write_file(test_prediction, test_output_path)
    # calculate metrics
    train_error = clf.cal_error_rate(np.array(y), np.array(prediction))
    test_error = clf.cal_error_rate(np.array(test_data.T[-1][1:]), np.array(test_prediction))
    error_list = ['error(train): '+str(train_error), 'error(test): '+ str(test_error)]
    # write metrics
    write_file(error_list, metrics_output_path)
    # with open(os.path.join('./','tree.pkl'), 'wb') as f:
    #     pickle.dump(clf.tree, f)
