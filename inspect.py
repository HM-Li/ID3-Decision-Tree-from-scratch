
import os
import sys
import numpy as np





# predict using majority vote algorithm and return the prediction result list
def majority_predict(attribute_list):
    values, counts = np.unique(attribute_list, return_counts=True)
    if len(values) == 0:
        return []
    idx = np.argmax(counts)
    prediction = values[idx]
    prediction_results = [prediction for attr in attribute_list]
    # = np.full(len(attribute_list),prediction)
    return prediction_results





# calculate entropy for a list and return the number
def cal_entropy(attribute_list):
    counts = np.unique(attribute_list, return_counts=True)
    entropy = 0.0
    for count in counts[1]:
        total = sum(counts[1])
        if count.astype('float') != 0.0:
            entropy -= ((count.astype('float')/total)*np.log2(count.astype('float')/total))
    return entropy





# calculate the error rate for a list and return the number
def cal_error_rate(attribute_list, prediction_list):
    assert((type(attribute_list) is np.ndarray) 
           & (type(prediction_list) is np.ndarray) 
           & (attribute_list.size==prediction_list.size)),"Input list format problem."
    if (attribute_list.size == 0) | (prediction_list.size == 0):
        return 0.0
    right_count = (attribute_list==prediction_list).sum()
    error_rate = (float(len(attribute_list))-right_count)/len(attribute_list)
    return error_rate





# read a file and return a list
def read_file(path):
    with open(path,'r') as file:
        lines = file.read().splitlines()
        contents = np.array([line.split(',') for line in lines])
    return contents





# read a file list and return a column
def read_column(contents, column):
    title = np.array([attribute.strip() for attribute in contents[0]])
    idx = np.where(title==column)
    # idx is a tuple of lists
    if idx[0].size == 0:
        print('no such column exists')
        return None
    elements = []
    [elements.extend(line[idx[0]]) for line in contents[1:]]
    return np.array(elements)




if __name__ == '__main__':
    # define paths
    input_path = os.path.join('./',sys.argv[1])
    output_path = os.path.join('./',sys.argv[2])
    # input_path = os.path.join('./','small_test.csv')
    # output_path = os.path.join('./','test.txt')

    input_data = read_file(input_path)
    # taken the last column is the label as granted
    label_column_name = input_data[0][-1].strip()
    # column = read_column(input_data, label_column_name)
    column = input_data.T[-1][1:]
    entropy = cal_entropy(column)
    prediction = majority_predict(column)
    error_rate = cal_error_rate(np.array(column), np.array(prediction))
    # write results to a file
    with open(output_path, 'w') as output_file:
        output_file.write('entropy: '+str(entropy)+'\n'+
                        'error: '+str(error_rate)+'\n')

