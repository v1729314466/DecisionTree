from math import log
import numpy as np
import json

#decition tree constructor
def tree_constructor(data_set,attrs):
    attrs_size = len(data_set[0])-1
    
    #get total entropy(S) for given set
    etp_S = entropy_calculator(data_set)  
    best_gain = 0
    best_attr = -1
    label_list = []
    
    for data in data_set:
        label_list.append(data[-1])
    
    #count each number of label in list, when it equals to list length, that mean the list is pure
    #and return value of label
    if label_list.count(label_list[0])==len(label_list):
        return int(label_list[0])
    
    #when all attribute has been removed, but the data's labels in the set are still not same,
    #using the most label's value.
    if len(data_set[0])==1:
        return int(max(set(label_list), key = label_list.count))
    
    #choose the best attribute according to it's gain
    for i in range(attrs_size):
        attr_list = []
        for data in data_set:
            attr_list.append(data[i])       
        attr_value = set(attr_list)
        sub_etp = 0
        for value in attr_value:
            subdata_set = splite(data_set,i,value)
            
            #calculate entropy for each subset.
            sub_etp +=len(subdata_set)/(len(data_set))*entropy_calculator(subdata_set) 
        gain = etp_S - sub_etp  
        if (gain>best_gain):  
            best_gain=gain
            best_attr = i

    best_attrLabel = attrs[best_attr]
    #build a list to store decition tree, 
    #and start with the currently best attribute name
    dt_tree = [best_attrLabel]
    #delete the used attribute name for list
    del(attrs[best_attr])
    
    attr_value = []
    for data in data_set:
        attr_value.append(data[best_attr])  
    
    value_set = set(attr_value)
    dic = {}
    for value in value_set:
        #print(value_set)
        subattrs=attrs[:]
        #build a dictionary to key and value
        #key is current attribute value
        #value in dicitonary is corresponding attribute name
        #start recursion to build branch
        #dic = {int(value): tree_constructor(splite(data_set,best_attr,value),subattrs)}
        dic[int(value)] = tree_constructor(splite(data_set,best_attr,value),subattrs)
    dt_tree.append(dic)
        
    return dt_tree

#calculate the given set's entropy
def entropy_calculator(data_set):
    set_size=len(data_set) 
    label_count={}  
    for data in data_set:       
        cur_label=data[-1]
        if cur_label not in label_count.keys():
            label_count[cur_label]=0
        label_count[cur_label]+=1
    result=0
    for key in label_count:
        result-=label_count[key]/set_size*log(label_count[key]/set_size,2)
    return result

#give a attribute, and divide to many branch
#value is attibute's value
def splite(data_set,attr,value): 
    branch_set=[]
    for branch in data_set:
        if branch[attr]==value:           
            #delete the current attribute from the branch
            cut_branch =branch[:attr]
            cut_branch.extend(branch[attr+1:])
            branch_set.append(cut_branch)
    return branch_set


def main():
    #load train set from train.txt
    data_set = np.loadtxt('../data/train.txt')
    data_set = np.array(data_set).T
    a = data_set.tolist()
    data_set = a 
    for ele in data_set:
        ele.append(ele.pop(0))
    #load attribute names for dataDesc.txt
    with open('../data/dataDesc.txt') as f:
        dataDesc = json.load(f)
    #delete label
    del dataDesc[0]
    b = []
    for feat in dataDesc:
        b.append(feat[0])
    dataDesc = b   
    labels = dataDesc
    #start to build the decition tree
    tree = tree_constructor(data_set, labels)
    #store tree in decitionTree.txt
    with open('../data/decitionTree.txt','w') as f:
        json.dump(tree, f)
    #print(tree)
    file_name = 'decitionTree.txt'
    return file_name

main()
