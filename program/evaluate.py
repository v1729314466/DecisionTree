import json

import numpy as np

#find the class lable, return to the row number
def findClassL(fname, dataDesc, list1):
    for ele in fname:
        if type(ele) is str:
            list1.append(ele)
        elif type(ele) is dict:
            for key, val in ele.items():
                if type(val) is list:
                    findClassL(val, dataDesc, list1)
            for i in dataDesc:
                if i[0] not in list1:
                    for j in range (len(dataDesc)):
                        if i[0] == dataDesc[j][0]:
                            return j


def predict(fname, testdata, dataDesc):
    tree = fname.copy()
    while type(tree) == list:
        for i in range(len(dataDesc)):
            if tree[0] == dataDesc[i][0]:
                break
        tree = tree[1][str(testdata[i - 1])]
    return tree


def main(fname):
    testset = np.loadtxt("../data/test.txt", dtype=int)
    with open('../data/' + fname + ".txt") as f:
        fname = json.load(f)
    with open('../data/dataDesc.txt') as h:
        dataDesc = json.load(h)
    result = 0
    classL = findClassL(fname, dataDesc,[])
    l = testset[classL]
    allAttribute = np.delete(testset, classL, axis = 0)
    allAttribute = allAttribute.T
    for i in range (len(testset[0])):
        if predict(fname, allAttribute[i], dataDesc) == l[i]:
            result += 1
    result = result / len(testset[0]) * 100
    print ("The accuracy of the", f.name[8:-4], "is: %.5f"%result, "%")

main("decitionTree")
main("treeFilePruned")
