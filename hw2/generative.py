import os
import cv2
import sys
import csv
import shutil
import random
import numpy as np
from PIL import Image
from matplotlib.mlab import PCA
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from operator import add
"""
Example Call:
    $> python2.7 generative.py training_face_dir cross_number
"""


def readFile(returnList, bmp_image_address):
    bmp_image = Image.open( bmp_image_address )

    line_horizon = (bmp_image.getpixel((i_horizon, i_vertical)) for i_vertical in range(bmp_image.height) for i_horizon in range(bmp_image.width))
    for x in range(bmp_image.height):
        for y in range(bmp_image.width):
            returnList.append(next(line_horizon))
    return returnList

def walkfile(dir_name):
    addr = []
    Row = 0
    ## read the file cwd and store in addr
    for root, dirs, files in os.walk("Data_Train"):
        for file in files:
            if file.endswith(".bmp"):
                addr.append(os.path.join(root, file))  
                Row += 1
    ## read the file according to addr
    bmp = []
    for address in addr:
        readFile(bmp, address)
    Col =  len(bmp)/Row
    bmparr = np.asarray(bmp).reshape(Row, Col) # reshape it into row x col array
    return bmparr

def training(dir_address,cross_number):
    print "> Training start"
    print "> Reading files.."
    bb = walkfile(str(dir_address))
    #initial
    training = []
    T_training = []
    testing = []
    T_testing = []
    class_cnt = 3
    training_num = 3000.0

    # i have 3000 rows, divid it into 5 chunk
    T_train = [[1, 0, 0] ]* 1000 + [[0, 1, 0]] * 1000 + [[0,0,1]] * 1000
    index = 0 # testing data index
    index_T = 0 # training data index
    data = 3000

    print "> Divided the data into 5 chunks for the " + str(cross_number) + "th cross validation set."
    for i in range(0,data):
        if (i % 5) != cross_number:
            training.append(bb[i])
            T_training.append(T_train[i])
            index_T += 1
        else:
            testing.append(bb[i])
            T_testing.append(T_train[i])
            index += 1

    class_trainging_cnt = [0,0,0]
    for i in T_training:
        ind = i.index(1)
        class_trainging_cnt[ind] += 1

    mean_class = np.zeros((class_cnt,len(training[1])))
    
    for i in range(0,len(T_training)):
        ind = T_training[i].index(1)
        mean_class[ind] += training[i]
        map(add, mean_class[ind],training[i])

    mean_class /= (training_num * class_cnt)
    
    print "> Getting Covariance Matrix.."
    
    cov_matrix = np.zeros((len(training[0]), len(training[0]))) # feature * feature
    for i in range(0,len(training)):
        temp_cov = []
        ind = T_training[i].index(1)
        temp_cov = training[i] - mean_class[ind]
        cov_matrix += np.dot(np.asmatrix(temp_cov).transpose(),np.asmatrix(temp_cov))

    cov_matrix /= training_num
    cov_inv = np.linalg.inv(cov_matrix)
    wk = []
    datanum = [1000]*3
    means = mean_class.transpose()
    W_matrix = np.dot(cov_inv,means)
    index = 0
    for i in datanum:
        wk.append(np.dot(np.asmatrix(means[:,index]),np.dot(np.asmatrix(cov_inv),np.asmatrix(means[:,index]).transpose()))[0,0] + (i / training_num))
        index += 1
    print "> Training Success!"
    evaluate(wk,W_matrix,testing,class_cnt)

        
def evaluate(wk,W_matrix,testing,class_cnt):
    print "> Evaluating.."
    right = [0.0 ,0.0 ,0.0]
    wrong = [0.0 ,0.0 ,0.0]
    color = []
    class_count = [0] * class_cnt
    predict_count = [0] * class_cnt
    for index in range(0,len(testing)):
        #A = np.dot(W_matrix.transpose(),np.asmatrix(testing[0]).transpose()).transpose()+ wk
        A =  wk + np.dot(W_matrix.transpose(),np.asmatrix(testing[index]).transpose()).transpose()
        ans = []
        for i in range(0,class_cnt):
            ans.append(np.exp(A.item(i)))
        predict = ans.index(max(ans))
        actual = T_testing[index].index(1)
        # print str(predict) + ' : ' +str(actual) 
        color.append(predict)
        predict_count[predict] += 1
        class_count[actual] += 1
        if predict != actual:
            wrong[actual] += 1
        else:
            right[actual] += 1
            
    for i in range(0,len(class_count)):
        print 'Cross validation for class '+ str(i) + ' get ' + str((right[i]/class_count[i])*100) + '%'

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print 'Usage: python2.7 generative.py ' + '<training faces dir> [<demo faces dir>]'
        sys.exit(1)
    dir_address = str(sys.argv[1])                                       # The data dir
    cross_number = str(sys.argv[2])                                      # Cross validation number
    training(dir_address,cross_number)