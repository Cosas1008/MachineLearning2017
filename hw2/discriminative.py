import numpy as np
from PIL import Image
from matplotlib.mlab import PCA
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.pyplot as plt
from operator import add

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

def training(dir_address,cross_number,PCA_num):
    print "> Training start"
    print "> Reading files.."
    bb = walkfile(str(dir_address))
    bb_pca = PCA(bb)
    #initial
    training = []
    T_training = []
    testing = []
    T_testing = []
    class_cnt = 3
    training_num = 2500.0

    # i have 3000 rows, divid it into 5 chunk
    T_train = [[1, 0, 0] ]* 1000 + [[0, 1, 0]] * 1000 + [[0,0,1]] * 1000
    index = 0 # testing data index
    index_T = 0 # training data index
    data = 3000

    print "> Divided the data into 5 chunks for the " + str(cross_number) + "th cross validation set."
    for i in range(0,data):
        if (i % 5) != cross_number:
            training.append(bb_pca.Y[i])
            T_training.append(T_train[i])
            index_T += 1
        else:
            testing.append(bb_pca.Y[i])
            T_testing.append(T_train[i])
            index += 1

            
    N = PCA_num
    training = np.argpartition(training, -N, axis=1)[:, -N:]
    testing  = np.argpartition(testing, -N, axis=1)[:, -N:]
    
    
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

    print "> Getting W Matrix.."

    training_num = len(training)
    feature_num = len(training[0])


    w_matrix = np.zeros(((class_cnt - 1)*feature_num,1 )) # feature*class_cnt x 1

    phi_matrix = np.zeros((training_num*(class_cnt-1),feature_num*(class_cnt-1)))

    for i in range(0,class_cnt-1):
        rowstart = i*training_num
        colstart = i*feature_num
        for j in range(0,training_num):
            phi_matrix[rowstart+j,colstart:colstart+feature_num] = training[j]

    R_matrix = np.zeros ((training_num*(class_cnt-1), training_num*(class_cnt-1)));
    p = np.zeros (((class_cnt-1)*training_num, 1));

    sum=0;
    y = np.asarray(T_training).reshape(len(T_training)*len(T_training[0]),1)[len(T_training)::,0]

    diff = 100.0
    time =0

    while (diff > 0.01 and time <1000):
        print 'In the iteration %f' %(diff)

        for j in range(0,training_num):
            sums = 1
            for i in range(0,class_cnt-1):
                # w_matrix((i-1)*feature_num+1:i*feature_num,1)' * X_train(j,:)'
                temp_c = np.exp(np.dot(np.asmatrix(w_matrix[i*feature_num:(i+1)*feature_num,0]),np.asmatrix(training[j]).transpose()))
                p [(i-1)*training_num+j,0] = temp_c
                sums += temp_c
            # print 'sum %d : %d \n' %(i, sums)

            for i in range(1,class_cnt-1):
                p[(i-1)*training_num+j,0] /= sums

        for i in range(0,class_cnt-1):
            for j in range(0,class_cnt-1):
                for k in range(0,training_num):
                    if (i==j):
                        R_matrix[k+i*training_num, k+j*training_num]= np.dot(p [i*training_num+k, 0],(1-p[i*training_num+k, 0]))
                    else:
                        R_matrix[k+i*training_num, k+j*training_num]=-1* np.dot(p [i*training_num+k, 0],(1-p[i*training_num+k, 0]))
        # diff_matrix = pinv(transpose(Phi_matrix) * R_matrix * Phi_matrix) * transpose(Phi_matrix) * (y-p);
        diff_matrix = np.dot(np.linalg.pinv(np.dot(np.dot(phi_matrix.transpose(),R_matrix),phi_matrix)),np.dot(phi_matrix.transpose(),(np.asmatrix(y).transpose()-p)))
        diff =0
        w_matrix += diff_matrix
        for i in range(1,(class_cnt-1)*feature_num):
            diff += np.asarray(diff_matrix)[i][0]

        diff = fabs (diff)
        time += 1
    print 'training finished!'
    evaluate(w_matrix,feature_num,class_cnt,testing,T_testing)

        
def evaluate(w_matrix,feature_num,class_cnt,testing,T_testing):
    print "> Evaluating.."
    right = 0.0
    answer = []
    wrong = 0
    for j in range(0,len(testing)):
        max_value = 0
        max_index = 0
        sums = 0
        for i in range(0,class_cnt):
            a = np.resize(w_matrix[i*feature_num:(i+1)*feature_num,0],(900,1))
            b = np.resize(testing[j],(1,900)).transpose()
            c = np.dot(a.transpose(),b)
            temp = np.exp(c[0][0])
            sums += temp
            if temp >= max_value:
                max_value = temp
                max_index = i

            if (1 > max_value):
                max_value = 1
                max_index = class_cnt 

        if (T_testing[j][max_index - 1] == 1):
            right += 1
        else:
            wrong += 1

    print 'Discriminative model cross validation get ' + str((right/len(testing))*100) + '%'

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print 'Usage: python2.7 generative.py ' + '<training faces dir> [<demo faces dir>] Number_of_feature'
        sys.exit(1)

    dir_address = str(sys.argv[2])                                       # The data dir
    cross_number = str(sys.argv[3])                                      # Cross validation number
    N =  str(sys.argv[4])
	training(dir_address,cross_number,N)