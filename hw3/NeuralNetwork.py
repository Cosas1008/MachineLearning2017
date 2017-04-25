import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
import os
from PIL import Image
import sys
class Data(object):
    
        
    def __init__(self, filedir, number_PCA, number_cross):
        print '> Initializing started'
        self.file_dir = filedir
        self.N = number_PCA
        self.cross_number = number_cross
        self.walkfile(self.file_dir)
        print '> Done initialized data with PCA of '+ str(self.N) +'!'
    
    def readFile(self,returnList, bmp_image_address):
        bmp_image = Image.open( bmp_image_address )

        line_horizon = (bmp_image.getpixel((i_horizon, i_vertical)) for i_vertical in range(bmp_image.height) for i_horizon in range(bmp_image.width))
        for x in range(bmp_image.height):
            for y in range(bmp_image.width):
                returnList.append(next(line_horizon))
        return returnList
    
    def walkfile(self, dirname):
        addr = []
        Row = 0
        ## read the file cwd and store in addr
        for root, dirs, files in os.walk(dirname):
            for file in files:
                if file.endswith(".bmp"):
                    addr.append(os.path.join(root, file))  
                    Row += 1
        ## read the file according to addr
        print '> Walk file start...'
        bmp = []
        for address in addr:
            self.readFile(bmp, address)
        Col =  len(bmp)/Row
        bmparr = np.asarray(bmp).reshape(Row, Col) # reshape it into row x col array
        print '> walkfile done!'
        self.data_PCA(bmparr)
    
    def data_PCA(self, bb):

        print "> Partitioning data start"
        bb = PCA(bb)
        #initial
        self.training = []
        self.T_training = []
        self.testing = []
        self.T_testing = []
        class_cnt = 3
        training_num = 3000.0

        # i have 3000 rows, divid it into 5 chunk
        self.T_train = [[1, 0, 0]]* 1000 + [[0, 1, 0]] * 1000 + [[0,0,1]] * 1000
        index = 0 # testing data index
        index_T = 0 # training data index
        data = 3000

        print "> Divided the data into 5 chunks for the " + str(self.cross_number) + "th cross validation set."
        for i in range(0,data):
            if (i % 5) != self.cross_number:
                self.training.append(bb.Y[i])
                self.T_training.append(self.T_train[i])
                index_T += 1
            else:
                self.testing.append(bb.Y[i])
                self.T_testing.append(self.T_train[i])
                index += 1

        self.training = np.argpartition(self.training, -self.N, axis=1)[:, -self.N:]
        self.testing  = np.argpartition(self.testing, -self.N, axis=1)[:, -self.N:]
        
        print '> Done data division!'
        


class Network(object):

    def __init__(self, Data,batchNum = 4, epochNum = 1, learningRate = 0.01):
        # data initialization
        self.X = Data.training # 2400 x 2
        self.y = np.asarray(Data.T_training) # 2400 x 3
        # initialize weights randomly with mean 0
        self.weight_layer10 = 2*np.random.random((2,5)) -1 # 2 x 5
        # initialize weights randomly with mean 0
        self.weight_layer11 = 2*np.random.random((5,3)) -1 # 5 x 3
        
        self.bias_layer10 = 2*np.random.random((1,5)) - 1 # hidden layer
        self.bias_layer11 = 2*np.random.random((1,3)) - 1 # output layer
        
        self.batch = batchNum # batch number
        self.n_epoch = epochNum # epoch number
        self.diff = 100.0
        self.rate = learningRate 
        self.rate2 = learningRate / 10
        self.correct_num = 0.0
        self.iter = 0
        
        # Momentum parameters
        self.mu = np.random.randn(1,4)
        self.v = np.zeros((1,4))
        
        # Old values
        self.weight_layer10_old = np.zeros((2,5))
        self.weight_layer11_old = np.zeros((5,3))
        self.bias_layer10_old = np.zeros((1,5))
        self.bias_layer11_old = np.zeros((1,3))
        self.diff_old = 100.0
        
    def forward_propagate(self,row):
        # training start
        # X : 2400 x 2 / y : 2400 x 3  / weight_layer10 : 2 x 5 / weight_layer11 : 5 x 3
        x = np.asarray(self.X[row]).reshape(len(self.X[row]),1)
        self.a0 = self.InnerProduct_ForProp(self.weight_layer10,x,self.bias_layer10)
        # a0 = -(np.dot(X[row],self.weight_layer10))
        self.y0 = nonlin(self.a0).T # 2400(1) x 5
        self.a1 = self.InnerProduct_ForProp(self.weight_layer11,self.y0,self.bias_layer11)
        self.y1 = nonlin(self.a1,3) # nonlinear : rectified / size : 2400(1) x 3
        '''
        y1_delta = (y[row] - y1)*(y1*(1-y1)) # 2400(1) x 3
        y0_delta = y1_delta.dot(self.weight_layer11.T) * (y0 * (1 - y0)) # 2400(1) x 5
        self.weight_layer11 += y0.T.dot(y1_delta) # 5 x 2400(1) x 3
        self.weight_layer10 += X.T.dot(y0_delta) # 2 x 2400(1) x 5
        # debuging
        print self.a0.shape
        print self.y0.shape
        print self.a1.shape
        print self.y1.shape
        '''
    
    def backpropergation(self,row):
        
        dE_2 = self.rectified_BackProp(self.y1, self.y[row])
        self.dE2 = dE_2.T
        dEdz_1, self.dEdW_11, self.dEdb_11 = self.InnerProduct_BackProp(self.dE2,self.y0.T,self.weight_layer11,self.bias_layer11)
        dEda_1 = self.sigmoid_BackProp(dEdz_1.T,self.a0)
        self.dE1 = np.asarray(dEda_1).reshape(5,1)
        dEdz_0, self.dEdW_01, self.dEdb_01 = self.InnerProduct_BackProp(self.dE1,np.asarray(self.X[row]).reshape(1,2),self.weight_layer10,self.bias_layer10)
        
        
    def training(self):
        training_data = 0
        while(self.diff > 0.01 or self.iter < 160 and (self.correct_num != len(self.y))):
            
            # epoch
            for row in range(len(self.X)/self.batch):
                # sum_error = 0
                for batch in range(self.batch):
                    #  Forward-propagation
                    self.forward_propagate(row*(batch+1) + batch)
                    #  Backward-propagation
                    self.backpropergation(row)
                    
                # update weight and bias
                self.update_weight()
                    
                # add the count
                training_data += 1
                #break # debug
                
            # update value and sum of difference
            self.update_value()
            # add iteration
            self.iter += 1
            
            self.correct_num = 0.0
            self.wrong_num = 0.0
            # testing with training data
            self.traing_testing()
            
            print 'Iteration: %d' % (self.iter)
            print 'Difference : ' + str(self.diff)
            print 'Correctness : %d. Training using : %d' %(self.correct_num, training_data)
            #break # debug
            
        print 'Correct number : %d' % (self.correct_num)
        print 'Wrong number : %d' % (self.wrong_num)
        print 'Correct rate : %d' % (self.correct_num/len(self.y))
        print '> Done training.'
        
                
    def traing_testing(self):
        for row in range(len(self.X)):
            A0 = self.InnerProduct_ForProp(self.weight_layer10,self.X[row],self.bias_layer10)
            Y0 = nonlin(A0).T
            A1 = self.InnerProduct_ForProp(self.weight_layer11,Y0,self.bias_layer11)
            ans = nonlin(A1,3)
            true = np.argmax(self.y[row])
            predict = np.argmax(ans)
            if (true == predict):
                self.correct_num += 1
            else:
                self.wrong_num += 1
    
    def testing(self, a):
        X_testing = a.testing
        T_testing = np.asarray(a.T_testing)
        
    def InnerProduct_ForProp(self, W, x, bias):
        # print W.shape
        # print x.shape
        y = np.dot(W.T,x).T + bias
        return y

    def InnerProduct_BackProp(self, dE,y,W,bias):
        dEx = np.dot(W,dE)
        dEw = np.dot(y.T,dE.T)
        dEb = dE.T
        
        return dEx, dEw, dEb

    def sigmoid_BackProp(self, predict, true):
        sig = nonlin(true)
        temp = nonlin(sig,2)
        predict *= temp
        return predict
    
    def rectified_BackProp(self, dE, a):
        temp = []
        for i in a:
            if(i < 0):
                temp.append(0)
            elif(i > 0):
                temp.append(1)
            else:
                temp.append(0.5)
        temp = np.asarray(temp).reshape(1,3)
        dEx = np.multiply(dE,temp)
        return dEx
    
    def update_weight(self):
        # print '> Done backpropergation'
        # Vanilla update
        
        self.weight_layer10 -= self.rate  * self.dEdW_01
        self.bias_layer10 -= self.rate * self.dEdb_01
        self.weight_layer11 -= self.rate2 * self.dEdW_11
        self.bias_layer11 -= self.rate2 * self.dEdb_11
        '''
        # Momentum update
        for i1 in range(len(self.dEdW_01)):
            for i2 in range(len(self.dEdW_01[0])):
                self.v[0][0] = self.mu.item(0) * self.v.item(0) - self.rate * self.dEdW_01[i1][i2]
                self.weight_layer10[i1][i2] += self.v.item(0)
        for j in range(len(self.dEdb_01[0])):
            self.v[0][1] = self.mu.item(1) * self.v.item(1) - self.rate * self.dEdb_01[0][j]
            self.bias_layer10[0][j]   += self.v.item(1)
        for k1 in range(len(self.dEdW_11)):
            for k2 in range(len(self.dEdW_11[0])):
                self.v[0][2] = self.mu.item(2) * self.v.item(2) - self.rate2 * self.dEdW_11[k1][k2]
                self.weight_layer11[k1][k2] += self.v.item(2)
        for l in range(len(self.dEdb_11[0])):
            self.v[0][3] = self.mu.item(3) * self.v.item(3) - self.rate2 * self.dEdb_11[0][l]
            self.bias_layer11[0][l]   += self.v.item(3)
        '''
    def update_value(self):
        # calculate the different
        self.diff  = np.sum(np.abs(self.weight_layer10 - self.weight_layer10_old))
        self.diff += np.sum(np.abs(self.bias_layer10 - self.bias_layer10_old))
        self.diff += np.sum(np.abs(self.weight_layer11 - self.weight_layer11_old))
        self.diff += np.sum(np.abs(self.bias_layer11 - self.bias_layer11_old))
        
        # update the old value
        self.weight_layer10_old = self.weight_layer10
        self.bias_layer10_old = self.bias_layer10
        self.weight_layer11_old = self.weight_layer11
        self.bias_layer11_old = self.bias_layer11
        
        # update the old difference
        self.diff_old = self.diff
        
    # end of class Network
    
def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))
    
# default : sigmoid function / 3 :  Rectified linear function
def nonlin(x,choice=1):
    if(choice==1):
        return 1/(1+np.exp(-x))
    if(choice==2):
        return x*(1-x) 
    # rectified
    if(choice==3):
        temp = []
        for i in x[0]:
            if (i > 0):
                temp.append(i)
            else:
                temp.append(0)
        temp = np.asarray(temp).reshape(1,len(x[0]))
        return temp

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print 'Usage: python2.7 NeuralNetwork.py ' + '<att faces dir>' + ' PCA_number ' + ' cross_number '
        sys.exit(1)
    
    #initial
	PCA_number = 2
	cross_number = 0
	Data = Data(str(sys.argv[1]), sys.argv[2], sys.argv[3])
	# str(sys.argv[1]) = "Data_Train"
	# net = Network([2, 3, 1])

	Networks = Network(Data)
	Networks.training()
	performance = Networks.correct_num / (Networks.correct_num + Networks.wrong_num) * 100.0
	print 'Result performance: %.2f % ' % (performance)
	sys.exit(1)
