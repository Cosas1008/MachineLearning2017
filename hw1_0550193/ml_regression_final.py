import csv
from math import * 
import numpy as np
from numpy.linalg import inv

class Gaussian_Regression_Model:
    
    name = ''               # name of the object
    index = 0               # for iteration purpose
    weight  = []             # weight
    model = []              # tuples of (means_x,means_y)
    phi = []
    var = []
    
    def __init__(self, name):
        self.name = name
        self.reset()
        
    def add_model(self,means):
        self.model.append((means[0],means[1])) # add tuple of (means_x,means_y)
    
    def add_weight(self, w):
        self.weight.append(w)
            
    def reset(self):
        self.name = ''
        self.model = []
        self.phi = []
        self.weight = []
        self.var = []
        
    def add_var(self,varin):
        self.var = varin
    
    def getVariance(self, data):
        temp_x = 0
        temp_y = 0
        for i in data:
            temp_x += i[0]
            temp_y += i[1]
        mean_x = temp_x / len(data)
        mean_y = temp_y / len(data)
        for i in data:
            temp_x += pow(fabs(i[0] - mean_x),2)
            temp_y += pow(fabs(i[1] - mean_y),2)
        var_x = temp_x / len(data)
        var_y = temp_y / len(data)
        var = (var_x, var_y)
        self.add_var(var)
        
    def training_ML(self, data, t):
        temp = []
        self.weight = [] # clear the previous trained weight
        index = self.index
        self.getVariance(data)
        for i in data:
            for j in self.model:
                x_mean = j[0]
                y_mean = j[1]
                x_var  = self.var[0]
                y_var  = self.var[1]
                result = -pow(i[0] - x_mean,2)/(2*pow(x_var,2)) - pow(i[1] - y_mean,2)/(2*pow(y_var,2))
                temp.append(exp(result))
            index += 1
        self.phi = np.asarray(temp).reshape(index , len(self.model)) # reshape temp to model array (# of data) x (# of trained weight)
        phi_trans = self.phi.transpose() # transpose phi_arr
        phi_inv = inv(np.dot(phi_trans,self.phi)) # inverse of phi_trans and phi_arr
        self.weight = np.dot(np.dot(phi_inv,phi_trans),height)
        
        print 'trained succeed'
        # write weight into weight.csv
        with open('weight.csv','wb') as f:
            writer = csv.writer(f)
            writer.writerows(gcm.weight.tolist())
        f.close()
    
    def training_MAP(self, data, t, lamb):
        temp = []
        self.weight = [] # clear the previous trained weight
        index = self.index
        self.getVariance(data)
        for i in data:
            for j in self.model:
                x_mean = i[0] # 1 to 1081
                y_mean = i[1] # 1 to 1081
                x_var  = self.var[0]
                y_var  = self.var[1]
                result  = -pow(i[0] - x_mean,2)/(2*pow(x_var,2)) - pow(i[1] - y_mean,2)/(2*pow(y_var,2))
                temp.append(exp(result))
            index += 1
        self.phi = np.asarray(temp).reshape(index , len(self.model)) # reshape temp to model array (# of data) x (# of trained weight)
        phi_trans = self.phi.transpose() # transpose phi_arr
        phi_inv = inv(np.dot(phi_trans,self.phi)) # inverse of phi_trans and phi_arr
        lambda_guess = lamb
        self.weight = np.dot(np.dot(np.identity(len(self.model))+phi_inv,phi_trans),height)
        
        print 'trained succeed'
        # write weight into weight.csv
        with open('weight.csv','wb') as f:
            writer = csv.writer(f)
            writer.writerows(gcm.weight.tolist())
        f.close()
        
    def errorFunction(self, dataPred, dataTrue):
        err = 0
        num = 0
        temp = 0
        for i,j in zip(dataPred,dataTrue):
            temp += pow(fabs(i - j),2)
            num += 1
        err = temp / 2 / num
        return err
    
    def errorFunction_MAP(self, dataPred, dataTrue):
        err = 0
        num = 0
        temp = 0
        for i,j in zip(dataPred,dataTrue):
            temp += pow(fabs(i - j),2)
            num += 1
        err = np.sqrt(fabs(temp) / 2 / num)
        return err
    
    def prediction(self, data):
        y_model = []
        num_model = int(np.asarray(self.phi).shape[1])
        index = 0
        phi_data_list = []
        phi_data = []
        for da in data:
            for i in self.model:
                x_mean = i[0] # 1 to 1081
                y_mean = i[1] # 1 to 1081
                x_var  = self.var[0]
                y_var  = self.var[1]
                try:
                    phi_data_list.append(exp(-((pow((da[0] - x_mean),2)/(2*pow(x_var,2))) + (pow((da[1] - y_mean),2)/(2*pow(y_var,2))))))
                except OverflowError:
                    ans = float('inf')
            index += 1
        phi_data = np.asarray(phi_data_list).reshape(index , len(self.model)) # reshape temp to model array (# of data) x (# of model)
        phi_trans = phi_data.transpose()                                 # transpose phi_data
        phi_inv = inv(np.dot(phi_trans,phi_data))                        # inverse of phi_trans and phi_data
        phi_final = np.dot(phi_data,phi_inv)                             # (# of data) x (# of model)
        self.weight = self.weight.reshape(len(gcm.model),1)              #  (# of model) x 1
        y_model = np.dot(phi_final,self.weight)                          # (# of data) x 1
        
        return y_model
 

###################################################################
# Training model with ML method

position = []   # this will store the data of x and y from csv
#tuple the data into (x,y)
with open('X_train.csv', 'rb') as f:
    reader = csv.reader(f,delimiter=',')
    for x,y in reader:
        position.append((float(x),float(y)))
#end of position

gcm = Gaussian_Regression_Model('gcm')

# assign Gaussian model
mu = []
mu_x = [i for i in range(1, 1082, 21)] # 40 sections
mu_y = [i for i in range(1, 1082, 27)] # 60 sections
for x in mu_x:
    for y in mu_y:
        mu.append((x,y))

# add model row
for m in mu:
        gcm.add_model(m) # dump in means
        
        
height = []  # this will store the data of height from csv
#store in vector of 40000 data
with open('T_train.csv','rb') as t:
    ts = csv.reader(t,delimiter='\n')
    for rows in ts:
        height.append(int(rows[0]))
#end of height
height = np.asarray(height).reshape(40000,1)
#modeling
gcm.training_ML(position,height)


# Change format of height[list] into list
temp = height
y_true = []
for i in temp:
    y_true.append(i)
    
#predicting
y_pred = []
y_pred = gcm.prediction(position)


rows = zip(y_pred)
with open('Trained_train_data.csv','wb') as f:
    wr = csv.writer(f, dialect='excel')
    for row in rows:
        wr.writerow(row[0])
f.close()
#calculate error using test data
err = gcm.errorFunction(y_pred,y_true)
err

###################################################################

#Cross-validation

#seperating data
position_1 = position[0:10000]
position_2 = position[10000:20000]
position_2 = position[20000:30000]
position_2 = position[30000:40000]
height_1 =  height[0:10000]
height_2 = height[10000:20000]
height_2 = height[20000:30000]
height_2 = height[30000:40000]

gcm = Gaussian_Regression_Model('gcm')

# assign Gaussian model
mu = []
mu_x = [i for i in range(1, 1082, 27)]
mu_y = [i for i in range(1, 1082, 54)]
for x in mu_x:
    for y in mu_y:
        mu.append((x,y))

# add model row
for m in mu:
        gcm.add_model(m) # dump in means

#modeling
gcm.training_ML(position_1,height_1)


# Change format of height[list] into list
temp = height
y_true = []
for i in temp:
    y_true.append(i)

#predicting
y_pred = []
y_pred = gcm.prediction(position_test)


#calculate error using test data
err = gcm.errorFunction(y_pred,y_true)
err

###################################################################

""" Code for training ML.csv
position_test = []   # this will store the data of x and y from csv
#tuple the data into (x,y)
with open('X_test.csv', 'rb') as f:
    reader = csv.reader(f,delimiter=',')
    for x,y in reader:
        position_test.append((float(x),float(y)))
#end of position


#predicting
y_pred = []
y_pred = gcm.prediction(position_test)


rows = zip(y_pred)
with open('ML.csv','wb') as f:
    wr = csv.writer(f, dialect='excel')
    for row in rows:
        wr.writerow(row)
f.close()
#calculate error using test data
err = gcm.errorFunction(y_pred,y_true)
err
"""
###################################################################

""" Code for training MAP.csv
# MAP method
position = []   # this will store the data of x and y from csv
#tuple the data into (x,y)
with open('X_train.csv', 'rb') as f:
    reader = csv.reader(f,delimiter=',')
    for x,y in reader:
        position.append((float(x),float(y)))
#end of position

gcm = Gaussian_Regression_Model('gcm')

# assign Gaussian model
mu = []
mu_x = [i for i in range(1, 1082, 21)] # 40 sections
mu_y = [i for i in range(1, 1082, 27)] # 60 sections
for x in mu_x:
    for y in mu_y:
        mu.append((x,y))

# add model row
for m in mu:
        gcm.add_model(m) # dump in means
        
        
height = []  # this will store the data of height from csv
#store in vector of 40000 data
with open('T_train.csv','rb') as t:
    ts = csv.reader(t,delimiter='\n')
    for rows in ts:
        height.append(int(rows[0]))
#end of height
height = np.asarray(height).reshape(40000,1)
#modeling
lambda_assign = 0.1        # 0 is the same as ML, test set = 0 0.1 1 10 30
gcm.training_MAP(position,height)


position_test = []   # this will store the data of x and y from csv
#tuple the data into (x,y)
with open('X_test.csv', 'rb') as f:
    reader = csv.reader(f,delimiter=',')
    for x,y in reader:
        position_test.append((float(x),float(y)))
#end of position

#predicting
y_pred = []
y_pred = gcm.prediction(position_test)


rows = zip(y_pred)
with open('MAP.csv','wb') as f:
    wr = csv.writer(f, dialect='excel')
    for row in rows:
        wr.writerow(row)
f.close()
#calculate error using test data
err = gcm.errorFunction_MAP(y_pred,y_true)
err

"""
