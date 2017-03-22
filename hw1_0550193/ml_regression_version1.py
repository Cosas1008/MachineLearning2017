class Gaussian_Regression_Model:
    
    name = ''               # name of the object
    index = 0               # for iteration purpose
    weight  = []             # weight
    model = []              # tuples of (means_x,means_y)
    phi = []
    
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
        return var
        
    def training(self, data, t):
        temp = []
        self.weight = [] # clear the previous trained weight
        index = self.index
        var = self.getVariance(data)
        for i in data:
            for j in self.model:
                x_mean = j[0]
                y_mean = j[1]
                x_var  = var[0]
                y_var  = var[1]
                result  = -pow(i[0] - x_mean,2)/x_var - pow(i[1] - y_mean,2)/y_var
                temp.append(exp(result))
            index += 1
        self.phi = np.asarray(temp).reshape(index , len(self.model)) # reshape temp to model array (# of data) x (# of trained weight)
        phi_trans = self.phi.transpose() # transpose phi_arr
        phi_inv = inv(np.dot(phi_trans,self.phi)) # inverse of phi_trans and phi_arr
        self.weight = np.dot(np.dot(inv(np.dot(phi_trans, self.phi)),phi_trans),height)
        
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
        var = self.getVariance(data)
        for i in data:
            for j in self.model:
                x_mean = j[0]
                y_mean = j[1]
                x_var  = var[0]
                y_var  = var[1]
                result  = -pow(i[0] - x_mean,2)/x_var - pow(i[1] - y_mean,2)/y_var
                temp.append(exp(result))
            index += 1
        self.phi = np.asarray(temp).reshape(index , len(self.model)) # reshape temp to model array (# of data) x (# of trained weight)
        phi_trans = self.phi.transpose() # transpose phi_arr
        phi_inv = inv(np.dot(phi_trans,self.phi)) # inverse of phi_trans and phi_arr
        lambda_guess = lamb
        self.weight = np.dot(np.dot(np.identity(len(self.model))+(inv(np.dot(phi_trans, self.phi))),phi_trans),height)
        
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
            temp += pow(i - j,2)
            num += 1
        err = fabs(temp) / 2 / num
        return err
    
    def errorFunction_MAP(self, dataPred, dataTrue):
        err = 0
        num = 0
        temp = 0
        for i,j in zip(dataPred,dataTrue):
            temp += pow(i - j,2)
            num += 1
        err = np.sqrt(fabs(temp) / 2 / num)
        return err
    
    def prediction(self, data):
        y_model = []
        num_model = int(np.asarray(self.phi).shape[1])
        var = self.getVariance(data)
        for da in data:
            temp = 0
            index = 0
            for i in self.model:
                x_mean = i[0]
                y_mean = i[1]
                x_var  = var[0]
                y_var  = var[1]
                try:
                    temp += exp(-(pow((da[0] - x_mean),2)/x_var + pow((da[1] - y_mean),2)/y_var)) * self.weight[index][0]
                except OverflowError:
                    ans = float('inf')
                index += 1
            result = temp / (index+1)
            y_model.append(result)
        return y_model
            
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
mu_x = [i for i in range(1, 1082, 54)]
mu_y = [i for i in range(1, 1082, 108)]
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
gcm.training(position,height)

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

# Change format of [list] into list
temp = height
y_true = []
for i in temp:
    y_true.append(i)

err = gcm.errorFunction(y_pred,y_true)
err