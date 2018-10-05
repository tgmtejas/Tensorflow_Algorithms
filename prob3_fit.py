import os  
import tensorflow as tf 
import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
#Change variables to tf.constant or tf.Variable whenever needed
trial_name = 'p6_reg0' # will add a unique sub-string to output of this program
degree = 6 # p, degree of model (LEAVE THIS FIXED TO p = 6 FOR THIS PROBLEM)
beta = 1 # regularization coefficient
alpha = 1.5 # step size coefficient
n_epoch = 1000 # number of epochs (full passes through the dataset)
eps = 0.00001 # controls convergence criterion

# begin simulation

def sigmoid(z):
	# WRITEME: write your code here to complete the routine
	return tf.divide(tf.cast(1, dtype = tf.float64), tf.add(tf.cast(1, dtype = tf.float64), tf.exp(-z)))
	

def predict(X, theta):  
	# WRITEME: write your code here to complete the routine
	p = (theta[0] + tf.matmul(X, tf.transpose(theta[1])))
	q = tf.round(sigmoid(p))
	return q
	
def regress(X, theta):
	# WRITEME: write your code here to complete the routine
	return sigmoid(theta[0] + tf.matmul(X, tf.transpose(theta[1])))

def bernoulli_log_likelihood(p, y, theta):
	# WRITEME: write your code here to complete the routine
	return -1
	
def computeCost(X, y, theta, beta): # loss is now Bernoulli cross-entropy/log likelihood
	# WRITEME: write your code here to complete the routine
	m= tf.cast(tf.shape(X), tf.float64)
	beta = tf.cast(beta, tf.float64)
	#tf.cast(1, tf.float64)
	a = tf.divide(tf.reduce_sum(-tf.multiply(y, tf.log(regress(X,theta))) - tf.multiply((tf.cast(1, tf.float64)-y), tf.log(tf.cast(1, tf.float64) - regress(X,theta)))),m[0])

	b=  tf.reduce_sum(tf.square(theta[1]))
	b= tf.cast(b, tf.float64)
	b = tf.divide(tf.multiply(beta, b), 2 * m[0])
	return a + b
	
def computeGrad(X, y, theta, beta): 
	# WRITEME: write your code here to complete the routine (
	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
	m= tf.cast(tf.shape(X), tf.float64)
	mle = regress(X, theta) - y
	dL_dfy = None # derivative w.r.t. to model output units (fy)
	dL_db = tf.divide(tf.reduce_sum(mle), m[0]) # derivative w.r.t. model weights w
	a = tf.divide(tf.matmul(mle, X, transpose_a= True), m[0])
	b = tf.divide(tf.multiply(tf.cast(beta, tf.float64),theta[1]),m[0])
	dL_dw = a + b # derivative w.r.t model bias b
	nabla = (dL_db, dL_dw) # nabla represents the full gradient
	return nabla
	
path = os.getcwd() + '/data/prob3.dat'  
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
save_path = os.getcwd() + '/Prob3_Files'
if not os.path.exists(save_path):
    os.makedirs(save_path)



positive = data2[data2['Accepted'].isin([1])]  
negative = data2[data2['Accepted'].isin([0])]
#TODO
#Convert positive and negative samples into tf.Variable 
positive_t = tf.Variable(positive)
negative_t = tf.Variable(negative)


x1 = data2['Test 1']  
x2 = data2['Test 2']
#Convert x1 and x2 to tensorflow variables
x1_t = tf.Variable(x1)
x2_t = tf.Variable(x2)

# apply feature map to input features x1 and x2
cnt = 0
for i in range(1, degree+1):  
	for j in range(0, i+1):
		data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
		cnt += 1

data2.drop('Test 1', axis=1, inplace=True)  
data2.drop('Test 2', axis=1, inplace=True)

# set X and y
cols = data2.shape[1]  
X2 = data2.iloc[:,1:cols]  
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)  
y2 = np.array(y2.values)  
w = np.zeros((1,X2.shape[1]))
b = np.array([0])
theta = (b, w)

#Random Seed
seed= 20510119
np.random.seed(seed)
tf.set_random_seed(seed)

#TODO
#Convert all numpy variables into tensorflow variables
X_t = tf.convert_to_tensor(X2, dtype= tf.float64)
y_t = tf.convert_to_tensor(y2, dtype= tf.float64)
w_t = tf.Variable(w, dtype = tf.float64, name= 'w')
b_t = tf.Variable(b, dtype = tf.float64, name='b')

X_p= tf.placeholder(tf.float64, shape = (X2.shape[0],X2.shape[1]))
y_p= tf.placeholder(tf.float64, shape= (X2.shape[0],1))



cost =[]
L = computeCost(X_t, y_t, theta, beta)
with tf.Session() as sess:
	print("-1 L = {0}".format(L.eval()))
	cost.append(L.eval())

i = 0
halt= 0
#Initialize graph and all variables
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	while(i < n_epoch and halt == 0):
		dL_db, dL_dw = computeGrad(X_t, y_t, theta, beta)
		#use feeddict to pass variables to pass holder
		p,q = sess.run([dL_db, dL_dw], feed_dict ={X_p: X2, y_p: y2})
		b = theta[0]
		w = theta[1]
		# update rules go here...
		# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
		b = b - alpha * p
		w = w - alpha * q
		theta = (b,w)

		L = computeCost(X_t, y_t, theta, beta)
		cost.append(L.eval())
	
		# WRITEME: write code to perform a check for convergence (or simply to halt early)
		if (cost[-2] - cost[-1]) < eps:
			break
	
		print(" {0} L = {1}".format(i,L.eval()))
		i += 1
		
		
	# print parameter values found after the search
	#print("w = ",w)
	print('W :', w)
	#print("b = ",b)
	print('b :', b)
	#Save everything into saver object in tensorflow
	saver = tf.train.Saver([w_t, b_t])
	saver.save(sess, save_path + '/Model', global_step= i)
	#Visualize using tensorboard
	writer= tf.summary.FileWriter(save_path+'/Prob3_tensorboard', sess.graph)

err = 0.0
with tf.Session() as sess:
	predictions = predict(X2, theta)
	m = tf.cast(tf.shape(X2), tf.float64)
	acc_t1 = (tf.matmul(tf.transpose(y_t), predictions))/m[0]
	acc1 = acc_t1.eval()
	acc_t2 = (tf.matmul(tf.transpose(1- y_t), (1- predictions)))/m[0]
	acc2 = acc_t2.eval()
	err = 1- (acc1 +acc2)

# compute error (100 - accuracy)

# WRITEME: write your code here calculate your actual classification error (using the "predictions" variable)
	print ('Error = {0}%'.format(err * 100.))


"""
# make contour plot
xx, yy = np.mgrid[-1.2:1.2:.01, -1.2:1.2:.01]
xx1 = xx.ravel()
yy1 = yy.ravel()
grid = np.c_[xx1, yy1]
grid_nl = []
# re-apply feature map to inputs x1 & x2
#Convert the below feature map into tensorflow environment

for i in range(1, degree+1):  
	for j in range(0, i+1):
		feat = np.power(xx1, i-j) * np.power(yy1, j)
		if (len(grid_nl) > 0):
			grid_nl = np.c_[grid_nl, feat]
		else:
			grid_nl = feat
probs = regress(grid_nl, theta2).reshape(xx.shape)

f, ax = plt.subplots(figsize=(8, 6))
ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

ax.scatter(x1, x2, c=y2, s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
       xlabel="$X_1$", ylabel="$X_2$")
# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)

plt.show()
"""

# make contour plot
xx, yy = np.mgrid[-1.2:1.2:.01, -1.2:1.2:.01]
xx1 = xx.ravel()
yy1 = yy.ravel()
grid = np.c_[xx1, yy1]
grid_nl = []
# re-apply feature map to inputs x1 & x2
#Convert the below feature map into tensorflow environment

for i in range(1, degree+1):  
    for j in range(0, i+1):
        feat = np.power(xx1, i-j) * np.power(yy1, j)
        if(len(grid_nl) > 0):
            grid_nl = np.c_[grid_nl, feat]
        else:
            grid_nl = feat

grid_nl_t = tf.convert_to_tensor(grid_nl, dtype = tf.float64)
prob_s= 0
with tf.Session() as sess:
    probs = regress(grid_nl_t, theta)
    probs = tf.reshape(probs, shape = xx.shape)  
    prob_s = probs.eval()


f, ax = plt.subplots(figsize=(8, 6))
ax.contour(xx, yy, prob_s, levels=[.5], cmap="Greys", vmin=0, vmax=.6)
x1 = np.expand_dims(x1, axis=1)
x2 = np.expand_dims(x2, axis=1)
ax.scatter(x1, x2, c=y2, s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
       xlabel="$X_1$", ylabel="$X_2$")
# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)
plt.savefig(save_path + '/Prob3_Test_Fit'+ 'degree'+str(degree) + '_alpha' + str(alpha) + '_beta' + str(beta) +'.jpg')
plt.show()



plt.plot(cost)
plt.title("Cost vs Epoch")
plt.xlabel("Number of Epochs")
plt.ylabel("Cost")
plt.savefig(save_path + '/Prob3_ErrorvsEpoch' + 'degree' +str(degree) + '_alpha' + str(alpha) + '_beta' + str(beta) +'.jpg')

plt.show()