import os 
import numpy as np
import tensorflow as tf
import pandas as pd  
import matplotlib.pyplot as plt  
import sys

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
Tejas Mahale
Multi Layer Perceptron the XOR Problem (1 hidden layer in this case)
'''

def softMAx(X, y, N, indices):
    #Forward
	N = tf.shape(X, out_type = tf.float64)[0]
	X_exp = tf.exp(X)
	X_esp_sum = tf.reduce_sum(X_exp, axis =1, keepdims = True)
	p = tf.divide(X_exp, X_esp_sum)
	p_ind = tf.gather_nd(p, indices)
	l1 =- tf.divide(tf.reduce_sum(tf.log(p_ind)), N)
	#Backward
	dX = p - p_hot
	dX = tf.divide(dX, N)
	return l1, p, dX


def scatter_matrix(indice_scatter, updates, shape):
	indicess = tf.constant([indice_scatter])
	scatterM = tf.scatter_nd(indicess, updates, shape)
	scatterM = tf.cast(scatterM, dtype = tf.float64)
	return scatterM


def computeCost(X,y,theta,reg, indices):
	W = theta[0]
	b = theta[1]
	W2 = theta[2]
	b2 = theta[3]
	z = tf.add(b, tf.matmul(X, W))
	h = tf.maximum(tf.cast(0, dtype= tf.float64), z)
	f = tf.add(b2, tf.matmul(h, W2))
	l1, _, _ = softMAx(f, y, N, indices)
	W_sum = tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(W2)))
	l2 = 0.5 * tf.multiply(W_sum, reg)
	return l1 + l2

#This is used in computenumgrad to check proper initialisation of parameters
def formatTheta(k, theta_list, param):
	if k==0:
		theta_l =(param, theta_list[1], theta_list[2], theta_list[3])
	elif k==1:
		theta_l = (theta_list[0], param, theta_list[2], theta_list[3])
	elif k==2:
		theta_l = (theta_list[0], theta_list[1], param, theta_list[3])
	else:
		theta_l = (theta_list[0], theta_list[1], theta_list[2], param)
	return theta_l


# check proper initialisation of parameters
def computeNumGrad(X,y,theta,reg, indices): # returns approximate nabla
	eps = 1e-5
	theta_list = list(theta)
	nabla_n = []
	for k in range(len(theta_list)):
		m = [param_dim[2*k], param_dim[2*k + 1]]
		param_grad = tf.zeros(tf.shape(theta_list[k]), dtype=tf.float64)
		for i in range(m[0]):
			for j in range(m[1]):

				param = theta_list[k] # Get parameter 
				
				#Prepare scatter matrix for Param
				indice_scatter = [i,j] if k==0 or k==2 else [i]
				scatter = scatter_matrix(indice_scatter, tf.constant([eps]),tf.shape(param))
				
				#J+
				param += scatter
				theta_plus = formatTheta(k, theta_list, param)
				J_plus = computeCost(X, y, theta_plus, reg, indices)
				
				#J-
				param -= 2 * scatter
				theta_minus = formatTheta(k, theta_list, param)
				J_minus = computeCost(X, y, theta_minus, reg, indices)
				
				
				#Compute Cost difference
				param_grad_update = tf.divide(tf.subtract(J_plus, J_minus), eps)
				param_grad_update =  tf.divide(param_grad_update, 2)
				param_grad_update = param_grad_update.eval()
				
				#Prepare param_grad scatter matrix
				param_scatter = scatter_matrix(indice_scatter, tf.constant([param_grad_update]), tf.shape(param))
				
				param_grad = param_grad + param_scatter
				
		nabla_n.append(param_grad)
	return tuple(nabla_n)


def computeGrad(X,y,theta,reg, indices): # returns nabla
	#Forward
	W = theta[0]
	b = theta[1]
	W2 = theta[2]
	b2 = theta[3]
	z = tf.add(b, tf.matmul(X, W))
	h = tf.maximum(tf.cast(0, dtype= tf.float64), z)
	f = tf.add(b2, tf.matmul(h, W2))
	#Backward
	_,_, dp = softMAx(f, y, N, indices)
	dh = tf.matmul(dp, W2, transpose_b = True)
	dh = tf.where(tf.greater(z, 0), dh, tf.zeros(tf.shape(dh), dtype=tf.float64))
	dW = tf.matmul(X, dh, transpose_a = True) + tf.multiply(W, reg)
	db = tf.reduce_sum(dh, axis =0)
	dW2 = tf.matmul(h, dp, transpose_a = True) + tf.multiply(W2, reg)
	db2 = tf.reduce_sum(dp, axis =0)
	nabla = tf.tuple([dW, db, dW2, db2])
	return nabla


def predict(X,theta, indices):
	W = theta[0]
	b = theta[1]
	W2 = theta[2]
	b2 = theta[3]
	z = tf.add(b, tf.matmul(X, W))
	h = tf.maximum(tf.cast(0, dtype= tf.float64), z)
	f_score = tf.add(b2, tf.matmul(h, W2))
	_,prob,_ = softMAx(f_score, y, N, indices)
	return (f_score,prob)
	

#20510119
tf.set_random_seed(20510119)
np.random.seed(20510119)

# Load in the data from disk
path = os.getcwd() + '/data/xor.dat'  
data = pd.read_csv(path, header=None) 

#Folder to save images
save_dir = os.getcwd() + '/Result_images' + '/prob_1c'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)
y = y.flatten()

#Create indices matrix with row number and corresponding label value
# It is used in tf.scatter_nd
indices = []
indices = [[i, y[i]] for i in range(y.shape[0])]
indices = np.array(indices)
indices = tf.constant(indices, dtype=tf.int64)

#Convert X and Y to tensor
X_tf = tf.constant(X)
Y_tf = tf.constant(y)

# initialize parameters randomly
D = X.shape[1] # Number of Feature
K = np.amax(y) + 1 # K = Classes

#Onehot of y
#This will be used in backpass
p_hot = tf.contrib.layers.one_hot_encoding(indices[:, 1], K)
p_hot = tf.cast(p_hot , dtype = tf.float64)

# initialize parameters in such a way to play nicely with the gradient-check! 
h = 6 #100 # size of hidden layer

initializer = tf.random_normal_initializer(mean=0.0, stddev=0.015, seed=20510119, dtype=tf.float64)
W = tf.Variable(initializer([D, h]), dtype = tf.float64, name = "W")
b = tf.Variable(tf.ones([h], dtype=tf.float64), dtype = tf.float64, name = "b")
W2 = tf.Variable(initializer([h, K]), dtype = tf.float64, name = "W2")
b2 = tf.Variable(tf.ones([K], dtype=tf.float64), dtype = tf.float64, name = "b2")
theta = (W,b,W2,b2) 


#Number of examples
N = tf.shape(X, out_type = tf.float64)[0]

#Save dimensions of parameters
param_dim = [D, h, h, 1, h, K, K, 1]


# some hyperparameters
reg = 1e-3 # regularization strength
reg = tf.constant(reg, dtype = tf.float64,  name = "reg") # regularization strength


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	nabla_n = computeNumGrad(X_tf,Y_tf,theta,reg, indices)
	nabla = computeGrad(X_tf,Y_tf,theta,reg, indices)
	nabla_n = list(nabla_n)
	nabla = list(nabla)

	for jj in range(0,len(nabla)):
		is_incorrect = 0 # set to false
		grad = nabla[jj]
		grad_n = nabla_n[jj]
		grad_sub = tf.subtract(grad_n,grad)
		grad_add = tf.add(grad_n,grad)
		err = tf.div(tf.norm(grad_sub, ord=2) , (tf.norm(grad_add, ord=2)))
		if(err.eval() > 1e-7):
			print("Param {0} is WRONG, error = {1}".format(jj, sess.run(err)))
		else:
			print("Param {0} is CORRECT, error = {1}".format(jj, sess.run(err)))


# re-init parameters
h = 6 #100 # size of hidden layer

initializer = tf.random_normal_initializer(mean=0.0, stddev=0.015, seed=20510119, dtype=tf.float64)
#W = tf.Variable(0.001  * initializer([D, h]), dtype = tf.float64, name = "W")
W = 0.01 * np.random.randn(D,h)
W = tf.Variable(W, dtype = tf.float64, name = "w_t")
b = tf.Variable(tf.zeros([h], dtype=tf.float64), dtype = tf.float64, name = "b")
#W2 = tf.Variable(0.001  * initializer([h, K]) , dtype = tf.float64, name = "W2")
W2 = 0.01 * np.random.randn(h,K)
W2 = tf.Variable(W2, dtype = tf.float64, name = "w_t1")
b2 = tf.Variable(tf.zeros([K], dtype=tf.float64), dtype = tf.float64, name = "b2")
theta = (W,b,W2,b2) 


# some hyperparameters
n_e = 1000
check = 10 # every so many pass/epochs, print loss/error to terminal
step_size = tf.constant(0.1, dtype = tf.float64,  name = "step_size") #Learning Rate
reg = tf.constant(0.01, dtype = tf.float64,  name = "reg") # regularization strength


# Placeholders for input
X_p= tf.placeholder(tf.float64, shape = (X.shape[0],X.shape[1]))
y_p= tf.placeholder(tf.float64, shape= (y.shape[0]))


#Computation for cost and gradient
ComputeCost = computeCost(X_p, y_p,theta, reg, indices)
ComputeGrad = computeGrad(X_p , y_p,theta, reg, indices)


# gradient descent loop
cost = []
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(n_e):
		loss = sess.run([ComputeCost],feed_dict = {X_p:X , y_p:y })
		cost.append(loss)
		if i % check == 0:
			print ("iteration %d: loss %f" % (i, loss[0]))
		# perform a parameter update
		grad_new1 = sess.run([ComputeGrad] , feed_dict = {X_p:X , y_p:y})
		grad_new2 = [item for sublist in grad_new1 for item in sublist]
		b_t1 = tf.subtract(b ,tf.multiply(step_size,grad_new2[1])) # b = b - LR * db
		W_t1 = tf.subtract(W , tf.multiply(step_size,grad_new2[0])) #W = W - LR * dW
		b2_t2 = tf.subtract(b2 ,tf.multiply(step_size,grad_new2[3])) # b2 = b2 - LR * db2
		W2_t2 = tf.subtract(W2 , tf.multiply(step_size,grad_new2[2])) #W2 = W2 - LR * dW2
		
		sess.run(tf.assign(b,b_t1))
		sess.run(tf.assign(W,W_t1))
		sess.run(tf.assign(b2,b2_t2))
		sess.run(tf.assign(W2,W2_t2))

	#Calculate Accuracy
	scores, probs = predict(X, theta, indices)
	predicted_class = sess.run(tf.argmax(scores, axis=1))
	print ('training accuracy: %.2f' % (sess.run(tf.reduce_mean(tf.to_float(predicted_class == y)))))

	#Plot Cost vs Epoch Curve
	reg1 = reg.eval()
	step_size1 = step_size.eval()
	fig = plt.figure(1)
	plt.plot(cost, label= 'Alpha:' + str(step_size1)+ ' & reg:' + str(reg1))
	plt.title("Cost vs Epoch")
	plt.xlabel("Number of Epochs")
	plt.ylabel("Cost")
	plt.legend(loc='upper right', fontsize='x-large')
	plt.savefig(save_dir + '/Loss_plot_stepSize'+str(step_size1) + '_reg' + str(reg1)+'.png')
plt.show()
