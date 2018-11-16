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
Softmax Regression \& the XOR Problem
'''

def softMAx(X, y, N, indices):
    #Forward pass
	N = tf.shape(X, out_type = tf.float64)[0]
	X_exp = tf.exp(X)
	X_esp_sum = tf.reduce_sum(X_exp, axis =1, keepdims = True)
	p = tf.divide(X_exp, X_esp_sum)
	p_ind = tf.gather_nd(p, indices)
	l1 =- tf.divide(tf.reduce_sum(tf.log(p_ind)), N)
	dX = p - p_hot
	dX = tf.divide(dX, N)
	return l1, p, dX

# scatter_matrix used in ComputeNUmGrad function to check initilizations
def scatter_matrix(indice_scatter, updates, shape):
	indicess = tf.constant([indice_scatter])
	scatterM = tf.scatter_nd(indicess, updates, shape)
	scatterM = tf.cast(scatterM, dtype = tf.float64)
	return scatterM

#This function checks initilization of parameters W and b
def computeNumGrad(X,y,theta,reg, indices): 
	eps = 1e-5
	theta_list = list(theta)
	nabla_n = []
	for k in range(len(theta_list)):
		m = [param_dim[2*k], param_dim[2*k + 1]]
		param_grad = tf.zeros(tf.shape(theta_list[k]), dtype=tf.float64)
		for i in range(m[0]):
			for j in range(m[1]):
				param = theta_list[k]

				#Prepare scatter matrix for Param
				indice_scatter = [i,j] if k==0 else [i]
				scatter = scatter_matrix(indice_scatter, tf.constant([eps]),tf.shape(param))

				#J+
				param += scatter
				if k==0:
					theta_plus =(param, theta_list[k ^ 1])
				else:
					theta_plus = (theta_list[k ^ 1], param)
					
				J_plus = computeCost(X, y, theta_plus, reg, indices)
				
				
				#J-
				param -= 2 * scatter

				if k==0:
					theta_minus =(param, theta_list[k ^ 1])
				else:
					theta_minus = (theta_list[k ^ 1], param)

				J_minus = computeCost(X, y, theta_minus, reg, indices)

				
				#Compute Cost difference
				param_grad_update = tf.divide(tf.subtract(J_plus, J_minus), (2 * eps))
				param_grad_update = param_grad_update.eval()
				#print(param_grad_update)

				#Prepare param_grad scatter matrix
				param_scatter = scatter_matrix(indice_scatter, tf.constant([param_grad_update]), tf.shape(param))
				
				param_grad = param_grad + param_scatter
				
		nabla_n.append(param_grad)
	
	return tuple(nabla_n)


#Compute gradient in backward pass
def computeGrad(X,y,theta,reg, indices): # returns nabla
	f = tf.add(theta[1], tf.matmul(X, theta[0]))
	_,_, dp = softMAx(f, y, N, indices)
	dW = tf.matmul(X, dp, transpose_a = True) + tf.multiply(theta[0], reg)
	db = tf.reduce_sum(dp, axis =0)
	nabla = tf.tuple([dW, db])
	return nabla



def computeCost(X,y,theta,reg, indices):
	f = tf.add(theta[1], tf.matmul(X, theta[0]))
	#Call Softmax
	l1, _, _ = softMAx(f, y, N, indices)
	l2 = 0.5 * tf.multiply(tf.reduce_sum(tf.square(theta[0])), reg)
	return tf.add(l1, l2)


def predict(X,theta, indices):
	f_score = tf.add(theta[1], tf.matmul(X, theta[0]))
	_,prob,_ = softMAx(f_score, y, N, indices)
	return (f_score,prob)


#Seed = 20510119
tf.set_random_seed(20510119)
np.random.seed(20510119) #Provide your unique Random seed

# Load in the data from disk
path = os.getcwd() + '/data/xor.dat'  
data = pd.read_csv(path, header=None) 


#Path to Save images
save_dir = os.getcwd() + '/Result_images' + '/prob_2a'
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

#Tensorflow labels and feature matrix
X_tf = tf.constant(X)
Y_tf = tf.constant(y)

#Placeholdeers
X_p= tf.placeholder(tf.float64, shape = (X.shape[0],X.shape[1]))
y_p= tf.placeholder(tf.float64, shape= (y.shape[0]))


# initialize parameters randomly
D = X.shape[1] #Number of features
K = np.amax(y) + 1 # k is number of classes

#Initialise parameters randomly or xavier
#initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=20510119, dtype=tf.float64)
#initializer = tf.contrib.layers.xavier_initializer(seed = 20510119, dtype=tf.float64)
#W_t = tf.Variable(0.01 * initializer([D, K]), dtype = tf.float64, name = "w_t")
W = 0.01 * np.random.randn(D,K)
W_t = tf.Variable(W, dtype = tf.float64, name = "w_t")
#b_t = tf.Variable(tf.random_normal([K], dtype=tf.float64), dtype = tf.float64, name = "b_t")
#b_t = np.zeros((K,1))
b_t = tf.zeros([K], dtype=tf.float64)
theta = (W_t, b_t)

#Onehot of y
#This will be used in backpass
p_hot = tf.contrib.layers.one_hot_encoding(indices[:, 1], K)
p_hot = tf.cast(p_hot , dtype = tf.float64)


#Number of examples
N = tf.shape(X, out_type = tf.float64)[0]

#Save dimensions of parameters
param_dim = [D, K, K, 1]

# some hyperparameters
reg = 1e-3 # regularization strength
reg = tf.constant(reg, dtype = tf.float64,  name = "reg") # regularization strength

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	nabla_n = computeNumGrad(X_tf, Y_tf,theta, reg, indices)
	#nabla_n = sess.run([],feed_dict = {X_p:X , y_p:y })
	nabla = computeGrad(X_tf, Y_tf,theta, reg, indices)
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


# Re-initialize parameters for generic training
#initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=20510119, dtype=tf.float64) #You can use Xavier or Ortho for weight init
#If using other init compare that with Guassian init and report your findings
#initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=20510119, dtype=tf.float64)
"""initializer = tf.contrib.layers.xavier_initializer(seed = 20510119, dtype=tf.float64)
W_t = tf.Variable(0.01 * initializer([D, K]), dtype = tf.float64, name = "w_t")"""
W = 0.01 * np.random.randn(D,K)
W_t = tf.Variable(W, dtype = tf.float64, name = "w_t")

b_t = tf.Variable(tf.zeros([K], dtype=tf.float64), dtype = tf.float64, name = "b_t")
theta = (W_t, b_t)

#play with hyperparameters for better performance 
n_e = 100 #number of epochs
check = 10 # every so many pass/epochs, print loss/error to terminal
step_size = tf.constant(1, dtype = tf.float64,  name = "step_size") #Learning rate Alpha
reg =tf.constant(0.1, dtype = tf.float64,  name = "reg") # regularization strength


X_p= tf.placeholder(tf.float64, shape = (X.shape[0],X.shape[1]))
y_p= tf.placeholder(tf.float64, shape= (y.shape[0]))

#Call functions outside of session
ComputeCost = computeCost(X_p, y_p,theta, reg, indices)
ComputeGrad = computeGrad(X_p , y_p,theta, reg, indices)

# gradient descent loop
num_examples = X.shape[0]
cost = [] 
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(n_e):
		loss = sess.run([ComputeCost],feed_dict = {X_p:X , y_p:y }) #Fetch cost value from graph
		cost.append(loss)
		if i % check == 0:
			print ("iteration %d: loss %f" % (i, loss[0]))
		# perform a parameter update
		grad_new1 = sess.run([ComputeGrad] , feed_dict = {X_p:X , y_p:y}) # Fetch gradients from graph
		grad_new2 = [item for sublist in grad_new1 for item in sublist]
		b_t1 = tf.subtract(b_t ,tf.multiply(step_size,grad_new2[1])) # b = b - LR * db
		W_t1 = tf.subtract(W_t , tf.multiply(step_size,grad_new2[0])) # W = W - LR * dw
		b_t2 = tf.convert_to_tensor(b_t1)
		W_t2 = tf.convert_to_tensor(W_t1)
		sess.run(tf.assign(b_t,b_t2)) #Insert values of W, b into graph 
		sess.run(tf.assign(W_t,W_t2))
	
	# evaluate training set accuracy
	scores, probs = predict(X_tf,theta, indices)
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


