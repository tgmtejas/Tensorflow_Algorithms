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
Softmax Regression \& the Spiral Problem
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
	
def computeGrad(X,y,theta,reg, indices): # returns nabla
	f = tf.add(theta[1], tf.matmul(X, theta[0]))
	_,_, dp = softMAx(f, y, N, indices)
	dW = tf.matmul(X, dp, transpose_a = True) + tf.multiply(theta[0], reg)
	db = tf.reduce_sum(dp)
	nabla = tf.tuple([dW, db])
	return nabla

def computeCost(X,y,theta,reg, indices):
	f = tf.add(theta[1], tf.matmul(X, theta[0]))
	# Call for Softmax
	l1, _, _ = softMAx(f, y, N, indices)
	l2 = 0.5 * tf.multiply(tf.reduce_sum(tf.square(theta[0])), reg)
	return l1 + l2

def predict(X,theta,indices):
	f_score = tf.add(theta[1], tf.matmul(X, theta[0]))
	_,prob,_ = softMAx(f_score, y, N, indices)
	return (f_score,prob)


#20510119
np.random.seed(20510119) #Provide your unique Random seed
tf.set_random_seed(20510119)

# Load in the data from disk
path = os.getcwd() + '/data/spiral_train.dat'  
data = pd.read_csv(path, header=None) 

#Create folder to save images
save_dir = os.getcwd() + '/Result_images' + '/prob_1b'
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


X_tf = tf.constant(X)
Y_tf = tf.constant(y)

#Placeholders for X and y
X_p= tf.placeholder(tf.float64, shape = (X.shape[0],X.shape[1]))
y_p= tf.placeholder(tf.float64, shape= (y.shape[0]))


D = X.shape[1] # Number of Features
K = np.amax(y) + 1 # K= Number of classes


# initialize parameters randomly
initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=20510119, dtype=tf.float64)
#initializer = tf.contrib.layers.xavier_initializer(seed = 20510119, dtype=tf.float64)
W_t = tf.Variable(0.01 * initializer([D, K]), dtype = tf.float64, name = "w_t")
#b_t = tf.Variable(tf.random_normal([K], dtype=tf.float64), dtype = tf.float64, name = "b_t")
b_t = tf.Variable(tf.zeros([K], dtype=tf.float64), dtype = tf.float64, name = "b_t")
theta = (W_t, b_t)

#Onehot of y used in backpass
p_hot = tf.contrib.layers.one_hot_encoding(indices[:, 1], 3)
p_hot = tf.cast(p_hot , dtype = tf.float64)

#Number of examples
N = tf.shape(X, out_type = tf.float64)[0]

# some hyperparameters
n_e = 150
check = 10 # every so many pass/epochs, print loss/error to terminal
step_size = tf.constant(1.0, dtype = tf.float64,  name = "step_size") # Learning Rate
reg = tf.constant(0.01, dtype = tf.float64,  name = "reg") # regularization strength


ComputeCost = computeCost(X_p, y_p,theta, reg, indices)
ComputeGrad = computeGrad(X_p , y_p,theta, reg, indices)

# gradient descent loop
cost = []
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(n_e):
		loss = sess.run([ComputeCost],feed_dict = {X_p:X , y_p:y }) # Fetch cost from graph
		cost.append(loss)
		if i % check == 0:
			print ("iteration %d: loss %f" % (i, loss[0]))
		grad_new1 = sess.run([ComputeGrad] , feed_dict = {X_p:X , y_p:y}) #Fetch gradients
		grad_new2 = [item for sublist in grad_new1 for item in sublist]
		b_t1 = tf.subtract(b_t ,tf.multiply(step_size,grad_new2[1])) # b = b - LR * db
		W_t1 = tf.subtract(W_t , tf.multiply(step_size,grad_new2[0]))# W = W - LR * dW
		b_t2 = tf.convert_to_tensor(b_t1)
		W_t2 = tf.convert_to_tensor(W_t1)
		sess.run(tf.assign(b_t,b_t2))
		sess.run(tf.assign(W_t,W_t2))

	#Compute Accuracy
	scores, probs = predict(X, theta, indices)
	predicted_class = sess.run(tf.argmax(scores, axis=1))
	print ('training accuracy: %.2f' % (sess.run(tf.reduce_mean(tf.to_float(predicted_class == y)))))

# plot the resulting classifier
	h = 0.02
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
	x1  = np.expand_dims(xx.ravel(), axis = 1)
	y1  = np.expand_dims(yy.ravel(), axis = 1)
	Z = sess.run(tf.matmul(tf.concat([x1 , y1], 1), W_t) + b_t)
	Z = np.argmax(Z, axis=1)
	Z = Z.reshape(xx.shape)
	fig = plt.figure(1)
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
	plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	reg1 = reg.eval()
	step_size1 = step_size.eval()
	plt.title("Classifier")
	plt.savefig(save_dir + '/Classifier_plot_stepSize'+str(step_size1) + '_reg' + str(reg1)+'.png')

	#Plot Cost vs Epoch Curve
	fig = plt.figure(2)
	plt.plot(cost, label= 'Alpha:' + str(step_size1)+ ' & reg:' + str(reg1))
	plt.title("Cost vs Epoch")
	plt.xlabel("Number of Epochs")
	plt.ylabel("Cost")
	plt.legend(loc='upper right', fontsize='x-large')
	plt.savefig(save_dir + '/Loss_plot_stepSize'+str(step_size1) + '_reg' + str(reg1)+'.png')
plt.show()