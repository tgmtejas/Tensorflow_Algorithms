import os  
import tensorflow as tf
import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np


# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
alpha = 0.024 # step size coefficient
eps = 0.00001 # controls convergence criterion
n_epoch = 10000 # number of epochs (full passes through the dataset)

# begin simulation

def regress(X, theta):
	# WRITEME: write your code here to complete the routine
	return (theta[0] + tf.multiply(X, theta[1]))

def gaussian_log_likelihood(mu, y, theta):
	# WRITEME: write your code here to complete the sub-routine
	return (regress(mu, theta) - y)
	
def computeCost(X, y, theta): # loss is now Bernoulli cross-entropy/log likelihood
	# WRITEME: write your code here to complete the routine
    #cost= tf.contrib.framework.local_variable(0, name='cost')
    return tf.divide(tf.reduce_sum(tf.square(gaussian_log_likelihood(X, y, theta))),tf.cast(2 * tf.shape(X)[0], tf.float64))
    
	
def computeGrad(X, y, theta):
	m= tf.cast(tf.shape(X), tf.float64)

	# WRITEME: write your code here to complete the routine
	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
	dL_dfy = None # derivative w.r.t. to model output units (fy)
	dL_db = tf.divide(tf.reduce_sum(gaussian_log_likelihood(X, y, theta)), m[0])# derivative w.r.t. model weights w
	dL_dw = tf.divide(tf.matmul(gaussian_log_likelihood(X, y, theta), X, transpose_a=True), m[0]) # derivative w.r.t model bias b
	nabla = (dL_db, dL_dw) # nabla represents the full gradient
	return nabla

path = os.getcwd() + '/data/prob1.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y'])
save_path = os.getcwd() + '/Prob1_Files'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# display some information about the dataset itself here
print(data.shape)
# WRITEME: write your code here to print out information/statistics about the data-set "data" using Pandas (consult the Pandas documentation to learn how)
print(data.describe())
# WRITEME: write your code here to create a simple scatterplot of the dataset itself and print/save to disk the result
data.plot(kind = 'scatter', x= 'X', y = 'Y')
plt.savefig(save_path +'/Prob1_scatter_Plot.jpg')

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)

#Random Seed
seed= 20510119
np.random.seed(seed)
tf.set_random_seed(seed)


#TODO convert np array to tensor objects
X_t = tf.convert_to_tensor(X, dtype= tf.float64)
y_t = tf.convert_to_tensor(y, dtype= tf.float64)

#TODO create an placeholder variable for X(input) and Y(output)
X_p= tf.placeholder(tf.float64, shape = (X.shape[0],1))
y_p= tf.placeholder(tf.float64, shape= (X.shape[0],1))

# convert to numpy arrays and initalize the parameter array theta 
w = np.zeros((1,X.shape[1]))
b = np.array([0])
theta = (b, w)

#Converting to tensor
w_t = tf.Variable(w, dtype = tf.float64, name= 'w')
b_t = tf.Variable(b, dtype = tf.float64, name='b')

# theta_t = tf.Variable(theta, dtype=tf.float64, name = 'theta')

with tf.Session() as sess:
	L = computeCost(X_t, y_t, theta)
	print("-1 L = {0}".format(L.eval()))
	L_best = L.eval()
i = 0
cost = [] # you can use this list variable to help you create the loss versus epoch plot at the end (if you want)
cost.append(L_best)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	while(i < n_epoch):
		dL_db, dL_dw = computeGrad(X_t, y_t, theta)
		#use feeddict to pass variables
		p,q = sess.run([dL_db, dL_dw], feed_dict ={X_p: X, y_p: y})
		b = theta[0]
		w = theta[1]
		# update rules go here...
		# WRITEME: write your code here toyyy perform a step of gradient descent & record anything else desired for later
		b = b - alpha * p
		w = w - alpha * q
		# (note: don't forget to override the theta variable...)
		theta = (b,w)
		L = computeCost(X, y, theta) # track our loss after performing a single step
		if (cost[-1]- L.eval()) < eps:
			break
		i += 1

		cost.append(L.eval())
		print(" {0} L = {1}".format(i,L.eval()))

		
        #TODO
	# print parameter values found after the search
	print('W :', w)
	print('b :', b)
	saver = tf.train.Saver([w_t, b_t])
	saver.save(sess, save_path + '/Model', global_step= i)
	#sess, os.getcwd() + '/Prob1_Files/Model', global_step= i

	#print W
#print b
#Save everything into saver object in tensorflow


#Visualize using tensorboard
with tf.Session() as sess:
	writer= tf.summary.FileWriter(save_path+'/Prob1_tensorboard', sess.graph)
	#tf.saved_model.simple_save(sess, os.getcwd(), inputs ={'x': X, 'y': y}, outputs= {'w':w, 'b':b})



kludge = 0.25 # helps with printing the plots (you can tweak this value if you like)
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_test = np.expand_dims(X_test, axis=1)

with tf.Session() as sess:
	plt.figure(1)
	plt.plot(X_test, sess.run(regress(X_test, theta)), label="Model")
	plt.scatter(X[:,0], y, edgecolor='g', s=20, label="Samples")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
	plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
	plt.legend(loc="best")
	plt.savefig(save_path + '/Prob1_Test_Fit.jpg')
# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)

# visualize the loss as a function of passes through the dataset
# WRITEME: write your code here create and save a plot of loss versus epoch
plt.figure(2)
plt.plot(cost)
plt.title("Cost vs Epoch")
plt.xlabel("Number of Epochs")
plt.ylabel("Cost")
plt.savefig(save_path + '/Prob1_ErrorvsEpoch.jpg')


plt.show() # convenience command to force plots to pop up on desktop
