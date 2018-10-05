import os  
import tensorflow as tf 
import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
trial_name = 'p1_fit' # will add a unique sub-string to output of this program
trial = 'alpha_-beta_-deg'
degree = 15 # p, order of model
beta = 0.01 # regularization coefficient
alpha = 1.5 # step size coefficient
eps = 0.00001 # controls convergence criterion
n_epoch = 1500 # number of epochs (full passes through the dataset)

# begin simulation

def regress(X, theta):
	# WRITEME: write your code here to complete the routine
	return (theta[0] + tf.matmul(X, tf.transpose(theta[1])))

def gaussian_log_likelihood(mu, y, theta):
	# WRITEME: write your code here to complete the routine
	return (regress(mu, theta) - y)
	
def computeCost(X, y, theta, beta): # loss is now Bernoulli cross-entropy/log likelihood
	# WRITEME: write your code here to complete the routine
	a = tf.divide(tf.reduce_sum(tf.square(gaussian_log_likelihood(X, y, theta))),tf.cast(2 * tf.shape(X)[0], tf.float64))
	b = tf.divide(tf.multiply(tf.cast(beta, tf.float64),tf.reduce_sum(tf.square(theta[1]))), tf.cast(2 * tf.shape(X)[0], tf.float64))
	return (a+b)
	
def computeGrad(X, y, theta, beta):
	m= tf.cast(tf.shape(X), tf.float64)
	# WRITEME: write your code here to complete the routine (
	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
	mle = gaussian_log_likelihood(X, y, theta)
	dL_dfy = None # derivative w.r.t. to model output units (fy)
	dL_db = tf.divide(tf.reduce_sum(mle), m[0]) # derivative w.r.t. model weights w
	a = tf.divide(tf.matmul(mle, X, transpose_a= True), m[0])
	b = tf.divide(tf.multiply(tf.cast(beta, tf.float64),theta[1]),m[0])
	dL_dw = a + b  # derivative w.r.t model bias b
	nabla = (dL_db, dL_dw) # nabla represents the full gnp.transpose(X_feat)radient
	return nabla

path = os.getcwd() + '/data/prob2.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y'])
save_path = os.getcwd() + '/Prob2_Files'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)

#TODO convert np array to tensor objects
X_t = tf.convert_to_tensor(X, dtype= tf.float64)
y_t = tf.convert_to_tensor(y, dtype= tf.float64)

#Random Seed
seed= 20510119
np.random.seed(seed)
tf.set_random_seed(seed)


#TODO create an placeholder variable for X(input) and Y(output)
X_p= tf.placeholder(tf.float64, shape = (X.shape[0],1))
y_p= tf.placeholder(tf.float64, shape= (X.shape[0],1))

# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you could use a loop and array concatenation)




def feature_x(X, degree):
	X_feat=[]
	for i in range(1, (degree)+ 1):
		X_feat.append(np.power(X, i))
	#X_feat =np.array(X_feat)
	X_feat= np.concatenate(X_feat, axis= 1)
	X_feat_t = tf.convert_to_tensor(X_feat, dtype= tf.float64)
	X_feat_p = tf.placeholder(tf.float64, shape = (X.shape[0],degree))
	return X_feat, X_feat_t, X_feat_p


	
X_feat, X_feat_t, X_feat_p = feature_x(X, degree)
# convert to numpy arrays and initalize the parameter array theta
w = np.zeros((1, X_feat.shape[1]))
b = np.array([0])
theta = (b, w)

#create tensorflow variables w,b and theta as soon above
#Converting to tensor
w_t = tf.Variable(w, dtype = tf.float64, name= 'w')
b_t = tf.Variable(b, dtype = tf.float64, name='b')

cost =[]
L = computeCost(X_feat_t, y_t, theta, beta)
with tf.Session() as sess:
	print("-1 L = {0}".format(L.eval()))
	cost.append(L.eval())

i = 0



with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	while(i < n_epoch):
		#use feeddict to pass variables 
		dL_db, dL_dw = computeGrad(X_feat_t, y_t, theta, beta)
		p,q = sess.run([dL_db, dL_dw], feed_dict ={X_feat_p: X_feat, y_p: y})


		b = theta[0]
		w = theta[1]
		# update rules go here...
		# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
		b = b - alpha * p
		w = w - alpha * q
		theta = (b,w)

		L = computeCost(X_feat, y, theta, beta)
		cost.append(L.eval())
	
	
		# WRITEME: write code to perform a check for convergence (or simply to halt early)
	
		print(" {0} L = {1}".format(i,L.eval()))
		i += 1

		if (cost[-2] - cost[-1]) < eps:
			break
	# print parameter values found after the search
	#TODO
	#print("w = ",w)
	print('W :', w)
	#print("b = ",b)
	print('b :', b)

	saver = tf.train.Saver([w_t, b_t])
	saver.save(sess, save_path + '/Model', global_step= i)


#Save everything into saver object in tensorflow
with tf.Session() as sess:
	writer= tf.summary.FileWriter(save_path+'/Prob2_tensorboard', sess.graph)
#Visualize using tensorboard

kludge = 0.25
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_feat = np.expand_dims(X_test, axis=1) # we need this otherwise, the dimension is missing (turns shape(value,) to shape(value,value))

# apply feature map to input features x1
X_feat, X_feat_t, X_feat_p = feature_x(X_feat, degree)
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you could use a loop and array concatenation)

with tf.Session() as sess:
	plt.figure(1)
	plt.plot(X_test, sess.run(regress(X_feat, theta)), label="Model")
	plt.scatter(X[:,0], y, edgecolor='g', s=20, label="Samples")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
	plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
	plt.legend(loc="best")
	plt.savefig(save_path + '/Prob2_Test_Fit'+ 'degree'+str(degree) + '_alpha' + str(alpha) + '_beta' + str(beta) +'.jpg')
# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)


plt.figure(2)
plt.plot(cost)
plt.title("Cost vs Epoch")
plt.xlabel("Number of Epochs")
plt.ylabel("Cost")
plt.savefig(save_path + '/Prob2_ErrorvsEpoch' + 'degree' +str(degree) + '_alpha' + str(alpha) + '_beta' + str(beta) +'.jpg')

plt.show()
