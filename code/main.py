import pickle
import pprint
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct

data_dir = '../data/'
results_dir = '../results/'
names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
pca_dim = 20

def get_data():
	#unpickle items
	data_batch_1 = unpickle(data_dir + 'data_batch_1')
	data_batch_2 = unpickle(data_dir + 'data_batch_2')
	data_batch_3 = unpickle(data_dir + 'data_batch_3')
	data_batch_4 = unpickle(data_dir + 'data_batch_4')
	data_batch_5 = unpickle(data_dir + 'data_batch_5')
	test_batch = unpickle(data_dir + 'test_batch')
	
	#data items
	data1 = data_batch_1[b'data']
	data2 = data_batch_2[b'data']
	data3 = data_batch_3[b'data']
	data4 = data_batch_4[b'data']
	data5 = data_batch_5[b'data']
	test = test_batch[b'data']

	#label items
	data_label_1 = data_batch_1[b'labels']
	data_label_2 = data_batch_2[b'labels']
	data_label_3 = data_batch_3[b'labels']
	data_label_4 = data_batch_4[b'labels']
	data_label_5 = data_batch_5[b'labels']
	test_label = test_batch[b'labels']

	#concatenated items
	data = np.concatenate([data1,data2,data3,data4,data5,test],axis=0)

	#IDing acc to data frames
	labels = np.concatenate([data_label_1, data_label_2, data_label_3, data_label_4, data_label_5,test_label])
	df_data = pd.DataFrame(data)
	df_labels = pd.Series(labels, name="labels")
	df_labels = pd.DataFrame(df_labels)
	df = pd.concat([df_data,df_labels],axis=1)
	return df



def print_mean(mean):
	for i, arr in enumerate(mean):
		im_arr = np.zeros((32,32,3), 'uint8')
		im = np.array(arr)
		im_arr[..., 0] = im[:1024].reshape((32,32))
		im_arr[..., 1] = im[1024:2048].reshape((32,32))
		im_arr[..., 2] = im[2048:].reshape((32,32))
		Image.fromarray(im_arr, 'RGB').save(results_dir + names[i] + '_mean.png')
	return


#PART1
def covmat_and_error_calc(data, mean):
	#Calculate covmat for each category
	covmat = []
	for i in range(10):
	    covmat.append(np.cov(data[i],rowvar=False))

	eig = []
	for i in range(10):
	    eig.append(np.linalg.eig(covmat[i]))

	#error calculation
	error = []
	for i in range(10):
	    sum_20 = 0
	    for j in range(pca_dim):
	        sum_20 += eig[i][0][j]
	    error.append(eig[i][0].sum() - sum_20)

	#plotting and displaying the error
	df_error = pd.DataFrame(error)
	bar_graph = df_error.plot(kind='bar',title='PCA Error Per Category', figsize=(10,8),fontsize=12, legend=False)
	bar_graph.set_ylabel("Error",fontsize=12)
	bar_graph.set_xlabel("Category",fontsize=12)
	plt.show()


# USED BY PART 2 AND PART 3
def pca_common_func(sol, mean):
	N = len(mean) #ADDED AFTER TEST
	A = np.eye(N) - np.outer(np.ones(N),np.ones(N))/N
	W = -0.5*np.matmul(np.matmul(A,sol),np.transpose(A))

	#calc eigen vals, vecs of W
	eigval, eigvec = np.linalg.eig(W)
	diag = np.matmul(np.matmul(eigvec.transpose(),W), eigvec)
	diag2d = diag[0:2,0:2]
	diag2d[0][1] = diag2d[1][0] = 0
	diag_2d_sqrt = np.sqrt(diag2d)
	eigvec_2d = W[:,0:2]
	v = np.matmul(diag_2d_sqrt,eigvec_2d.transpose())

	for i in range(len(mean)):
		x = v[0][i]
		y = v[1][i]
		plt.plot(x, y, 'bo')
		plt.text(x * (1 + 0.05), y * (1 - 0.1), names[i], fontsize=12)
	plt.show()
	print('4 SUCESS')
	return


#PART2
def pca_calc(mean):
	sol = list()
	N = len(mean)
	for i in range(N):
		D = list()
		for j in range(N):
			diff = mean[i]-mean[j]
			D_2 = np.dot(diff, diff)
			D.append(D_2)
		sol.append(D)
	sol = np.array(sol)
	np.savetxt('../results/part_2/part2_error_matrix.txt', sol)
	pca_common_func(sol, mean)
	return


#PART3
def pca_similarity(data, mean):
	print('1 SUCESS')
	#create covmat, eigvec
	covmat, eigvec = [], []
	for i in range(len(mean)):
	    covmat.append(np.cov(data[i],rowvar=False))
	for i in range(len(mean)):
	    eigvec.append(np.linalg.eig(covmat[i])[1])
	covmat, eigvec = np.array(covmat), np.array(eigvec)
	
	#set everything after 20 to zero
	for i in range(len(mean)):
		eigvec[i][:, pca_dim:] = 0

	print('2 SUCESS')

	#find error matrix
	cond_err_mat = np.zeros((len(mean), len(mean)))
	err = np.zeros((len(mean), len(mean)))

	data = np.array(data) #ADDED AFTER TEST

	transformed_data = np.zeros(data.shape)
	for i in range(len(mean)):
		mn = mean[i]
		mn_mat = np.tile(mn, (len(data[i]),1)) #ADDED AFTER TEST
		for j in range(len(mean)):
			sum_mat = np.zeros(data[0].shape)
			new_data = np.zeros(data[0].shape)
			for k in range(pca_dim):
				sum_mat += np.outer(np.dot(data[i]-mn_mat, eigvec[j][:,k].reshape((-1,1))),eigvec[j][:,k])
			sum_mat += mn_mat #REPLACE INSTANCES OF MN WITH MN_MAT
			new_data = data[i] - sum_mat
			new_data = np.multiply(new_data, new_data)
			sum_rows = np.sum(new_data, axis=1)
			avg = np.mean(sum_rows)
			cond_err_mat[i][j] = avg

	print('3 SUCESS')

	for i in range(len(mean)):
		for j in range(len(mean)):
			err[i][j] = (cond_err_mat[i][j] + cond_err_mat[j][i])/2

	print('4 SUCESS')
	np.savetxt('part3_error_matrix.txt', err)
	pca_common_func(err, mean)
	return

# THE MAIN FUNCTION
def main():
	df = get_data()

	#Separate data by labels and compute mean image for each label
	data, mean = [], []
	for i in range(0,10):
	    data.append(df[df['labels']==i])
	    data[i] = data[i].drop(['labels'],axis=1)
	    mean_arr = np.mean(data[i],axis=0)
	    mean.append(mean_arr)
	    data[i] = data[i].as_matrix()

	#########################################
	# THESE FUNCTIONS SHOULD BE UNIT TESTED #
	#########################################

	#PART 1
	print_mean(mean)
	covmat_and_error_calc(data, mean)

	#PART 2
	pca_calc(mean)

	#PART 3
	pca_similarity(np.array(data), np.array(mean))
	return


if '__main__' == __name__:
	main()
	print('SUCCESS')