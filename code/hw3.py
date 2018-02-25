# In[121]:


import numpy as np
import pandas as pd


# In[122]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# In[123]:
file = '../data/'

data_batch_1 = unpickle(file + 'data_batch_1')
data_batch_2 = unpickle(file + 'data_batch_2')
data_batch_3 = unpickle(file + 'data_batch_3')
data_batch_4 = unpickle(file + 'data_batch_4')
data_batch_5 = unpickle(file + 'data_batch_5')
test_batch = unpickle(file + 'test_batch')


# In[124]:


data1 = data_batch_1[b'data']
data2 = data_batch_2[b'data']
data3 = data_batch_3[b'data']
data4 = data_batch_4[b'data']
data5 = data_batch_5[b'data']
test = test_batch[b'data']


# In[125]:


data = np.concatenate([data1,data2,data3,data4,data5,test],axis=0)

# In[127]:


data_label_1 = data_batch_1[b'labels']
data_label_2 = data_batch_2[b'labels']
data_label_3 = data_batch_3[b'labels']
data_label_4 = data_batch_4[b'labels']
data_label_5 = data_batch_5[b'labels']
test_label = test_batch[b'labels']
labels = np.concatenate([data_label_1, data_label_2, data_label_3, data_label_4, data_label_5,test_label])
df_data = pd.DataFrame(data)
df_labels = pd.Series(labels, name="labels")
df_labels = pd.DataFrame(df_labels)
df = pd.concat([df_data,df_labels],axis=1)


# In[128]:


#Separate data by labels and compute mean image for each label

data = []
mean = []
for i in range(0,10):
    data.append(df[df['labels']==i])
    data[i] = data[i].drop(['labels'],axis=1)
    mean.append(data[i].mean())
    data[i] = data[i].as_matrix()


# In[129]:


#Calculate covmat for each category
covmat = []
for i in range(0,10):
    covmat.append(np.cov(data[i],rowvar=False))


# In[130]:


data_norm = []
for i in range(0,10):    
    data_norm.append(data[i].astype(float))
    for j in range(0,len(data[i])):
        data_norm[i][j] = data_norm[i][j] - mean[i]


# In[131]:


#Find matrix of eigenvectors for covmat
eig = []
for i in range(0,10):
    eig.append(np.linalg.eig(covmat[i]))


# In[132]:


error = []

for i in range(0,10):
    sum_20 = 0
    for j in range(0,20):
        sum_20 += eig[i][0][j]
    error.append(eig[i][0].sum() - sum_20)


# In[133]:


import matplotlib.pyplot as plt
df_error = pd.DataFrame(error)
bar_graph = df_error.plot(kind='bar',title='PCA Error Per Category', figsize=(10,8),fontsize=12, legend=False)
bar_graph.set_ylabel("Error",fontsize=12)
bar_graph.set_xlabel("Category",fontsize=12)
plt.show()


# In[109]:


'''
eig_vects_0 = eig_0[1]
eig_vects_0_T = np.transpose(eig_vects_0)
data_norm_0_T = np.transpose(data_norm_0)
for i in range(0,5000):
    r[i] = np.matmul(eig_vects_0_T,data_norm_0[i])
    if(i%50 == 0): print(i/50)
#r = np.matmul(data_norm_0,eig_vects_0_T)
'''


# In[102]:


'''
covmat_0_r = np.matmul(eig_vects_0_T,covmat_0)
covmat_0_r = np.matmul(covmat_0_r,eig_vects_0)
covmat_0_pc = np.zeros((20,20))
for i in range(0,20):
    for j in range(0,20):
        covmat_0_pc[i,j] = covmat_0_r[i,j]
'''

