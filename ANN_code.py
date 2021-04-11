#!/usr/bin/env python
# coding: utf-8

# # ------------------------------ ANN TOXICITY PREDICTION MODEL-------------------------
# 

# ### Including Packages

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ### Importing Dataset

# In[3]:


protein_df = pd.read_csv("G:/ABHINAV/IITG/SEMESTER 8/BTP/my_BTP_phase_II/BTP_DATASETS/Final_datasets/protein_data_final.csv")
protein_df = protein_df.dropna(axis = 0)
protein_df


# In[59]:


protein_df.dtypes #need to convert object_type
# protein_df.shape
vl = protein_df.Length.value_counts()
vl = vl.sort_index()
vl_cnts = list(vl)
print(vl_cnts)
# print(vl)
print(sum(vl_cnts[0:35]))


# In[60]:


count_len = protein_df.groupby('Length').apply(lambda df : df.toxicity[df.toxicity == 1].count())
count_len = count_len.sort_index()  
# count_len
cnt_l = list(count_len)

print(sum(cnt_l[0:35]))
# print(sum(cnt_l[80:]))


# In[61]:


s = (protein_df.dtypes == 'object')
s[s].index


# ### Converting datatype

# In[62]:


protein_df['Mass_float'] = protein_df['Mass'].map(lambda x : float(x.replace(',', '')))


# In[63]:


for i in range(100):
    protein_df = protein_df.astype({str(i) : 'float'})
protein_df = protein_df.astype({'Organism ID':'float', 'Length':'float', 'Taxonomic lineage IDs':'float','Taxonomic lineage IDs':'float','toxicity':'float'})


# ### Feature Scaling 

# #### Normalizing mass

# In[64]:


mass_mean = protein_df.Mass_float.mean()
mass_std = protein_df.Mass_float.std()

print(mass_std)
Mass_norm = protein_df.Mass_float.map(lambda p : (p - mass_mean) / mass_std)
protein_df['Mass_norm'] = Mass_norm


# #### Normalizing length

# In[65]:


len_max = protein_df.Length.max()
len_min = protein_df.Length.min()
eps = 1e-5
print(len_max, len_min)
protein_df['Length_norm'] = (protein_df.Length - len_min) / (len_max - len_min + eps)


# #### Normalizing Sequence 

# In[66]:


seq = protein_df.Sequence
prot_seq = np.zeros((42933, 100), dtype = float)
idx = 0
for s in seq:
    idy = 0
    for i in s:
#         print(s[i])
        prot_seq[idx][idy] = ((((ord(i) - ord('A')) + 1) / 26.0) *2.0) - 1.0
#         prot_seq[idx][idy] = (((ord(i) - ord('A')) + 1) / 26.0) 
        idy += 1
    idx += 1
# prot_seq[0 : 20][:]

indcs = []
for i in range(100):
    s = "L"
    s = s + str(i + 1)
    indcs.append(s)
# print(indcs)


# In[67]:


df2 = pd.DataFrame(prot_seq, columns=indcs)
protein_df = pd.concat([protein_df, df2], axis = 1)


# In[68]:


# df2


# In[69]:


protein_df.shape


# ### Selecting features

# In[70]:


y = protein_df.toxicity #output feature

features = ['Length_norm', 'Mass_norm']
# features = ['Length_norm']
features = features + indcs

X = protein_df[features]
X.head()


# In[ ]:





# ### Splitting Dataset

# In[73]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1,
                                                      random_state=0)


# In[74]:


X_valid.values.shape
# y_valid.value_counts()


# ### Model Fitting and Prediction

# In[126]:


import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, callbacks
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Lambda
from sklearn.utils import class_weight


# early_stopping = callbacks.EarlyStopping(min_delta=0.001, patience=20, restore_best_weights=True,)


# define the keras model
model = Sequential()
model.add(Dense(256, input_dim=102, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(8, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(1, activation='sigmoid'))
# model.add(Lambda(round_x))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Recall', 'Precision', 'accuracy'])

# weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
# weights = [0.51, 4.665]
# weights = [0.51, 4.665]


# In[127]:


pr_model = model.fit(X_train, y_train,validation_data=(X_valid, y_valid), epochs=100, batch_size=128)
# callbacks=[early_stopping]
# class_weight={0:weights[0],1:weights[1]} 




# In[128]:


# evaluate the keras model
_, recall, precision, accuracy = model.evaluate(X_valid, y_valid)
print('\nAccuracy: %.2f' % (accuracy*100))
print("precision, recall", precision, recall)


# In[129]:


# convert the training history to a dataframe
history_df = pd.DataFrame(pr_model.history)
history_df.columns


# In[130]:


history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))


# In[131]:




# use Pandas native plot method
history_df['loss'].plot();


# In[132]:


f1 = (2.0 * precision * recall)/ (precision + recall)
print(f1)


# In[133]:


y_pred = model.predict(X_valid)
y_pred = np.round(y_pred)
y_pred = np.squeeze(y_pred)
y_pred
y_p = pd.Series(y_pred)
print ("predicted : ")
y_p.value_counts()


# In[134]:


print("actual: ")
y_valid.value_counts()
# y_valid.shape


# In[135]:


bl = y_valid == y_pred
blt = bl.tolist()
print(type(blt))


# In[ ]:





# In[136]:


cnt1 = 0
cnt0 = 0

for i in range(len(blt)):
    if (blt[i] and y_pred[i] >= 0.5):
        cnt1 += 1
    elif blt[i] and y_pred[i] < 0.5:
        cnt0 += 1
print(cnt1, cnt0)        


# In[ ]:





# In[ ]:





# In[ ]:




