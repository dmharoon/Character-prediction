import csv
import numpy as np
from scipy.misc import imsave
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os
import h5py
from keras import backend as K
import time
import difflib
import h5py as h5py


with open('train.csv','r') as csvfile, open('test.csv','r') as csv_test:
    rows = csv.reader(csvfile, delimiter=',')
    rows_test=csv.reader(csv_test,delimiter=',')
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    x_nxt_idx=[]
    xtst_nxt_idx=[]
    y_nxtid=[]
    words=[]
    char=''
    for i,row in enumerate(rows):
        if i is not 0:
            if int(row[2]) is not -1 :
                char = char + row[1]                        
            else:
                char = char + row[1]               
                words.append(char)
                char = ''
            y_train.append(row[1])
            x_nxt_idx.append(row[2])
            if i==450: print (row[1])
            row = np.reshape(np.array(row[4:]),(16,8))
            x_train.append(row)
    for i,row in enumerate(rows_test):
        if i is not 0:
            y_nxtid.append(int(row[2]))
            y_test.append(row[1])
            xtst_nxt_idx.append(row[2])
            if i==450: print (row[1])
            row = np.reshape(np.array(row[4:]),(16,8))
            x_test.append(row)


x_train=np.array(x_train)
x_test=np.array(x_test)

#print (x_train.shape)
#print (x_test.shape)
#print ('y_train:',y_train.shape)
#print (y_train[449],y_test[200])


# In[58]:


x3_train = np.zeros((x_train.shape[0],16,24,1))
x3_test = np.zeros((x_test.shape[0],16,24,1))
for i in range(x_train.shape[0]):
    x3_train[i,:,8:16,0] = x_train[i]
    if x_nxt_idx[i] != -1 and i < x_train.shape[0]-1:
        x3_train[i,:,16:24,0] = x_train[i+1]
        #if x_nxt_idx[x_nxt_idx[i]] != -1 and i < x_train.shape[0]-2:
        #    x3_train[i,:,24:32,0] = x_train[i+2]
    if i>0 and x_nxt_idx[i-1] != -1:
        x3_train[i,:,0:8,0] = x_train[i-1]
for i in range(x_test.shape[0]):
    x3_test[i,:,8:16,0] = x_test[i]
    if xtst_nxt_idx[i] != -1 and i < x_test.shape[0]-1:
        x3_test[i,:,16:24,0] = x_test[i+1]
        #if xtst_nxt_idx[xtst_nxt_idx[i]] != -1 and i < x_test.shape[0]-2:
        #    x3_test[i,:,24:32,0] = x_test[i+2]
    if i > 0 and xtst_nxt_idx[i-1] != -1:
        x3_test[i,:,0:8,0] = x_test[i-1]
#x3_train.shape
#x3_test.shape


# In[59]:


set_y=set(y_train)
match={}
for char in set_y:
    match[char]=ord(char)-97
#for key in sorted(match.keys()):
#    print ("%s: %s" % (key, match[key]))
for i,label in enumerate(y_train):
    y_train[i]=match[label]
for i,label in enumerate(y_test):
    y_test[i]=match[label]


# In[60]:


model_no=1
num_classes=26
batch_size = 64
epochs= 25
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_handwriting_trained_model %d.h5'%model_no


# In[61]:


y_train=np.array(y_train)
y_test=np.array(y_test)
#new_shape_xtr = x3_train.shape + (1,)
new_shape_ytr = y_train.shape + (1,)
#new_shape_xtst = x3_test.shape + (1,)
new_shape_ytst = y_test.shape + (1,)
#x3_train =x3_train.reshape(new_shape_xtr)
y_train =y_train.reshape(new_shape_ytr)
#x3_test =x3_test.reshape(new_shape_xtst)
y_test =y_test.reshape(new_shape_ytst)


# In[62]:


y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)


# In[63]:


print (x3_train.shape)
print (x3_test.shape)


# In[64]:


model = Sequential()
model.add(Conv2D(64,(5,5),padding='same',input_shape=(16,24,1)))
model.add(Activation('relu'))
model.add(Conv2D(128,(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
model.add(Conv2D(256,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
#model.add(Conv2D(256,(3,3),padding='same'))
#model.add(Activation('relu'))
#model.add(Conv2D(256,(3,3)))
#model.add(Activation('relu'))
#model.add(Conv2D(512,2))
#model.add(Activation('relu'))
model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Dropout(0.5))
#model.add(Dense(256))
#model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# In[65]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[67]:


x3_train = x3_train.astype('float32')
x3_test = x3_test.astype('float32')


print (x3_train.shape,x3_test.shape,y_train.shape,y_test.shape)


# In[ ]:

earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
checkpointer = keras.callbacks.ModelCheckpoint(filepath='weights_%s.hdf5'%model_no, verbose=1, save_best_only=True)


start=time.clock()
model.fit(x3_train,y_train,batch_size=batch_size,epochs=epochs,validation_split=0.2,shuffle=True)
time_taken = time.clock() - start
print (time_taken)


# In[ ]:

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x3_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[ ]:


predictions=model.predict(x3_test)

result=[]
result.append(['Id','Prediction'])
for i in range(predictions.shape[0]):
    label = (np.argmax(predictions[i]))
    result.append([i,chr(label+97)])


# In[ ]:



## For Ensemble classifier
# d=[1,2,3,4,5,6,7]
# f=[]
# rows=[]
# for i in range(7):
#     f.append(open('submissions/sub2/Submission%d.csv'%d[i],'r'))
#     rows.append(csv.reader(f[i], delimiter=','))
    
# result=[]
# result.append(['Id','Prediction'])
# for (iii,row0),(i, row1),(j,row2),(k,row3),(ii, row4),(jj,row5),(kk,row6) in zip(enumerate(rows[0]),enumerate(rows[1]),enumerate(rows[2]),enumerate(rows[3]),enumerate(rows[4]),enumerate(rows[5]),enumerate(rows[6])):
    
#     c= Counter([row0[1],row1[1],row2[1],row3[1],row4[1],row5[1],row6[1]])
#     if i!=0: 
#         result.append([i-1,(c.most_common()[0][0])])
    
# for i in range(7):
#     f[i].close()


#print (set(words))

wrd = ''
result_new=[]
result_new.append(['Id','Prediction'])
for i in range(len(x_test)):
    if y_nxtid[i] == -1:
        wrd += result[i+1][1]
        replWrd = difflib.get_close_matches(wrd,list(set(words)))
        for j in range(len(wrd)):
            idx = [k for k in range(len(replWrd)) if len(replWrd[k])==len(wrd)]
            if len(idx) is not 0 :
                result_new.append([i-len(wrd)+j+1,replWrd[idx[0]][j]])
            else: 
                result_new.append([i-len(wrd)+j+1,wrd[j]])
        wrd=''
    else:
        wrd += result[i+1][1]
print (len(result_new), len(y_test))


model_no=100
with open('Submission%d.csv'%model_no,'w') as outfile:
    writer=csv.writer(outfile)
    writer.writerows(result_new)