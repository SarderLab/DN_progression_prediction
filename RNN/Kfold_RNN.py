import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from data_loader import import_data,import_labels
from rnn_train import train_network
from rnn_train2 import train_network2
from rnn_predict import predict_holdout
import pandas as pd

def make_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) # make directory if it does not exit already # make new directory
# choose GPU or CPU device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Define location of features and labels
txt_loc='/blue/pinaki.sarder/nlucarelli/FeatureExtraction/DN/Abnormal_tubule_features.txt'
lbl_loc='/blue/pinaki.sarder/nlucarelli/FeatureExtraction/DN/og_label_file.csv'

txt_loc_v = '/blue/pinaki.sarder/nlucarelli/FeatureExtraction/DN_validation/Abnormal_tubule_features.txt'
lbl_loc_v = '/blue/pinaki.sarder/nlucarelli/FeatureExtraction/DN_validation/validation_Abnormal_tubule_labels.csv'

# Define where to store saved models
model_path='/blue/pinaki.sarder/nlucarelli/FeatureExtraction/DN/models_update2'
make_folder(model_path)
# Define name of text file to write predictions to
out_text=model_path+'/prot_predictions_tub2_semi.txt'
f=open(out_text,'w')
f.close()

loss_txt = 'loss.txt'
f=open(loss_txt,'w')
f.close()

index_txt = 'indices.txt'
f = open(index_txt,'w')
f.close()
# Import the features
all_features = np.array(import_data(txt_loc))
val_features = np.array(import_data(txt_loc_v))
print(len(all_features))
print(len(all_features[0][0]))
# Import the labels
labels=np.array(import_labels(lbl_loc,1))
labels_v = np.array(import_labels(lbl_loc_v,1))

predictions=[]
test_boys = []
p_stats=[]
fold=0
# Dropout between LSTM units
drop=0.5
# Initial learning rate
learning_rate = 0.001
# Number of steps to train
training_steps = 500
tuning_steps = 100
# Size of each training batch
batch_size = 256
# Length of glomerular input sequence for training!
glom_samplings=10
# Number of times to shuffle full-length glomerular sequences for prediction

predict_shuffles=1024
# How often to save model parameters, set to save only once
save_interval=training_steps
save_interval2 = tuning_steps
#Determine number of classes from labels
num_classes = np.max(np.array(labels))
print(num_classes)
kf=KFold(10,shuffle=True)
for trials in range(10):
    for train_index, test_index in kf.split(all_features):
        print('---------------------------------------------------')
        print('Go!')
        print('Trial: '+str(trials))
        print(' Fold: '+str(fold))
        print('---------------------------------------------------')
        tf.reset_default_graph()
        #Train the network and return the path of the saved model parameters
        saved_model_path=train_network(txt_loc,lbl_loc,txt_loc_v,train_index,learning_rate,training_steps,batch_size,
            save_interval,num_classes,model_path,glom_samplings,drop)
        tf.reset_default_graph()

        for i in range(5):

        #Predict with the network on the holdout set and full-length glomerular sequences
            p=predict_holdout(txt_loc_v,lbl_loc_v,list(range(len(val_features))),saved_model_path,num_classes,predict_shuffles)
            tf.reset_default_graph()
            saved_model_path = train_network2(txt_loc,lbl_loc,txt_loc_v,train_index,learning_rate,tuning_steps,batch_size,
                save_interval2,num_classes,saved_model_path,glom_samplings,drop,p)
            tf.reset_default_graph()
            # predictions.append(p)
            # l=labels_v#[test_index]
        p=predict_holdout(txt_loc,lbl_loc,test_index,saved_model_path,num_classes,predict_shuffles)

# Save the holdout predictions for performance analysis
        l=labels[test_index]
        for idx in range(len(p)):
            p_stats.append([p[idx,0],p[idx,1],l[idx]])
            test_boys.append(test_index[idx])
        fold+=1
# Write predicitons to text file
print(p_stats)
with open(out_text,'a') as f:
    for l in range(0,len(p_stats)):
        f.write(str(p_stats[l][0])+','+str(p_stats[l][1])+','+str(p_stats[l][2])+','+str(test_boys[l])+'\n')
