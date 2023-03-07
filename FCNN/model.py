import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras import layers
from data_loader_glom import import_data

net_type = 'glom' #'glom'/'tub'
training_steps = 500
folds = 10

loss_file = 'losses.txt'
f = open(loss_file,'w')
f.close()

performance_file = 'predictions.txt'
f = open(performance_file,'w')
f.close()

glom_file = 'Glomeruli_combined_features.txt'
tub_file = 'Abnormal_tubule_features.txt'

glom_data = import_data(glom_file)
tub_data = import_data(tub_file)

prot_labs = pd.read_csv('prot_label.csv',header=None).to_numpy()

gmean_record=[]
gstd_record=[]

tmean_record = []
tstd_record = []

for case in glom_data:
    gmean_record.append(np.mean(np.asarray(case),0))
    gstd_record.append(np.std(np.asarray(case),0))
# Ignore NaNs
gfeature_mean=np.nanmean(gmean_record,0)
gfeature_std=np.nanstd(gstd_record,0)

for case in tub_data:
    tmean_record.append(np.mean(np.asarray(case),0))
    tstd_record.append(np.std(np.asarray(case),0))

tfeature_mean = np.nanmean(tmean_record,0)
tfeature_std = np.nanstd(tstd_record,0)

if net_type == 'glom':
    feature_mean = gfeature_mean
    feature_std = gfeature_std
elif net_type == 'tub':
    feature_mean = tfeature_mean
    feature_std = tfeature_std
else:
    print('incorrect network selected')
    exit()

prot_labs = np.log(prot_labs+1)
prot_mins = np.amin(prot_labs,0)
prot_maxs = np.amax(prot_labs,0)
prot_labs = (prot_labs - prot_mins) / (prot_maxs - prot_mins)

if net_type == 'glom':
    i_size = len(glom_data[0][0])
elif net_type == 'tub':
    i_size = len(tub_data[0][0])
else:
    print('incorrect network selected')
    exit()

def build_net():

    inputs = tf.keras.Input(shape = i_size)

    x = layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    outputs = layers.LeakyReLU()(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    return model

model_build = build_net()

def loss(prediction,label):

    loss = tf.reduce_mean(abs(prediction-label))

    return loss

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(train_batch,train_label):
    with tf.GradientTape() as tape:
        prediction = model_build(train_batch,training=True)
        batch_loss = loss(prediction,train_label)

        grad = tape.gradient(batch_loss,model_build.trainable_variables)
        optimizer.apply_gradients(zip(grad,model_build.trainable_variables))

        return batch_loss

@tf.function
def test_step(test_batch,test_label):
    test = model_build(test_batch,training=False)
    test_loss = loss(test,test_label)
    clipped_prediction = tf.cast(test,tf.float32)
    clipped_label = tf.cast(test_label,tf.float32)

    return test_loss,clipped_prediction,clipped_label

def train(data,labels,steps):

    for i in range(1):#train_indices,test_indices in kf.split([i for i in range(prot_labs.shape[0])]):

        test_indices = [j for j in range(prot_labs.shape[0])]
        data_p = np.asarray(data)
        #data_p = data_p[train_indices]

        labels_p = labels#[train_indices]

        test_set_p = np.asarray(data)
        #test_set_p = test_set_p[test_indices]
        test_labels_p = labels#[test_indices]

        data_ = np.empty((0,i_size))
        labels_ = np.empty((0))

        for i in range(len(data_p)):
            temp_ = data_p[i]
            temp_ = np.asarray(temp_)
            temp_l = np.tile(labels_p[i],temp_.shape[0])
            data_ = np.concatenate((data_,temp_),0)
            labels_ = np.concatenate((labels_,temp_l),0)

        test_set = np.empty((0,i_size))
        test_labels = np.empty((0))
        test_indices_ = np.empty((0))

        for i in range(len(test_set_p)):
            temp_ = test_set_p[i]
            temp_ = np.asarray(temp_)
            temp_l = np.tile(test_labels_p[i],temp_.shape[0])
            temp_idx = np.tile(test_indices[i],temp_.shape[0])
            test_set = np.concatenate((test_set,temp_),0)
            test_labels = np.concatenate((test_labels,temp_l),0)
            test_indices_ = np.concatenate((test_indices_,temp_idx),0)

        shuffle_size = data_.shape[0]

        data_-= feature_mean
        data_/= feature_std

        test_set-=feature_mean
        test_set/=feature_std

        labels_ = labels_.astype('float32')
        test_labels = test_labels.astype('float32')

        ds_data = tf.data.Dataset.from_tensor_slices(data_)
        ds_labels = tf.data.Dataset.from_tensor_slices(labels_)

        ds = tf.data.Dataset.zip((ds_data,ds_labels))
        ds = ds.repeat()
        ds = ds.shuffle(shuffle_size,reshuffle_each_iteration=True)
        ds = ds.batch(400,drop_remainder=True)
        ds = ds.prefetch(buffer_size=1)

        ds_test = test_set
        ds_test_y = test_labels

        iter=0

        for idx,batch in enumerate(ds):
            tr_b = batch[0]
            tr_y = batch[1]

            tr_l = train_step(tr_b,tr_y)
            tr_losses.append(tr_l)

            h_l,h_p,h_y = test_step(ds_test,ds_test_y)
            ho_losses.append(h_l)

            h_p = h_p.numpy()
            h_y = h_y.numpy()

            iter+=1

            if iter == steps:
                for idx in test_indices_:
                    indices.append(idx)

                te_losses.append(h_l)

                for i in h_p:
                    final_preds.append(i)
                for i in h_y:
                    final_targets.append(i)

                vars = model_build.trainable_variables
                vars = vars[0].numpy()


                break
    return vars



if net_type == 'glom':
    i_data = glom_data
elif net_type == 'tub':
    i_data = tub_data
else:
    print('incorrect network selected')
    exit()

weights = np.zeros((i_size,prot_labs.shape[1]))

for i in range(prot_labs.shape[1]):

    tr_losses = []
    ho_losses = []
    te_losses = []

    final_preds = []
    final_targets = []
    indices = []

    labels = prot_labs[:,i]
    kf = KFold(folds,shuffle=True)

    coeff = train(i_data,labels,training_steps)
    weights[:,i] = np.squeeze(coeff)

    with open(loss_file,'a') as f:
        for j in range(len(tr_losses)):
            f.write(str(tr_losses[j]) + ',' + str(ho_losses[j]) + '\n')
        f.write('---'+'\n')

    with open(performance_file,'a') as f:
        for j in range(len(final_preds)):
            f.write(str(indices[j]) + ',' + str(final_preds[j]) + ',' + str(final_targets[j]) + '\n')
        f.write('---'+'\n')

write_var = pd.DataFrame(weights)
write_var.to_csv('weights.csv')


#PREPARE DATA
