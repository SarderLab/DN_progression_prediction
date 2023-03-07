import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
# from scipy.ndimage.filters import gaussian_filter1d
from sklearn.model_selection import train_test_split
import pandas as pd
#LOAD DATA HERE
data = pd.read_csv('Glomeruli_combined_sig_proteomics.csv',header=None).to_numpy()
print(data.shape)
val_data = pd.read_csv('validation_sig_proteomics_2.csv',header=None).to_numpy()
print(val_data.shape)
labels = pd.read_csv('Glomeruli_combined_labels.csv',header=None).to_numpy()
labels = labels[:,1]


print(data.shape)
print(val_data.shape)
# exit()
def build_net():

    inputs = tf.keras.Input(shape = (45,))

    x = layers.Dense(30,kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)

    # x = layers.Dense(1600)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU()(x)
    # x = layers.Dropout(0.5)(x)

    # x = layers.Dense(1200)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU()(x)
    # x = layers.Dropout(0.5)(x)

    x = layers.Dense(30,kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)

    # x = layers.Dense(15,kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU()(x)
    # x = layers.Dropout(0.5)(x)

    x = layers.Dense(2,kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    outputs = layers.Softmax()(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    return model


model = build_net()

def loss(prediction,label):

    calc = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = calc(label,prediction)

    return tf.reduce_mean(loss)



optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(train_batch,train_label):
    with tf.GradientTape() as tape:

        prediction = model(train_batch,training=True)
        batch_loss = loss(prediction,train_label)

        grad = tape.gradient(batch_loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grad,model.trainable_variables))

        return batch_loss

@tf.function
def test_step2(test_batch):
    with tf.GradientTape() as tape:
        prediction = model(test_batch,training=False)


        return prediction


@tf.function
def test_step(test_batch,test_label):
    test = model(test_batch,training=False)
    test_loss = loss(test,test_label)
    clipped_prediction=tf.cast(tf.math.argmax(test,-1),tf.float32)
    clipped_label = tf.cast(test_label,tf.float32)
    val_accuracy =1. - tf.reduce_mean(tf.math.abs(tf.math.subtract(clipped_prediction,clipped_label)))

    return test_loss,test,test_label,val_accuracy

def train(data,val_data,labels,steps):
    # data,holdout,labels,holdout_labels = train_test_split(data,labels,test_size=1)
    for train_indices,test_indices in kf.split([i for i in range(56)]):
        data_ = data[train_indices,:]
        labels_ = labels[train_indices]

        test_set = data[test_indices,:]
        test_labels = labels[test_indices]

        ds_data = tf.data.Dataset.from_tensor_slices(data_)
        ds_labels = tf.data.Dataset.from_tensor_slices(labels_)

        ds = tf.data.Dataset.zip((ds_data,ds_labels))
        ds = ds.repeat()
        ds = ds.shuffle(50,reshuffle_each_iteration=True)
        ds = ds.batch(50,drop_remainder=True)
        ds = ds.prefetch(buffer_size=1)


        ds_test = val_data


        iter = 0

        for idx,batch in enumerate(ds):
            tr_b = batch[0]
            tr_y = batch[1]

            tr_l = train_step(tr_b,tr_y)
            tr_losses.append(tr_l)

            # h_l,h_acc,lab,val_acc = test_step(test_set,test_labels)
            # holdout_losses.append(h_l)
            #
            # h_acc = h_acc.numpy()
            # lab = lab.numpy()
            # h_acc = np.argmax(h_acc,-1)
            # final_preds.append(h_acc)
            # final_preds.append(lab)
            # v_acc.append(val_acc)

            iter+=1

            if iter == steps:

                for i in range(5):
                    test = test_step2(ds_test)

                    pseudo_labels = np.array([np.argmax(x) if np.max(x) > 0.75 else -1 for x in test])
                    pseudo_set = np.array([[x,y] for x,y in zip(val_data,pseudo_labels) if y!=-1])

                    x_values = np.array([x for x,y in pseudo_set])
                    y_values = np.array([y for x,y in pseudo_set])

                    all_data = np.concatenate((data_,x_values))
                    all_labels = np.concatenate((labels_,y_values))

                    ds_2 = tf.data.Dataset.from_tensor_slices(all_data)
                    ds_2l = tf.data.Dataset.from_tensor_slices(all_labels)

                    ds2 = tf.data.Dataset.zip((ds_2,ds_2l))
                    ds2 = ds2.repeat()
                    ds2 = ds2.shuffle(50,reshuffle_each_iteration=True)
                    ds2 = ds2.batch(50,drop_remainder=True)
                    ds2 = ds2.prefetch(buffer_size=1)

                    iter2 = 0

                    for idx,batch in enumerate(ds2):
                        tr_b = batch[0]
                        tr_y = batch[1]

                        tr_l = train_step(tr_b,tr_y)
                        tr_losses.append(tr_l)

                        iter2+=1

                        if iter2==100:
                            break


                h_l,h_acc,lab,val_acc = test_step(test_set,test_labels)
                holdout_losses.append(h_l)

                h_acc = h_acc.numpy()
                lab = lab.numpy()
                # h_acc = np.argmax(h_acc,-1)
                final_preds.append(h_acc)
                final_labs.append(lab)
                break

tr_losses = []
te_losses = []
holdout_losses = []
final_preds = []
final_labs = []
v_acc = []

kf = KFold(10,shuffle = True)

for i in range(10):
    train(data,val_data,labels,500)

out_text = 'predictions.txt'
with open(out_text,'w') as f:
    for i in range(0,len(final_preds)):
        cases = len(final_preds[i])
        idx=0
        for case in final_preds[i]:
            f.write(str(case[0])+','+str(case[1])+','+str(final_labs[i][idx])+'\n')
            idx+=1
