import sys
import os
import tensorflow as tf
import numpy as np

from data_loader import import_data,random_batcher,import_labels

def predict_holdout(txt_loc,lbl_loc,test_index,model_path,num_classes,batch_size):

    # Import feature and labels!
    all_features = np.array(import_data(txt_loc))
    labels=np.array(import_labels(lbl_loc,1))
    labels = np.eye(2)[labels]
    # Network parameters
    num_hidden = 50
    num_hidden_2 = 25


    # Get data for standardization
    mean_record=[]
    std_record=[]
    for case in all_features:
        mean_record.append(np.mean(np.asarray(case),0))
        std_record.append(np.std(np.asarray(case),0))
    feature_mean=np.nanmean(mean_record,0)
    feature_std=np.nanstd(std_record,0)

    feature_mean = np.nan_to_num(feature_mean)
    feature_std = np.nan_to_num(feature_std)

    feature_mean[feature_mean==np.inf] = 0

    # Pull the holdout features
    holdout_features=all_features[test_index]
    holdout_labels=labels[test_index]

    # Number of input features
    num_input=len(holdout_features[0][0])

    # Unroll each set of features so that we have a vector of all glomeruli for each case!
    holdout_l=holdout_labels
    holdout_f=holdout_features
    holdout_f_vector=[]
    for case in range(0,len(holdout_f)):
        case_features=np.zeros((len(holdout_f[case]),len(holdout_f[case][0])))
        for glom in range(0,len(holdout_f[case])):
            case_features[glom,:]=holdout_f[case][glom]
        holdout_f_vector.append(case_features)

    # Placeholders for graph input
    X = tf.placeholder(tf.float32, [None, None, num_input])
    Y = tf.placeholder(tf.float32, [None, 2])

    dropout2 = tf.placeholder(tf.float32,shape=(),name='dropout2')

    # Function to set up LSTM
    def lstm(x_,unit_num,seq_flag):
        lstm_cell =tf.keras.layers.LSTMCell(unit_num, unit_forget_bias=True)
        lstm_layer= tf.keras.layers.RNN(lstm_cell,return_sequences=seq_flag, dtype=tf.float32)
        output=lstm_layer(x_)
        return output

    # For performance analysis
    def MSE(x,y):
        return np.sum((x-y)**2)/batch_size

    # Dense layer to choose input importance
    D_in=tf.layers.dense(X,num_input,activation=tf.nn.leaky_relu)

    # First LSTM unit
    cell_1 = lstm(x_=D_in,unit_num=num_hidden,seq_flag=True)
    # Inter-LSTM dropout
    cell_1=tf.nn.dropout(cell_1,dropout2)
    # Second LSTM unit
    cell_2 = lstm(x_=cell_1,unit_num=num_hidden_2,seq_flag=False)
    # The predicted value
    prediction=tf.layers.dense(cell_2,2)
    # A clipped version of the predicted value
    #clipped_prediction=tf.clip_by_value(prediction,1,num_classes)
    soft_prediction = tf.nn.softmax(prediction)
    clipped_prediction=tf.cast(tf.math.argmax(prediction,-1),tf.float32)
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=prediction))

    # Performance metric
    MSE_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=prediction))

    # Initialize saver object to restore the model
    saver=tf.train.Saver()

    with tf.Session() as sess:
        # Restore the model
        saver.restore(sess,model_path)
        # Place to keep predicted values
        predictions=np.zeros((len(holdout_f_vector),batch_size,2))

        # For all cases
        for step in range(0,len(holdout_f_vector)):
            # Get the holdout cases
            holdout_idx=step
            holdout_batch=np.zeros((batch_size,len(holdout_f_vector[step]),num_input))
            holdout_label=np.zeros((batch_size,2))

            # Randomly permute the sequence of gloms to create a batch of testing sequences
            for step_i in range(0,batch_size):
                holdout_batch[step_i,:,:]=holdout_f_vector[step]
                np.random.shuffle(holdout_batch[step_i,:,:])
                holdout_label[step_i,:]=labels[holdout_idx,:]
            # Standardize
            holdout_batch-=feature_mean
            holdout_batch/=feature_std
            holdout_batch[:,:,65] = 0
            holdout_batch[:,:,66] = 0
            holdout_batch[:,:,67] = 0

            holdout_batch=np.nan_to_num(holdout_batch)
            c_p = sess.run([soft_prediction], feed_dict={X: holdout_batch, Y: holdout_label, dropout2: 1})

            predictions[step,:,:]=np.squeeze(c_p)
        #predictions are now in the shape of [cases,1024/case,2]
        predictions = np.squeeze(np.mean(predictions,1))
        #predictions are now in the shape of [cases,2]

    return predictions
