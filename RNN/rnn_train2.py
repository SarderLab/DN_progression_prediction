import sys
import os
import numpy as np
import tensorflow as tf

from data_loader import import_data,random_batcher,import_labels

def train_network2(txt_loc,lbl_loc,txt_loc_v,train_index,learning_rate,training_steps,batch_size,
    save_interval,num_classes,model_path,glom_samplings,drop,predictions):
    display_step = 1

    # Network Parameters
    num_hidden = 50
    num_hidden_2 = 25


    # Where to write the training and validation loss for the current fold (gets overwritten every time
    loss_txt='loss.txt'
    index_txt = 'indices.txt'
    # Import data and labels
    all_features = np.array(import_data(txt_loc))
    labels=np.array(import_labels(lbl_loc,1))
    labels = np.eye(2)[labels]

    val_features = np.array(import_data(txt_loc_v))

    #TRY IT WITH A LONGER PRETRAINING WHY DON"T YA
    #SEE IF THAT FIXES THE INCREASING GRADIENTS
    #TRY TRAINING WITH BOTH PSEUDO AND ACTUAL SETS AT SAME TIME

    pseudo_labels = np.array([np.argmax(x) if np.max(x) > 0.75 else -1 for x in predictions])
    pseudo_set = np.array([[x,y] for x,y in zip(val_features,pseudo_labels) if y!=-1])

    x_values = np.array([x for x,y in pseudo_set])
    y_values = np.array([y for x,y in pseudo_set])
    # Before doing anything, identify the mean and standard deviation of all features,
    # for standardization purposes
    mean_record=[]
    std_record=[]


    for case in all_features:
        mean_record.append(np.mean(np.asarray(case),0))
        std_record.append(np.std(np.asarray(case),0))
    # Ignore NaNs
    feature_mean=np.nanmean(mean_record,0)
    feature_std=np.nanstd(std_record,0)

    # feature_mean = np.nan_to_num(feature_mean)
    feature_std = np.nan_to_num(feature_std)

    feature_mean[feature_mean==np.inf] = 0

    # Split cases for training and cases for testing
    train_features=all_features[train_index]
    train_labels=labels[train_index]

    y_values = np.eye(2)[y_values]

    train_features = np.concatenate((train_features,x_values))
    train_labels = np.concatenate((train_labels,y_values))

    holdout_features=val_features#[test_index]
    holdout_labels=y_values#[test_index]


    # Number of input features for the network
    num_input=len(train_features[0][0])

    # Define placeholders for graph input
    X = tf.placeholder(tf.float32, [None, glom_samplings, num_input])
    Y = tf.placeholder(tf.float32, [None, 2])

    dropout2 = tf.placeholder(tf.float32,shape=(),name='dropout2')

    # Function to return LSTM during graph definition
    def lstm(x_,unit_num,seq_flag):
        lstm_cell =tf.keras.layers.LSTMCell(unit_num, unit_forget_bias=True)
        lstm_layer= tf.keras.layers.RNN(lstm_cell,return_sequences=seq_flag,unroll=True, dtype=tf.float32)
        output=lstm_layer(x_)
        return output

    # Define a function that returns mean squared error on a batch
    def MSE(x,y):
        return np.sum((x-y)**2)/batch_size

    # Dense layer to select input feature importance
    D_in=tf.layers.dense(X,num_input,activation=tf.nn.leaky_relu)
    # First lstm layer
    cell_1 = lstm(x_=D_in,unit_num=num_hidden,seq_flag=True)
    # Dropout between LSTM cells
    cell_1=tf.nn.dropout(cell_1,dropout2)
    # Second LSTM layer
    cell_2 = lstm(x_=cell_1,unit_num=num_hidden_2,seq_flag=False)
    # Network output
    prediction=tf.layers.dense(cell_2,2)
    # Clipped prediction value between 1 and max class value
    #clipped_prediction=tf.clip_by_value(prediction,1,num_classes)
    soft_prediction = tf.nn.softmax(prediction)
    clipped_prediction=tf.cast(tf.math.argmax(prediction,-1),tf.float32)
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Performance metric
    MSE_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=prediction))
    clipped_label = tf.cast(tf.math.argmax(Y,-1),tf.float32)

    val_accuracy =1. - tf.reduce_mean(tf.math.abs(tf.math.subtract(clipped_prediction,clipped_label)))
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Save model
    saver=tf.train.Saver(max_to_keep=int(round(training_steps/save_interval)))

    with tf.Session() as sess:
        # Restore the model
        saver.restore(sess,model_path)

        for step in range(1, training_steps+1):

            # Pull a random batch of sequences from the training data
            batch_x,batch_y = random_batcher(train_features,glom_samplings,batch_size,train_labels)
            # Standardize the features before input to network
            batch_x-=feature_mean
            batch_x/=feature_std
            # Replace NaNs from standardization procedure with zeros or else the gradient is screwed
            batch_x[:,:,65] = 0
            batch_x[:,:,66] = 0
            batch_x[:,:,67] = 0
            batch_x=np.nan_to_num(batch_x)

            # Run the training operation, return the training loss, MSE, and sample predictions
            _,loss, acc,pred = sess.run([train_op,loss_op, val_accuracy,prediction], feed_dict={X: batch_x,
                                                                 Y: batch_y, dropout2: drop})


            if step % display_step == 0 or step == 1:

                # Run the graph for validation, return the validation loss, validation MSE, and a clipped prediction
                # c_p = sess.run([clipped_prediction], feed_dict={X: holdout_batch, dropout2: 1})
                # Print some information to the command line
                print("Pseudo Step " + str(step) + ", Train Loss= " + \
                      "{:.4f}".format(loss)

                      )

                # save the losses for future analysis!
                with open(loss_txt,'a') as f:
                    f.write(str(step)+':'+str(loss)+'\n')

        # for step in range(1, training_steps+1):
        #
        #     # Pull a random batch of sequences from the training data
        #     batch_x,batch_y = random_batcher(train_features,glom_samplings,batch_size,train_labels)
        #     # Standardize the features before input to network
        #     batch_x-=feature_mean
        #     batch_x/=feature_std
        #     # Replace NaNs from standardization procedure with zeros or else the gradient is screwed
        #     # batch_x[:,:,65] = 0
        #     # batch_x[:,:,66] = 0
        #     # batch_x[:,:,67] = 0
        #
        #     batch_x=np.nan_to_num(batch_x)
        #
        #
        #     # Run the training operation, return the training loss, MSE, and sample predictions
        #     _,loss, acc,pred = sess.run([train_op,loss_op, val_accuracy,prediction], feed_dict={X: batch_x,
        #                                                          Y: batch_y, dropout2: drop})
        #
        #
        #     if step % display_step == 0 or step == 1:
        #
        #         # Run the graph for validation, return the validation loss, validation MSE, and a clipped prediction
        #         # c_p = sess.run([clipped_prediction], feed_dict={X: holdout_batch, dropout2: 1})
        #         # Print some information to the command line
        #         print("Tuning Step " + str(step) + ", Train Loss= " + \
        #               "{:.4f}".format(loss)
        #
        #               )
        #
        #         # save the losses for future analysis!
        #         with open(loss_txt,'a') as f:
        #             f.write(str(step)+':'+str(loss)+'\n')
            # save some checkpoints
            if (step+1) % save_interval == 1:
                saved_path=saver.save(sess, model_path + '/' +str(step)+'_model.ckpt')
        return saved_path
