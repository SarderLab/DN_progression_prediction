import sys
import os
import numpy as np
import tensorflow as tf

from data_loader import import_data,random_batcher,import_labels

def train_network(txt_loc,lbl_loc,txt_loc_v,train_index,learning_rate,training_steps,batch_size,
    save_interval,num_classes,model_path,glom_samplings,drop):

    # How often to display training information. This also controls how often to validate from the holdout data
    display_step = 1

    # Network Parameters
    num_hidden = 50
    num_hidden_2 = 25


    # Where to write the training and validation loss for the current fold (gets overwritten every time)
    loss_txt='loss.txt'
    index_txt = 'indices.txt'
    # Import data and labels
    all_features = np.array(import_data(txt_loc))
    labels=np.array(import_labels(lbl_loc,1))
    labels = np.eye(2)[labels]

    # val_features = np.array(import_data(txt_loc_v))


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

    # holdout_features=val_features#[test_index]
    # holdout_labels=labels#[test_index]


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

    #Save holdout indices and labels to loss file in case we want to know later
    # f=open(loss_txt,'w')
    # f=open(index_txt,'a+')
    # for h in test_index:
    #     f.write(str(h))
    #     f.write(',')
    # for h in holdout_labels:
    #     f.write(str(h))
    #     f.write(',')
    # f.write('\n')
    # f.close()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        #For training steps
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
            # print(np.any(np.isnan(batch_x)))

            # Pull a random fixed-length holdout batch of sequences from the holdout patients, for validation
            # holdout_batch,holdout_label=random_batcher(holdout_features,glom_samplings,len(holdout_features),holdout_labels)
            #
            # # Same thing as training batch prep
            # holdout_batch-=feature_mean
            # holdout_batch/=feature_std
            #
            # # holdout_batch[:,:,65] = 0
            # # holdout_batch[:,:,66] = 0
            # # holdout_batch[:,:,67] = 0
            #
            # holdout_batch=np.nan_to_num(holdout_batch)

            # Run the training operation, return the training loss, MSE, and sample predictions
            _,loss, acc,pred = sess.run([train_op,loss_op, val_accuracy,prediction], feed_dict={X: batch_x,
                                                                 Y: batch_y, dropout2: drop})


            if step % display_step == 0 or step == 1:

                # Run the graph for validation, return the validation loss, validation MSE, and a clipped prediction
                # c_p = sess.run([clipped_prediction], feed_dict={X: holdout_batch, dropout2: 1})
                # Print some information to the command line
                print("Step " + str(step) + ", Train Loss= " + \
                      "{:.4f}".format(loss)

                      )

                # save the losses for future analysis!
                with open(loss_txt,'a') as f:
                    f.write(str(step)+':'+str(loss)+'\n')
            # save some checkpoints
            if (step+1) % save_interval == 1:
                saved_path=saver.save(sess, model_path + '/' +str(step)+'_model.ckpt')

        #HOLDOUT STUFF STARTS HERE
        # holdout_f = holdout_features
        # holdout_f_vector=[]
        # for case in range(0,len(holdout_f)):
        #     case_features=np.zeros((len(holdout_f[case]),len(holdout_f[case][0])))
        #     for glom in range(0,len(holdout_f[case])):
        #         case_features[glom,:]=holdout_f[case][glom]
        #     holdout_f_vector.append(case_features)
        #
        # predictions=np.zeros((len(holdout_f_vector),batch_size,2))
        # for step in range(0,len(holdout_f_vector)):
        #     # Get the holdout cases
        #     holdout_idx=step
        #     holdout_batch=np.zeros((1024,len(holdout_f_vector[step]),num_input))
        #     holdout_label=np.zeros((1024,2))
        #
        #     for step_i in range(0,1024):
        #         holdout_batch[step_i,:,:]=holdout_f_vector[step]
        #         np.random.shuffle(holdout_batch[step_i,:,:])
        #         holdout_label[step_i,:]=labels[holdout_idx,:]
        #     # Standardize
        #     holdout_batch-=feature_mean
        #     holdout_batch/=feature_std
        #
        #     holdout_batch=np.nan_to_num(holdout_batch)
        #
        #     c_p = sess.run([soft_prediction], feed_dict={X: holdout_batch, dropout2: 1})
        #     predictions[step,:,:]=np.squeeze(c_p)
        #
        # #predictions are now in the shape of [cases,1024/case,2]
        # print(predictions.shape)
        # predictions = np.squeeze(np.mean(predictions,1))
        # #predictions are now in the shape of [cases,2]
        # print(predictions.shape)
        # exit()
        #
        # for i in range(100):
        #
        #     holdout_batch,holdout_label=random_batcher(holdout_features,glom_samplings,len(holdout_features),holdout_labels)
        #     holdout_batch-=feature_mean
        #     holdout_batch/=feature_std
        #
        #     #Steps: get predictions from a full string of gloms/tubules from the validation set (using the holdout_f_vector notation in predict)
        #     #Then only grab the ones that are over a certain threshold
        #     #Continue with the syntax outlined in the FNN architecture
        #     #Retrain with original set as well
        #
        #     c_p = sess.run([clipped_prediction], feed_dict={X: holdout_batch, dropout2: 1})
        #AND ENDS HERE


        return saved_path
        print("Optimization Finished!")
