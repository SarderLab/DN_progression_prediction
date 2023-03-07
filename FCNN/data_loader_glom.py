import numpy as np
import random
import pandas as pd
def import_data(txt_file):
    f = open(txt_file, 'r')
    lines = f.readlines()
    f.close()
    lines = np.array(lines)
    feats=[]
    features=[]
    for line in lines:

        if "---" in line:
            feats.append(features)
            features=[]
            continue


        featurevals=line.split(',')
        f_=[]
        for f_val in featurevals[:-1]:
            f_.append(float(f_val))
        features.append(f_)
    return feats

def import_labels(lbl_file,idx):
    f = pd.read_csv(lbl_file,header=None).to_numpy()
    f = np.transpose(f)
    f = f[idx,:]
    # labels=[]
    # labels=np.zeros(f.shape[1])
    # for l in range(len(labels)):
        # labels[l]=f[l]
    return f

def random_batcher(feats,time_step,batch_size,labels):

    num_cases=len(feats)
    num_features=len(feats[0][1])
    batched_data=np.zeros( ( batch_size,time_step,len(feats[0][1]) ) )
    batched_labels=np.zeros((batch_size,2))
    for i in range(0,batch_size):
        random_idx=random.randrange(num_cases)
        random_case=feats[random_idx]
        random_label=labels[random_idx]

        batched_labels[i]=random_label
        for j in range(0,time_step):
            random_glom = random.choice(random_case)

            for v in range(0,num_features):
                batched_data[i,j,v]=random_glom[v]

    return batched_data,batched_labels



def import_indices(filename):

    indices = pd.read_table(filename,delimiter=',').to_numpy()
    usable_indices = []
    add_on = []

    q = 0

    for i in range(indices.shape[0]):
        if indices[i,0] == q + 1:
            q += 1
            usable_indices.append(add_on)
            add_on = []

        add_on.append(indices[i,1])
    usable_indices.append(add_on)



    return usable_indices
