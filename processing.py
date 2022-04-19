from sklearn import preprocessing
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from scipy import stats

# SMAP data 불러오기
dataset_dir = "C:/Users/minseok/Desktop/project/data_set/"
SMAP_path_dir = dataset_dir + "SMAP/"
SMAP_label_dir = "anomaly point"
SMAP_train_dir = "data2/train/"
SMAP_test_dir = "data2/test/"

def SMAP():

    #file_list 불러오기
    file_list = os.listdir(SMAP_path_dir + SMAP_label_dir)
    name_list = list()
    for i in file_list:
        name = i.split('.')
        name_list.append(name[0])
    
    # train, test, label 불러오기
    index = 0
    for i in tqdm(name_list):
        train_set = np.load(SMAP_path_dir + SMAP_train_dir + i + '.npy')
        test_set = np.load(SMAP_path_dir + SMAP_test_dir + i + '.npy')
        label_set = np.genfromtxt(SMAP_path_dir + SMAP_label_dir + '/' + i + '.csv', delimiter=',', skip_header=1)
        if train_set.shape[-1] == 55:
            if index == 0:
                train_dataset = train_set
                test_dataset = test_set
                label_dataset = label_set
                index = 1
            else:
                train_dataset = np.concatenate((train_dataset, train_set), axis=0)
                test_dataset = np.concatenate((test_dataset, test_set), axis=0)
                label_dataset = np.concatenate((label_dataset, label_set), axis=0)
    
    return train_dataset, test_dataset, label_dataset

# MSL data 불러오기
def MSL():

    #file_list 불러오기
    file_list = os.listdir(SMAP_path_dir + SMAP_label_dir)
    name_list = list()
    for i in file_list:
        name = i.split('.')
        name_list.append(name[0])
    
    # train, test, label 불러오기
    index = 0
    for i in tqdm(name_list):
        train_set = np.load(SMAP_path_dir + SMAP_train_dir + i + '.npy')
        test_set = np.load(SMAP_path_dir + SMAP_test_dir + i + '.npy')
        label_set = np.genfromtxt(SMAP_path_dir + SMAP_label_dir + '/' + i + '.csv', delimiter=',', skip_header=1)
        if train_set.shape[-1] == 25:
            if index == 0:
                train_dataset = train_set
                test_dataset = test_set
                label_dataset = label_set
                index = 1
            else:
                train_dataset = np.concatenate((train_dataset, train_set), axis=0)
                test_dataset = np.concatenate((test_dataset, test_set), axis=0)
                label_dataset = np.concatenate((label_dataset, label_set), axis=0)
    
    return train_dataset, test_dataset, label_dataset

#SWaT 데이터 불러오기
def SWaT():
    original = pd.read_csv(dataset_dir + "SWaT/SWaT_Dataset_Normal_v0.csv", header=1)
    train_set = original.drop([' Timestamp', 'Normal/Attack'], axis=1)
    train_set = np.array(train_set)
    test_original = pd.read_csv(dataset_dir + "SWaT/SWaT_Dataset_Attack_v0.csv")
    test_set = test_original.drop([' Timestamp', 'Normal/Attack'], axis=1)
    label_set = test_original['Normal/Attack']
    label_list = list()
    for i in label_set:
        if i == 'Normal':
            label_list.append(0)
        else:
            label_list.append(1)
    label_set = np.array(label_list)
    test_set = np.array(test_set)
    return train_set, test_set, label_set

#Server machine data 불러오기
def SMD():
    path_dir = dataset_dir + 'ServerMachineDataset'
    index = 0
    for i in range(1,9):
        if index ==0:
            train = np.genfromtxt(os.path.join(path_dir, 'train', 'machine-1-1.txt'), dtype=np.float32, delimiter=',')
            test = np.genfromtxt(os.path.join(path_dir, 'test', 'machine-1-1.txt'), dtype=np.float32, delimiter=',')
            test_label = np.genfromtxt(os.path.join(path_dir, 'test_label', 'machine-1-1.txt'), dtype=np.float32, delimiter=',')
            index = 1
        else:
            train_ = np.genfromtxt(os.path.join(path_dir, 'train', 'machine-1-' + str(i) + '.txt'), dtype=np.float32, delimiter=',')
            test_ = np.genfromtxt(os.path.join(path_dir, 'test', 'machine-1-' + str(i) + '.txt'), dtype=np.float32, delimiter=',')
            test_label_ = np.genfromtxt(os.path.join(path_dir, 'test_label', 'machine-1-' + str(i) + '.txt'), dtype=np.float32, delimiter=',')
            train = np.concatenate((train,train_))
            test = np.concatenate((test,test_))
            test_label = np.concatenate((test_label,test_label_))
    for i in range(1,10):
        train_ = np.genfromtxt(os.path.join(path_dir, 'train', 'machine-2-' + str(i) + '.txt'), dtype=np.float32, delimiter=',')
        test_ = np.genfromtxt(os.path.join(path_dir, 'test', 'machine-2-' + str(i) + '.txt'), dtype=np.float32, delimiter=',')
        test_label_ = np.genfromtxt(os.path.join(path_dir, 'test_label', 'machine-2-' + str(i) + '.txt'), dtype=np.float32, delimiter=',')
        train = np.concatenate((train,train_))
        test = np.concatenate((test,test_))
        test_label = np.concatenate((test_label,test_label_))
    for i in range(1,12):
        train_ = np.genfromtxt(os.path.join(path_dir, 'train', 'machine-3-' + str(i) + '.txt'), dtype=np.float32, delimiter=',')
        test_ = np.genfromtxt(os.path.join(path_dir, 'test', 'machine-3-' + str(i) + '.txt'), dtype=np.float32, delimiter=',')
        test_label_ = np.genfromtxt(os.path.join(path_dir, 'test_label', 'machine-3-' + str(i) + '.txt'), dtype=np.float32, delimiter=',')
        train = np.concatenate((train,train_))
        test = np.concatenate((test,test_))
        test_label = np.concatenate((test_label,test_label_))
    return train, test, test_label

def window_overlap(input_data, length):
    num = input_data.shape[0] - (length-1)
    dataset = list()
    for i in range(num):
        sample = input_data[i:i+length, :]
        dataset.append(sample)
    return np.array(dataset)

def window_nonoverlap(input_data, length):
    num = input_data.shape[0]//length
    dataset = list()
    for i in range(num):
        sample = input_data[i*length:(i+1)*length, :]
        dataset.append(sample)
    return np.array(dataset)

def label_window(data, length):
    label_set = list()
    num = data.shape[0] // length
    for i in range(num):
        sample = data[i*length:(i+1)*length]
        if np.sum(np.array(sample)) >= 1:
            label_set.append(1)
        else:
            label_set.append(0)
    return np.array(label_set)

def min_max(train,test):
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(train)
    train_set = scaler.transform(train)
    test_set = scaler.transform(test)
    return train_set, test_set

def kde_score(score_set, length):
    critic_extend = list()
    for c in score_set:
        critic_extend.extend(np.repeat(c,length).tolist())
    critic_extend = np.array(critic_extend).reshape((-1, length))
    critic_kde = list()
    num = length + score_set.shape[0] -1
    for i in range(num):
        critic_inter = list()
        for j in range(max(0,i-num+length), min(i+1, length)):
            critic_inter.append(critic_extend[i-j,j])
        
        if len(critic_inter) > 1:
            discr_inter = np.array(critic_inter)
            try:
                critic_kde.append(discr_inter[np.argmax(stats.gaussian_kde(discr_inter)(critic_inter))])
            except np.linalg.LinAlgError:
                critic_kde.append(np.median(discr_inter))
        else:
            critic_kde.append(np.median(np.array(critic_inter)))
    return np.array(critic_kde)

def pred(reconstruct_set):
    length = reconstruct_set.shape[1]
    num = reconstruct_set.shape[1] + reconstruct_set.shape[0] - 1
    prediction = list()
    for i in range(num):
        inter = list()
        for j in range(max(0,i-num+length), min(i+1, length)):
            inter.append(reconstruct_set[i-j,j])
        if inter:
            prediction.append(np.median(np.array(inter),axis=0))
    return np.array(prediction)