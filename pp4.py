 #!/usr/local/bin/python3
###################################
# CS B555 Fall 2019, PP4
#
# Name and user id: Deepali Tolia : dtolia

import copy
import csv
import random
import sys
import time
from itertools import islice
import numpy as np
from matplotlib import pyplot


 # Parse the directory
def parse_dir(dir_name, index):
    zn = []
    wn = []
    doc_num = 0
    dn = []
    v = []
    doc_word_topic = {}
    list_of_words = []
    fname_list = np.genfromtxt(dir_name + "/" + index, delimiter=',', dtype=int)
    index_label = np.genfromtxt(dir_name +"/" + index, delimiter=',', dtype=int)[:,1]
    for i in range(len(fname_list)):
        curr_doc = []
        try:
            file = open((dir_name + "/" + str(fname_list[i][0])), 'r')  # encoding='cp437')
            with file:
                line = file.readline().strip().split(" ")
                for word in line:
                    # Unique words in vocab
                    if word not in v:
                        v.append(word)
                    curr_doc.append(word)
                    # z(n)
                    zn.append(random.randint(0, K-1))
                    # Duplicate words
                    list_of_words.append(word)
                for word in line:
                    # w(n) and d(n)
                    wn.append(v.index(word))
                    dn.append(doc_num)
            # Document-word dictionary
            doc_word_topic[doc_num] = curr_doc
            doc_num += 1
        except:
            print("Can't open file : ", str(fname_list[i][0]))
    return wn, dn, np.asarray(zn), v, doc_word_topic, list_of_words, index_label

# Initialize cd, ct, and p matrices
def matrix_initialize():
    cd = np.zeros((len(doc_word_topic), K))
    ct = np.zeros((K, len(v)))
    p = np.zeros((K))
    for n in range(len(list_of_words)):
        cd[dn[n]][zn[n]] += 1
        ct[zn[n]][wn[n]] += 1
    return cd, ct, p

# Collapsed gibbs sampler for LDA
def collapsed_gibbs(p):
    beta = 0.01
    alpha = 5/K
    part1 = len(v)*beta
    part2 = K*alpha
    for i in range(500):
        for n in range(len(list_of_words)):
            word = wn[phi_n[n]]
            topic = zn[phi_n[n]]
            doc = dn[phi_n[n]]
            cd[doc][topic] -= 1
            ct[topic][word] -= 1
            for k in range(K):
                numerator = ((ct[k][word] + beta) * (cd[doc][k] + alpha))
                denominator = ((part1 + np.sum(ct[k,:])) * (part2 + np.sum(cd[doc,:])))
                p[k] = numerator / denominator
            # Normalize p
            p =np.divide(p, np.sum(p))
            # Random choice of p for topic
            topic = np.random.choice(range(0,K),p=p)
            zn[phi_n[n]] = topic
            cd[doc][topic] += 1
            ct[topic][word] += 1
    return zn, ct, cd

# Divide data in training and test set
def data_partition(phi, label):
    index_list = [*range(0, len(phi), 1)]
    random.shuffle(index_list)
    arr_length = round(len(phi) / 3)
    if arr_length < 1:
        one_third = index_list[0:1]
        two_third = index_list[1:]
    else:
        one_third = index_list[0:arr_length]
        two_third = index_list[arr_length:]
    ptest = np.take(phi, one_third, axis=0)
    ptrain = np.take(phi, two_third, axis=0)
    ttrain = np.take(label, two_third, axis=0)
    ttest = np.take(label, one_third, axis=0)
    return ptrain, ttrain, ptest, ttest


# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Calculate MAP solution wmap
def wmap_calc(alpha,pdata, tdata):
    row, col = pdata.shape
    w_curr = np.zeros(col)
    calc1 = np.dot(-alpha, np.identity(col))
    for i in range(100):
        y = sigmoid(np.inner(w_curr.T, pdata))
        r = np.subtract(y, np.square(y))
        R = np.diag(r)
        d = np.subtract(tdata, y)
        calc2 = np.dot(np.dot(pdata.T, R),pdata)
        prod1 = np.linalg.inv(np.subtract(calc1, calc2))
        prod2 = np.subtract(np.dot(pdata.T, d), np.dot(alpha, w_curr))
        w_next = np.subtract(w_curr, np.dot(prod1, prod2))
        if i != 0:
            w = np.linalg.norm(w_next - w_curr) / np.linalg.norm(w_curr)
            if w < 0.001:
                return w_next
        w_curr = w_next
    return w_curr


# Calculate ytest
def ytest_calc(wmap, ptest):
    return sigmoid(np.inner(wmap.T, ptest))

# Predicting label values
def prediction(ytest):
    predict = []
    for val in ytest:
        if val < 0.5:
            predict.append(0)
        elif val >= 0.5:
            predict.append(1)
    return predict


# Generalised linear model
def GLM(ptrain, ttrain, ptest, ttest):
    arr_length = len(ptrain)
    tarr_length = len(ttrain)
    error_list = []
    mean_list = []
    sd_list = []
    end_time = []
    for i in range(1, 11):
        data_per = i / 10
        pdata = ptrain[0:round(data_per * arr_length)]
        tdata = ttrain[0:round(data_per * tarr_length)]
        mean = np.mean(tdata)
        sd = np.std(tdata)
        mean_list.append(mean)
        sd_list.append(sd)
        alpha = 0.01
        start_time = time.time()
        wmap = wmap_calc(alpha,pdata, tdata)
        end_time.append(time.time() - start_time)
        ytest = ytest_calc(wmap, ptest)
        predict = prediction(ytest)
        error = np.abs(np.subtract(predict, ttest))
        error_list.append(np.mean(error))
    return mean_list, error_list,np.average(end_time)

# Logistic regression fpr classification
def logistic_regression(phi, label):
    error_step = []
    time_i = []
    start_time = time.time()
    for i in range(1, 31):
        ptrain, ttrain, ptest, ttest = data_partition(phi, label)
        start_time_i = time.time()
        mean, error_value, wmap_time = GLM(ptrain, ttrain, ptest, ttest)
        error_step.append(error_value)
        time_i.append((time.time() - start_time_i))
    print("--- %s seconds ---" % (time.time() - start_time))
    np.reshape(error_step, (30, 10))
    avg = np.mean(np.array(error_step), axis=0)
    mean_for_both.append(avg)
    # Calculate performance
    performance = []
    for mean_error in avg:
        performance.append(1-mean_error)
    performance_for_both.append(performance)
    std = np.std(np.array(error_step), axis=0)
    std_for_both.append(std)

# Main Function
if __name__ == "__main__":
    start = time.time()
    sub_dir_name = sys.argv[1]
    index = sys.argv[2]
    dir_name = "pp4data/" + sub_dir_name
    K = 20
    wn, dn, zn, v, doc_word_topic, list_of_words, label = parse_dir(dir_name, index)
    phi_n = np.random.permutation(len(list_of_words))
    cd, ct, p = matrix_initialize()
    zn, ct, cd = collapsed_gibbs(p)
    topics = {}
    for doc in range(ct.shape[0]):
        index = np.argsort(ct[doc])
        frequent = list(islice(reversed(index), 0, 5))
        for word_index in frequent:
            if doc in topics:
                topics[doc] += [v[int(word_index)]]
            else:
                topics[doc] = [v[int(word_index)]]
    with open('topicwords.csv', 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in topics.items():
            writer.writerow([key, value])
    print("Time required for Gibbs sampling:\n--- %s seconds ---" % (time.time() - start))

    #alpha = 0.01
    alpha = 5/K
    topic_represent = copy.deepcopy(cd)
    for doc in range(topic_represent.shape[0]):
        for k in range(K):
            topic_represent[doc][k] = (cd[doc][k] + alpha) / ((K*alpha)+np.sum(cd[doc,:]))


    bow = np.zeros((len(doc_word_topic), len(v)))
    for doc in doc_word_topic:
        for word in doc_word_topic[doc]:
            bow[doc][v.index(word)] += 1

    mean_for_both = []
    performance_for_both = []
    std_for_both = []
    print("Time required for Topic representation:")
    logistic_regression(topic_represent, label)
    print("Time required for Bag of words:")
    logistic_regression(bow, label)

    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    pyplot.errorbar(x, performance_for_both[0], std_for_both[0], linewidth=1.0, label="Topic representation", color='blue', ecolor='red')
    pyplot.errorbar(x, performance_for_both[1], std_for_both[1], linewidth=1.0, label="Bag of Words", color='green', ecolor='orange')
    pyplot.xlabel("Data Size")
    pyplot.ylabel("Performance")
    pyplot.legend()
    pyplot.show()
