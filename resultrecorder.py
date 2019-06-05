#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:52:48 2019
Result_recorder and analyzer, this file create a class which contains various
kinds of result recorder and generators, they may output the information to the 
terminal, txt files, pdf files or npy files. Here are the functions:

1.  train_recorder(r_accuracy, r_loss, r_msglen, lis_msg, spk_msg)
    This function will record fundamental training information to a txt
    
    1.1 edit_dist(str1, str2)
    Calculate the edit distance of two strings. Insert/delete/replace all +1.    
    
2.  consist_checker(msg_all)
    This function will check whether for the same object (100 samples for each) 
    the speaker or listener will produce the same message. The output is the 
    percentage of most common output
    
    2.1 sample_msg_gen(agent, attributes, images_dict, n_samples)
    Given the agent, the attributes we want to check, and the images_dict, 
    generate n_samples messages. The results are stored in two dictionaries

3.  msg_recorder(msg)
    Record the msg to txt and npy file

    3.1 all_msg_gen(agent, colors, shapes)
    Help to generate messages for all the objects with ONE sample for each

4.  compos_cal(msg, colors, shapes)
    Calculate the compositionalities using metric mentioned in:
        Language as an evolutionary system -- Appendix A (Kirby 2005)
    
@author: xiayezi
"""

import numpy as np
import os
from data import *
from obverter import *
#from train import args

def edit_dist(str1, str2):
    '''
        Calculate the edit distance of two strings.
        Insert/delete/replace all cause 1.
    '''
    len1, len2 = len(str1), len(str2)
    DM = [0]
    for i in range(len1):
        DM.append(i+1)
        
    for j in range(len2):
        DM_new=[j+1]
        for i in range(len1):
            tmp = 0 if str1[i]==str2[j] else 1
            new = min(DM[i+1]+1, DM_new[i]+1, DM[i]+tmp)
            DM_new.append(new)
        DM = DM_new
        
    return DM[-1]

#train_recorder(1, 0.5, 0.4 ,0,'intention','execution',rpt_gap = 10)
def train_recorder(idx_round, r_accuracy, r_loss, r_msglen, lis_msg, spk_msg, rpt_gap = 10):
    '''
        This function will record fundamental training information to a txt file. 
        Usually we record these each round.
        
        Input: 
            idx_round: the index of round
            r_accuracy: #correctly predicted/#total pairs in this round
            r_loss: cross entropy between true label and predictions, one round
            r_msglen: average message length, note that we can tune the cutting
                      probability in obverter.py to change the expected msg_len
            lis/spk_msg: message of listener and speaker, they should be dicts
                         so we can calculate Jaccard Similarity between the msg
                         for each object
            rpt_gap: the gap between two records to the terminal
    '''
    path = 'runs/' + 'test'
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
    
    Edit_dist = edit_dist(lis_msg, spk_msg)
  
    with open(path+'/train_recorder.txt','a') as f:
        f.write('Round %d: Acc: %.5f\t Loss: %.5f\t Msg_len: %d\t Edit_Dist: %d \n' \
                %(idx_round, r_accuracy, r_loss, r_msglen, Edit_dist))
    
    if int(idx_round) % int(rpt_gap) == 1:
        print('Round %d: Acc: %.5f\t Loss: %.5f\t Msg_len: %d\t Edit_Dist: %d \n' \
                %(idx_round, r_accuracy, r_loss, r_msglen, Edit_dist))


def sample_msg_gen(agent, attributes, images_dict, max_sentence_len, 
                   vocab_size, device, n_samples=1):
    '''
        Input:
            agent: one ConvModel, with the trained networks
            attributes: [(color, shape)], a list of tuples, we generate msg 
                for n_samples for each attribute
            images_dict: data contains all the samples
            n_samples: # samples of this object, note that when generating the 
                data, different samples have different location/rotation
        Output:
            msg: messages produced by the agent use dictionary to store them      
    '''
    msg = {}
    
    for color, shape in attributes:
        msg[color, shape] = []
        # ------ Randomly select one sample to generate ------------
        for n in range(n_samples):
            # ------- If n_samples=1, select randomly, otherwise by order
            nn = random.randint(0, n-1) if n_samples==1 else n
            img = images_dict[color, shape, nn] / 256
            
            act, _ = msg_gen_decoder(agent, img, max_sentence_len, vocab_size, device)
            msg[color, shape].append(spk_act)
     
    return msg 
            
spk_msg_all = {}      
spk_msg_all['green','box'] = ['aaa', 'bbb', 'ccc']       
spk_msg_all['blue','box'] = ['aaa', 'bc', 'bc'] 

def max_list(lt):
    '''
        Find the element that occurs most in one list
    '''
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)   
    return max_str, lt.count(max_str)

def consist_checker(msg_all):
    '''
        This function will check whether for the same object (100 samples each) 
        the speaker and listener will produce the same message. The result will 
        stored in a txt.  
        Input:
            msg_all: a dictionary like 
                spk_msg_all['green','box'] = ['aaa', 'bc', 'bc']
        This function may output the ratio of most common message to the number
        of sample, under each keys. For the above example, the output should
        contain:
                consistants['green','box'] = 2/3
    '''
    consistants = {}
    for ky in msg_all.keys():
        max_str, max_cnt = max_list(msg_all[ky])
        consistants[ky] = max_cnt/len(msg_all[ky])
    
    return consistants

    










