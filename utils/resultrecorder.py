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

5.  zero_shot(test_objects, agent1, agent2, n_samples = 20)
    Conduct zero-shot test. Given the list of non-seen objects and two agents,
    calcuate the success rate of playing game.
    
@author: xiayezi
"""

import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt

from data_prepare.data import *
from models.obverter import *
#from train import args


# concept1, concept2 = ('blue','box'), ('red','circle')
# concept1, concept2 = ('blue','circle'), ('red','circle')
def hanmming_dist(concept1, concept2):
    '''
        Calculate the hanmming distance of two concepts.
        The input concepts should be tuple, e.g. ('red','box')
        We require concept1 and 2 have the same number of attributes,
        i.e., len(concept1)==len(concept2)
    '''
    acc_dist = 0
    for i in range(len(concept1)):
        if concept1[i]!=concept2[i]:
            acc_dist += 1
    
    return acc_dist
    
    
    

# str1, str2 = 'horse', 'rose'
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
def train_recorder(idx_round, r_accuracy, r_loss, r_msglen, lis_msg, spk_msg, rpt_gap = 10, folder='test'):
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
                         so we can calculate Edit distance between the msg
                         for each object
            rpt_gap: the gap between two records to the terminal
        Output: 
            avg_dist of lis_msg and spk_msg
    '''
    path = 'runs/' + folder
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
    
    sum_dist, cnt = 0, 0
    for kys in lis_msg.keys():
        sum_dist += edit_dist(lis_msg[kys], spk_msg[kys])
        cnt += 1
    
    avg_dist = sum_dist/cnt
  
    with open(path+'/train_recorder.txt','a') as f:
        f.write('Round %d: Acc: %.5f\t Loss: %.5f\t Msg_len: %.3f\t Edit_Dist: %d \n' \
                %(idx_round, r_accuracy, r_loss, r_msglen, avg_dist))
    
    if int(idx_round) % int(rpt_gap) == 1:
        print('Round %d: Acc: %.5f\t Loss: %.5f\t Msg_len: %.3f\t Edit_Dist: %d \n' \
                %(idx_round, r_accuracy, r_loss, r_msglen, avg_dist))
        
    return avg_dist

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
            if n_samples == 1:
                nn = 1
            else:
                nn = random.randint(0, n_samples-1)
                
            img = images_dict[color, shape, nn] / 256
            
            act, _, _ = msg_gen_decoder(agent, img, max_sentence_len, vocab_size, device)
            msg[color, shape].append(act)
     
    return msg 
            
#spk_msg_all = {}      
#spk_msg_all['green','box'] = ['aaa', 'bbb', 'ccc']       
#spk_msg_all['blue','box'] = ['aaa', 'bc', 'bc'] 

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

def consist_recorder(msg_all, idx_round, name='consist', folder='test'):
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
    avg_cons = sum(consistants.values())/len(consistants.values())
    
    path = 'runs/' + folder +'/msg'
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)  
    
    name_id = name+'-%d'%idx_round
    with open(path+'/'+name_id+'.txt','a') as f:
        for ky in consistants.keys():
            f.write(str(ky))
            f.write(':\t' + '%.4f'%(consistants[ky]) + '\n')   
        f.write('The consistancy in round %d is %.4f\n'%(idx_round, avg_cons))   
    print('The average consistancy in round %d is %.4f'%(idx_round, avg_cons))


'''
msg = {}   
# === Degenerate =====   
msg['green','box'] = ['aa']       
msg['blue','box'] = ['aa'] 
msg['green','circle'] = ['aa']       
msg['blue','circle'] = ['aa'] 
compos_cal(msg)
# === Holistic =====   
msg['green','box'] = ['aa']       
msg['blue','box'] = ['bb'] 
msg['green','circle'] = ['ba']       
msg['blue','circle'] = ['ab']
compos_cal(msg)
# === Compositional == 
msg['green','box'] = ['aa']       
msg['blue','box'] = ['ba'] 
msg['green','circle'] = ['ab']       
msg['blue','circle'] = ['bb'] 
compos_cal(msg) 
'''


def compos_cal(msg):
    '''
        Calculate the compositionalities using metric mentioned in:
        Language as an evolutionary system -- Appendix A (Kirby 2005)
        Input:
            msg: dictionary for a all possible {(color, shape):msg}
        Output:
            corr_pearson:   person correlation
            corr_spearma:  spearman correlation
    '''
    keys_list = list(msg.keys())
    concept_pairs = []
    message_pairs = []
    # ===== Form concepts and message pairs ========
    for i in range(len(keys_list)):
        #for j in range(i+1, len(keys_list)):
        for j in range(len(keys_list)):
            tmp1 = (keys_list[i],keys_list[j])
            concept_pairs.append((keys_list[i],keys_list[j]))
            tmp2 = (msg[tmp1[0]][0],msg[tmp1[1]][0])
            message_pairs.append(tmp2)
            
    # ===== Calculate distant for these pairs ======
    concept_HD = []
    message_ED = []
    for i in range(len(concept_pairs)):
        concept1, concept2 = concept_pairs[i]
        message1, message2 = message_pairs[i]
        concept_HD.append(hanmming_dist(concept1, concept2))
        message_ED.append(edit_dist(message1, message2))
    
    if np.sum(message_ED)==0:
        message_ED = np.asarray(message_ED)+0.1
        message_ED[-1] -= 0.01
 
    dist_table = pd.DataFrame({'HD':np.asarray(concept_HD),
                               'ED':np.asarray(message_ED)})    
    corr_pearson = dist_table.corr()['ED']['HD']
    corr_spearma = dist_table.corr('spearman')['ED']['HD']
     
    return corr_pearson, corr_spearma

'''
msg = {}   
# === Degenerate =====   
msg['green','box'] = ['aa']       
msg['blue','box'] = ['aa'] 
msg['green','circle'] = ['aa']       
msg['blue','circle'] = ['bb'] 
advanced_compos_cal(msg)
# === Holistic =====   
msg['green','box'] = ['aa']       
msg['blue','box'] = ['bb'] 
msg['green','circle'] = ['ba']       
msg['blue','circle'] = ['ab']
advanced_compos_cal(msg)
# === Compositional == 
msg['green','box'] = ['aa']       
msg['blue','box'] = ['ba'] 
msg['green','circle'] = ['ab']       
msg['blue','circle'] = ['bb'] 
advanced_compos_cal(msg) 
'''

def advanced_compos_cal(msg):
    '''
        Calculate the compositionalities using metric mentioned in:
        Language as an evolutionary system -- Appendix A (Kirby 2005)
        We need shuffle the mapping between concepts and messages, then see the
        discribution of correlation coefficient.
        Input:
            msg: dictionary for a all possible {(color, shape):msg}
    '''
    iterations = 10000
    true_pearson, true_spearma = compos_cal(msg)
    dist_pearson, dist_spearma = [], []
    shuffle_msg = {}
    for i in range(iterations):
        value_list = list(msg.values())
        key_list = list(msg.keys())
        for j in range(len(key_list)):
            ridx = np.random.randint(0,len(key_list))
            shuffle_msg[key_list[j]] = value_list[ridx]
        
        tmp1, tmp2 = compos_cal(shuffle_msg)
        dist_pearson.append(tmp1)
        dist_spearma.append(tmp2)
    ratio_pearson = np.sum(dist_pearson<true_pearson)/iterations
    ratio_spearma = np.sum(dist_spearma<true_spearma)/iterations
    
    return ratio_pearson, ratio_spearma

def msg_recorder(msg, idx_round, name='msg', folder='test'):
    '''
        Help to record the msg (dictionary) to a txt file and npy file.
        The input msg should be a dictionary, similar to:
            {('green','box'): 'bb', ('blue','box'): 'bc'}
        Also calculate the compsitionality and write into the file and terminal
    '''
    comp = compos_cal(msg)
    
    path = 'runs/' + folder +'/msg'
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)   
    
    name_id = name+'-%d'%idx_round
    
    np.save(path+'/'+name_id+'.npy',msg)
    # if want read: a=np.load('test_msg.npy').item()
    
    with open(path+'/'+name_id+'.txt','a') as f:
        for ky in msg.keys():
            f.write(str(ky))
            max_msg,_ = max_list(msg[ky])
            f.write(':\t' + max_msg + '\n')
        f.write('The compositionality of this language is (%4f,%4f)\n'%(comp[0],comp[1]))
    print('The compositionality of this language is (%4f,%4f)'%(comp[0],comp[1]))

def zero_shot(test_objects, agent1, agent2, images_dict, max_sentence_len, 
                   vocab_size, device, path, idx_round, n_samples = 20):
    '''
        Conduct zero-shot test. Given the list of non-seen objects and two agents,
        calcuate the success rate of playing game.    
    '''
    # ========= Prepare the data for test ============
    a1_result = 0
    a2_result = 0
    n_objects = len(test_objects)
    agent1.train(False)
    agent2.train(False)
    
    for color, shape in test_objects:        
        for n in range(n_samples):
            img = images_dict[color, shape, n] / 256            
            
            a1_msg, _, a1_acts = msg_gen_decoder(agent1, img, max_sentence_len, vocab_size, device)
            prob1 = prd_gen_decoder(agent2, img, a1_acts, max_sentence_len, vocab_size, device)
            
            a2_msg, _, a2_acts = msg_gen_decoder(agent2, img, max_sentence_len, vocab_size, device)
            prob2 = prd_gen_decoder(agent1, img, a2_acts, max_sentence_len, vocab_size, device)            
            
            if prob1 > 0.95: a1_result += 1
            if prob2 > 0.95: a2_result += 1

    with open(path+'/zero_shot.txt','a') as f: 
        f.write('Acc of round %d is: (%4f,%4f)\n'%(idx_round,a1_result/(n_samples*n_objects),
                                      a2_result/(n_samples*n_objects)))

            
    return a1_result/(n_samples*n_objects), a2_result/(n_samples*n_objects)









