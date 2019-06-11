#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:49:07 2019

@author: xiayezi
"""

import numpy as np
import os
import random

import sys
sys.path.append("..")
from utils.resultrecorder import *

msg={}
msg['green','box'] = ['aa']       
msg['blue','box'] = ['aa'] 
msg['green','circle'] = ['aa']       
msg['blue','circle'] = ['aa'] 
compos_cal(msg)

attributes = ['A','B','C','D','E','F']
numbers = ['0','1','2','3','4','5','6','7','8','9']

def count_elements(concept):
    '''
        For each attribute, count the number of its occurance in concept
        Input:
            concept: a list of string
        Output:
            should be a tuple, consistant with the key in msg
    '''
    tmp_out = []
    for i in range(len(attributes)):
        tmp_out.append(str(concept.count(attributes[i])))
        
    return tuple(tmp_out)
        

with open('dev_messages.txt','r') as f:
    concept_list = []
    message_list = []
    cnt = 0    
    for line in f:
        tmp_message = ''
        if line == '---\n': 
            cnt = 0
            continue
        if line[0] in attributes:
            cnt += 1
            tmp_concept = count_elements(line)
        if line[1] in numbers:
            cnt += 1
            tmp = []
            for j in range(len(line)):
                if line[j] in numbers: tmp.append(line[j])
            tmp_message = ''.join(tmp)
        if cnt == 2:
            concept_list.append(tmp_concept)
            message_list.append(tmp_message)
        else:
            continue

msg_numerial = {}

for i in range(len(concept_list)):
    msg_numerial[concept_list[i]] = message_list[i]


msg_numerial_downsmp = {}
msg_numerial_downsmp_perfect = {}
for i in range(1000):
    keys = random.choice(list(msg_numerial.keys()))
    msg_numerial_downsmp[keys] = msg_numerial[keys]
    tmp = []
    for num in keys:
        tmp.append(num)
    msg_numerial_downsmp_perfect[keys] = ''.join(tmp)
    


measure = compos_cal(msg_numerial_downsmp)  
print('The compositionality measure is:',measure)


measure2 = compos_cal(msg_numerial_downsmp_perfect)  
print('The perfect compositionality measure is:',measure2)
'''

for i in range(10):
    keys1 = random.choice(list(msg_numerial_downsmp_perfect.keys()))
    keys2 = random.choice(list(msg_numerial_downsmp_perfect.keys()))
    msg1 = msg_numerial_downsmp_perfect[keys1]
    msg2 = msg_numerial_downsmp_perfect[keys2]
    ED = hanmming_dist(msg1,msg2)
    HD = hanmming_dist(keys1,keys2)
    if (ED!=HD):
        print(keys1,keys2,msg1,msg2)
    print(ED,HD)
'''







          