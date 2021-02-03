'''
Author: your name
Date: 2020-03-07 19:33:56
LastEditTime: 2020-11-08 12:34:17
LastEditors: Please set LastEditors
Description: In User Settings Edit
'''
import pickle

ss_path='GM12878_200bp_pos+neg_fold.txt'
ss_object=open(ss_path,'r')
ss_list=[]
for line in ss_object.readlines():
    ss_list.append(line.strip())
num_line=len(ss_list)
ss_dict=dict()
for i in range(0,num_line,3):

    key=ss_list[i+1].replace('U','T')
    temp_value=ss_list[i+2].split(' (')
    value=[temp_value[0],float(temp_value[1][:-1].strip())]
    ss_dict[key]=value
print(ss_dict)
with open ('GM12878_200bp_fold.pkl','wb') as f:
    pickle.dump(ss_dict,f)