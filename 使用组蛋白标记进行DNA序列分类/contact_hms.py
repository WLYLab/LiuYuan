# -*- coding: utf-8 -*-
import os
import numpy as np 
import argparse

def getopt():
    parse=argparse.ArgumentParser()
    parse.add_argument('-p','--path',type=str)
    parse.add_argument('-o','--output',type=str)
    args=parse.parse_args()
    return args
if __name__ == "__main__":
    args=getopt()
    hsm_path=args.path
    file_list=os.listdir(hsm_path)
    data_list=[]
    file_name_list_object=open('file_name_list_3channels+meth.txt','w')
    for file_name in file_list:
        data=np.genfromtxt(hsm_path+'/'+file_name,delimiter=',',dtype=float,missing_values='NA',filling_values=0)
        print(data.shape)
        data=data.reshape(-1,data.shape[1],1)
        data_list.append(data)
        file_name_list_object.write(file_name+'\n')
    file_name_list_object.close()
    data=np.concatenate(data_list,axis=2)
    print (data.shape)
    print(data[1]) 
    np.save('GM12878_com_'+args.output+'_300bp',data)



