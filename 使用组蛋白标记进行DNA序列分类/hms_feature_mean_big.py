import os 
import numpy as np 
import re
csv_dir='3csv_dnase'
csv_list=os.listdir(csv_dir)
feature=[]
for line in csv_list:
    file_path=csv_dir+"\\"+line
    ftbs=np.genfromtxt(file_path,delimiter=',',missing_values='NA',filling_values='0')
    ftbs=np.array(ftbs)
    mean_feature=np.mean(ftbs,axis=1)
    #sum_feature=np.sum(ftbs,axis=1)
    #max_feature=np.max(ftbs,axis=1)
    feature.append(mean_feature)
    #feature.append(sum_feature)
    #feature.append(max_feature)
final_feature=np.vstack(feature)
final_feature=final_feature.T
np.savetxt('GM12878_com_300bp_hms_mean.csv',final_feature,delimiter=',',fmt="%.4f")



