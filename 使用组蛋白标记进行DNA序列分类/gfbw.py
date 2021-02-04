import os 
bigwig_path='bigwig'
file_name_list=os.listdir(bigwig_path)
bed_name='GM12878_com.bed'
for file_name in file_name_list:
    com_str='bwtool extract bed'+' '+bed_name+' '+'./'+bigwig_path+'/'+file_name+' '+"GM12878_com"+file_name.split('.')[0]+'.csv'
    os.system(com_str)
