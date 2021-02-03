
# coding: utf-8

# In[ ]:


#AthMethPre Xiang, S., et al., AthMethPre: a web server for the prediction and query of mRNA m 6 A sites in Arabidopsis thaliana. Molecular BioSystems, 2016. 12(11): p. 3333-3337.
import numpy as np
import pandas as pd
import sys
import itertools
def read_fasta_file(path):
    '''
    used for load fasta data and transformd into numpy.array format
    '''
    fh=open(m6a_benchmark_dataset)
    seq=[]
    for line in fh:
        if line.startswith('>'):
            continue
        else:
            seq.append(line.replace('\n','').replace('\r',''))
    fh.close()
    matrix_data=np.array([list(e) for e in seq])
    return matrix_data
def AthMethPre_extract_one_line(data_line):
    '''
    extract features from one line, such as one m6A sample
    '''
    A=[0,0,0,1]
    T=[0,0,1,0]
    C=[0,1,0,0]
    G=[1,0,0,0]
    N=[0,0,0,0]
    feature_representation={"A":A,"T":T,"C":C,"G":G,"N":N}
    beginning=0
    end=len(data_line)-1
    one_line_feature=[]
    alphabet='ATCG'
    matrix_two=["".join(e) for e in itertools.product(alphabet, repeat=2)] # AA AU AC AG UU UC ...
    matrix_three=["".join(e) for e in itertools.product(alphabet, repeat=3)]# AAA AAU AAC ...
    matrix_four=["".join(e) for e in itertools.product(alphabet, repeat=4)]# AAAA AAAU AAAC ...
    #matrix_five=["".join(e) for e in itertools.product(alphabet, repeat=5)]# AAAA AAAU AAAC ...
    #print matrix_two
    #print len(matrix_two)
    #print len(matrix_three)
    #print len(matrix_four)
    print len(data_line)
    feature_two=np.zeros(len(matrix_two))
    feature_three=np.zeros(len(matrix_three))
    feature_four=np.zeros(len(matrix_four))
    #feature_five=np.zeros(1024)
    two=[]
    three=[]
    four=[]
    for index,data in enumerate(data_line):
        '''if index==beginning or index==end:
            one_line_feature.extend(feature_representation["N"]) 
            print  len(one_line_feature)
        elif data in feature_representation.keys():
            one_line_feature.extend(feature_representation[data])
            #print one_line_feature
            print len(one_line_feature)'''
        if "".join(data_line[index:(index+2)]) in matrix_two and index <= end-1:
            feature_two[matrix_two.index("".join(data_line[index:(index+2)]))]+=1
            two.append(matrix_two.index("".join(data_line[index:(index+2)])))
            
        if "".join(data_line[index:(index+3)]) in matrix_three and index <= end-2:
            feature_three[matrix_three.index("".join(data_line[index:(index+3)]))]+=1
            three.append(matrix_three.index("".join(data_line[index:(index+3)])))
        if "".join(data_line[index:(index+4)]) in matrix_four and index <=end-3:
            feature_four[matrix_four.index("".join(data_line[index:(index+4)]))]+=1
            four.append(matrix_four.index("".join(data_line[index:(index+4)])))
        #if "".join(data_line[index:(index+5)]) in matrix_five and index <=end-4:
            #feature_five[matrix_five.index("".join(data_line[index:(index+5)]))]+=1
    #print len(one_line_feature)        
    #print max(two)
    #print max(three) 
    #print max(four)   
    sum_two=np.sum(feature_two)
    #print len(feature_two)
    #print feature_two
    #print feature_three
    #print feature_four
    #print sum_two
    sum_three=np.sum(feature_three)
    #print sum_three
    sum_four=np.sum(feature_four)
    #print sum_four
    #sum_five=np.sum(feature_five)
    for i in feature_two:
        one_line_feature.append(i/sum_two)
    #print len(one_line_feature)
    for i in feature_three:
        one_line_feature.append(i/sum_three)
    #print len(one_line_feature)
    for i in feature_four:
        one_line_feature.append(i/sum_four)
    #print len(one_line_feature)
    #one_line_feature.extend(feature_five/sum_five)
    print len(one_line_feature)
    return one_line_feature

def AthMethPre_feature_extraction(matrix_data):
    final_feature_matrix=[AthMethPre_extract_one_line(e) for e in matrix_data]
    return final_feature_matrix


m6a_benchmark_dataset=sys.argv[1]
matrix_data=read_fasta_file(m6a_benchmark_dataset)
print matrix_data
final_feature_matrix=AthMethPre_feature_extraction(matrix_data)
print(np.array(final_feature_matrix).shape)
pd.DataFrame(final_feature_matrix).to_csv(sys.argv[2],header=None,index=False)



