import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler
def getopt():
    parse=argparse.ArgumentParser()
    parse.add_argument('-i','--input',type=str)
    parse.add_argument('-o', '--output', type=str)
    args=parse.parse_args()
    return args
args=getopt()
input_file = pd.read_csv(args.input, header=None, index_col=None)
input_file=input_file.values[:,0:]
input_file = pd.DataFrame(input_file).astype(float)
scaler=MinMaxScaler()
output_file=scaler.fit_transform(np.array(input_file))
pd.DataFrame(output_file).to_csv(args.output,header=None,index=False)