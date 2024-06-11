import pandas as pd
import numpy as np

from utils.data_loader import load_from_tsfile_to_dataframe
from utils.regressor_tools import process_data
#import mlflow
#from tsfeatures import tsfeatures
np.set_printoptions(threshold=np.inf)
#pd.set_option('display.max_rows', None)  
#pd.set_option('display.max_columns', None) 
from utils.personal_utils import *
from compression import *
from utils.compression_algos import *

import os
import matplotlib.pyplot as plt


# Testing of compress_dataset
data_path = "/home/sim/Desktop/TS Extrinsic Regression/data/AppliancesEnergy_TEST.ts"
#data_path = "/home/sim/Desktop/TS Extrinsic Regression/data/BeijingPM25Quality_TEST.ts"


dataset_array = load_dataset(data_path)
dataset_id = os.path.basename(data_path).split('_')[0]


a = dataset_array[0,:,0]
b = cpt_compress(a, 0.1, True)


fig, ax = plt.subplots()
# Plot the arrays
ax.plot(a, label='Array a')
#ax.plot(b, label='Array b')



""" for i in np.arange(0, 1.04, 0.04):
    decompressed_dataset = compress_dataset(dataset_array.copy(), dataset_id, True, "dct", i) 
    print(compute_avg_rmse_of_dataset(dataset_array, decompressed_dataset))
 """
#dataset_array = np.random.randint(0, 10, size=(3, 10, 3))

print("")

for i in np.arange(0, 1.04, 0.04):
    decompressed_dataset = compress_dataset(dataset_array.copy(), dataset_id, False, "cpt", i) 
    print("drop_out value: ",round(i,2),"comp_ratio", calculateCompRatio(dataset_array, decompressed_dataset, True))
    print("drop_out value: ",round(i,2),"comp_ratio", calculateCompRatio(dataset_array, decompressed_dataset, False))

    #print(calculateCompRatio(decompressed_dataset, np.zeros((42,144,24)),True))