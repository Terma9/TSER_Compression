import numpy as np
import gzip

from utils.compression_algos import *

# define global dictionary
dataset_params = {
    'AppliancesEnergy':          {'block_size': 1008, 'num_dim': 24, 'len_ts': 144},
    'BeijingPM25Quality':        {'block_size': 1008, 'num_dim': 9, 'len_ts': 24},
    'IEEEPPG':                   {'block_size': 1000, 'num_dim': 5, 'len_ts': 1000},
    'FloodModeling1':           {'block_size': 1064, 'num_dim': 1, 'len_ts': 266},



    'Covid3Month':               {'block_size': 1008, 'num_dim': 1, 'len_ts': 84}, 
    'BenzeneConcentration':      {'block_size': 960, 'num_dim': 8, 'len_ts': 240},
    'NewsTitleSentiment':        {'block_size': 1008, 'num_dim': 3, 'len_ts': 144},
    'HouseholdPowerConsumption1':{'block_size': 1440, 'num_dim': 5, 'len_ts': 1440},

    'HouseholdPowerConsumption2':{'block_size': 1440, 'num_dim': 5, 'len_ts': 1440},
    'FloodModeling2':            {'block_size': 1064, 'num_dim': 1, 'len_ts': 266},
    'BeijingPM10Quality':        {'block_size': 1008, 'num_dim': 9, 'len_ts': 24},
    
}

# Compression of dataset-array applied on blocks around size 1000.(Exact size depends on dataset to not cut time series in parts.)
# Last block is compressed together with the previous block if it is smaller than block_size/2.
# Either returns the array with the coefficients of compression(for load_and_prepare_dataset)
# or returns the array with decompressed values.(For calculating compression size.) Depending on andDecompress.

# Default Quantization Level is 100, for dct best result was 0, for others I still have to check!
# Default Wavelet is db4, for Flood and Covid switch to haar -> Change automatically!
# Default level is 99 -> which gets translated to max_level in dwt_compress
# (I keep explicit params, to later reprocude the Studies)

def compress_dataset(dataset_array, dataset_id, andDecompress:bool, compression_type, compression_param, level = 99 , wavelet = "db4", quantization_level = 100):

    # Retrieve dataset parameters
    num_dim = dataset_params[dataset_id]['num_dim']
    len_ts = dataset_params[dataset_id]['len_ts'] 
    block_size = dataset_params[dataset_id]['block_size']


    # Flatten the dataset array, put one matrix under the other: (num_dp, len_ts, dim) -> (len_flat_dim, dim)
    array_flatdim = dataset_array.reshape(-1, num_dim).copy()

    # Adapt return type to complex128 if dft is used and we don't want decompress directly
    if compression_type == 'dft' and andDecompress == False:
       array_flatdim = array_flatdim.astype(np.complex128)


    # !!!! Following is the study result block, uncomment after used studies! -> with that i gainedd my data on which the autoML run, so better keep it! -> but new wavelet study came to same result


    # Change wavelet regarding studies results
    if dataset_id in ["FloodModeling1", "Covid3Month", "HouseholdPowerConsumption", "NewsTitleSentiment"]:
        wavelet = "haar"
    else:
        wavelet = "db4"

    
    if dataset_id == "NewsTitleSentiment":
        level = 1
    else: 
        level = pywt.dwt_max_level(dataset_params[dataset_id]['block_size'], wavelet)


    # Change Quantisation level after studies: Level 1 if below 0.95 dropout. Level 1 for higher than 0.9 dropout.
    #if compression_param <= 0.95:
    #    quantization_level = 1
    #else:
    #    quantization_level = 0



    # 0 is best quantization level for dct! for the others I don't know!
    #if compression_type == 'dct':
    #    quantization_level = 0

    # !!!!!


    # Length of all the time series in one dimension -> Assumption, all dimensions have the same length
    len_flat_dim = array_flatdim.shape[0]


    # Just create an array pointer for later!
    array_flatdim_coeff = array_flatdim

    
    # If the last block is smaller than block_size/2, we will transform it togehter with the previous block.
    # If len_last_block zero, then we don't need to do anything. Last transform fits perfectly.
    # could simply put this in the first if of the loop -> cleaner code, little bit more calculation 
    last_transfrom_large = False
    len_last_block = len_flat_dim % block_size
    if  len_last_block != 0 and len_last_block < block_size/2:
        last_transfrom_large = True

    # Create blocks and so transforms for each dimension individually. Since different dimensions have different ts-curves. The transform would transform worse, if we mix dimensions for transforming.
    for i  in range(num_dim):
        end_loop = False
        start_idx = 0
        end_idx = block_size

        coeff_list = []

        while not end_loop:

            # We check if we are in the block before the last block. If yes we check if we need to transform the last block together with the previous block.
            # If both yes we merge together and end loop.
            if (end_idx + 1 * block_size) > len_flat_dim and last_transfrom_large:
                end_idx = len_flat_dim
                end_loop = True

        
            # If the last block (is larger than block_size/2, only then can condition be true) and is larger than len_flat_dim, we need to adjust the end_idx. 
            # We do >= to end loop, also when it fits perfectly. First assignment is then redundant. But is very important to end loop!
            if end_idx >= len_flat_dim:
                end_idx = len_flat_dim
                end_loop = True

            coeff_list += apply_compression(array_flatdim[start_idx:end_idx, i], compression_type, compression_param, andDecompress, level, wavelet, quantization_level).tolist()
    

            # Adjust indices for next block
            start_idx = end_idx
            end_idx += block_size
        
        # Be aware of the line indents
        if i == 0:
            array_flatdim_coeff = np.zeros((len(coeff_list),num_dim))
        
        array_flatdim_coeff[:,i] = np.array(coeff_list)
        


    # Depending on andDecompress return the compressed array(coefficients) or the decompressed array(Approximation).
    if andDecompress == False:
        # Return array in format (len_flat_dim, dim).
        return array_flatdim_coeff
    else: 
        #Bring back from (len_flat_dim, dim) to (num_dp, len_ts, dim). Remember reshape always only returns views!
        dataset_array_compressed = array_flatdim_coeff.reshape(-1,len_ts, num_dim)
        return dataset_array_compressed



# Maybe add wrapper to load data directly from path!

# Tested internal logic! And one simple test case!
# dataset_array.shape -> (num_dp, len_ts, num_dim), array_flatdim_comp.shape -> (len_flat_dim, num_dim)
def calculateCompRatio(dataset_array, array_flatdim_comp):


    # Also gzip the raw_data
    dataset_array_gzipd = gzip.compress(dataset_array.tobytes(), compresslevel=9)

    #print("Number of elements of Coeff-Array", array_flatdim_comp.size))
    #print("Number of Zersos", np.sum(array_flatdim_comp.flatten() == 0)
    
    
    # Compress with gzip
    # Save it as dim after dim, to keep ts and blocks together saved. 
    array_flatdim_1d = array_flatdim_comp.reshape(-1, order='F') 
    
    # Transform to pyton bytes object. It flattens the array row after row (and slice after slice). (I flattened already manually.)
    # Then gives us a byte-string with each element in byte format.(Just the numbers of the array one after another. No metadata.)
    compressed_data_bytestring = array_flatdim_1d.tobytes()

    
    # Compress the byte-string with gzip. -> Input is byte string and then returns a byte-string. ->> Test more by yourself!
    compressed_data_gzipd = gzip.compress(compressed_data_bytestring, compresslevel=9)


    # .nbytes simply returns: np.prod(a.shape) * a.itemsize -> simply how many bytes are filled in the array. No metadata. Only counts the byte of the elements.
    num_bytes_raw = dataset_array.nbytes

    # len(bytestring) returns number of bytes in the bytestring! No metadata. No nothing!
    return len(dataset_array_gzipd) / len(compressed_data_gzipd)

























 #Not sure if it makes difference, but keeps the idea-logic!
""" 
# Decompress from File!
# Return array now, but adjust later to save to file!
def decompress_dataset(array_flatdim_comp = None, dataset_id = "", compression_type=None, compression_param=None, from_file = None):
    
    # Retrieve dataset parameters
    num_dim = dataset_params[dataset_id]['num_dim']
    len_ts = dataset_params[dataset_id]['len_ts']
    #block_size = dataset_params[dataset_id]['block_size'] -> not needed, only for compression
 
    # Load compressed data
    if from_file != None:
        array_totally_flat = np.fromfile(from_file)

        # Reshape the data. Works as intended: Cuts array after num_dim, then start new rum with next num_dim block and so on.
        array_flatdim_comp = array_totally_flat.reshape(-1, num_dim)

    
    # Calculat the number of datapoints in the dataset. Saver than passing it as an argument. Should always be whole number./integer.
    num_dp = array_flatdim_comp.shape[0] / len_ts

    decomp_dataset_array = np.zeros((num_dp,len_ts, num_dim))

    # have to go over blocks!
    # need to remember the size of last block!


    # Go for each dim over all ts and decompress
    for i in range(num_dim):
        for j in range(num_dp):
            start_idx = 0
            end_idx = len_ts
            decomp_dataset_array[j,:,i] = apply_decompression(array_flatdim_comp[start_idx:end_idx,i], compression_type, compression_param)

            start_idx = end_idx
            end_idx += len_ts

    
    return decomp_dataset_array
 """

    
   