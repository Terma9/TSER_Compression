import numpy as np
import gzip

from utils.compression_algos import *

# define global dictionary
dataset_params = {
    'AppliancesEnergy': {'block_size': 1008, 'num_dim': 3, 'len_ts': 100},
    'BeijingPM25Quality': {'block_size': 15, 'num_dim': 4, 'len_ts': 120},
    # Add more datasets as needed
}


# To compress to file or from file, add destination or source path to the function call.






# Compression of dataset-array applied on blocks around size 1000.(Exact size depends on dataset to not cut time series in parts.)
# Last block is compressed together with the previous block if it is smaller than block_size/2.
# Either returns the array with the coefficients of compression(for load_and_prepare_dataset)
# or returns the array with decompressed values.(For calculating compression size.) Depending on andDecompress.
# Starts with 



def compress_dataset(dataset_array, dataset_id, andDecompress:bool, compression_type=None, compression_param=None):

    # Retrieve dataset parameters
    num_dim = dataset_params[dataset_id]['num_dim']
    len_ts = dataset_params[dataset_id]['len_ts'] 
    block_size = dataset_params[dataset_id]['block_size']

    # Boolean if last transformation is large or small
    last_transfrom_large = False


    # Flatten the dataset array, put one matrix under the other: (num_dp, len_ts, dim) -> (len_flat_dim, dim)
    array_flatdim = dataset_array.reshape(-1, num_dim)

    # Length of all the time series in one dimension -> Assumption, all dimensions have the same length
    len_flat_dim = array_flatdim.shape[0]

    
    # If the last block is smaller than block_size/2, we will transform it togehter with the previous block.
    # If len_last_block zero, then we don't need to do anything. Last transform fits perfectly.
    len_last_block = len_flat_dim % block_size
    if  len_last_block != 0 and len_last_block < block_size/2:
        last_transfrom_large = True

    # Create blocks and so transforms for each dimension individually. Since different dimensions have different ts-shapes. The transform would transform worse, if we mix dimensions for transforming.
    for i  in range(num_dim):
        end_loop = False
        start_idx = 0
        end_idx = block_size

        while not end_loop:

            # If the last block is smaller than block_size/2, we will transform it togehter with the previous block.
            # end_idx + 1 * block_size is the last index of the block before last block, so again + block_size to get end of last block.
            # > and not >= because if it is equal, the last block fits perfectly (but can't happen here as we only enter this loop if last block is filled less than half)
            if last_transfrom_large and (end_idx + 2 * block_size) > len_flat_dim:
                end_idx = len_flat_dim
                end_loop = True

            
            # If the last block (is bigger than block_size/2) and is bigger than len_flat_dim, we need to adjust the end_idx.
            # We do >= to end loop, also when it fits perfectly. First assignment is then redundant.
            if end_idx >= len_flat_dim:
                end_idx = len_flat_dim
                end_loop = True


            # Compress the block
            array_flatdim[start_idx:end_idx, i] = apply_compression(array_flatdim[start_idx:end_idx, i], compression_type, compression_param, andDecompress)

            # Adjust indices for next block
            start_idx = end_idx
            end_idx += block_size

        # Depending on andDecompress return the compressed array(coefficients) or the decompressed array(Approximation).
        if andDecompress == False:
            # Return array in format (len_flat_dim, dim).
            return array_flatdim
        else: 
            #Bring back from (len_flat_dim, dim) to (num_dp, len_ts, dim). Remember reshape always only returns views!
            dataset_array_compressed = array_flatdim.reshape(-1,len_ts, num_dim)
            return dataset_array_compressed



        return array_flatdim

        if path_to_save != None:
            # Add Default Filename
            dataset_id + '_compressed.bin'
            # Flattten on 2d-array puts rows behind each other! -> to reconstruct only need number of columns(=num_dim)
            array_totally_flat = array_flatdim.flatten()

            # Save to file. (Adjust with npy in case. Then no flattening needed.) Keep in mind. Cant transfer .bin to other machines!
            array_totally_flat.tofile(path_to_save)


    



# Maybe add wrapper to load directly from file!

# Checked internal logic! And one simple test case!
def calculateCompRatio(dataset_array, array_flatdim_comp):

    

    # Compress with gzip
    # Save it as dim after dim, to keep ts and blocks together saved.  Not sure if it makes difference, but keeps the idea-logic!
    array_flatdim_1d = array_flatdim_comp.reshape(-1, order='F')
    
    # Transform to pyton bytes object. It flattens the array row after row (and slice after slice). (Iflattened already manually.)
    # Then gives us a byte-string with each element in byte format.(Just the numbers of the array one after another. No metadata.
    compressed_data_bytestring = array_flatdim_1d.tobytes()

    # Compress the byte-string with gzip. -> Input is byte string and then returns a byte-string. ->> Test more by yourself!
    compressed_data_gzipd = gzip.compress(compressed_data_bytestring, compresslevel=9)


    # .nbytes simply returns: np.prod(a.shape) * a.itemsize -> simply how many bytes are filled in the array. No metadata. Only counts the byte of the elements.
    num_bytes_raw = dataset_array.nbytes

    # len(bytestring) returns number of bytes in the bytestring! No metadata. No nothing!
    return num_bytes_raw / len(compressed_data_gzipd)





   



















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

    
   