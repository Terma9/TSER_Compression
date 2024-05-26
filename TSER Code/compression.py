import numpy as np

# define global variable

dataset_params = {
    'appliances': {'block_size': 10, 'num_dim': 3, 'len_ts': 100},
    'dataset2': {'block_size': 15, 'num_dim': 4, 'len_ts': 120},
    # Add more datasets as needed
}






# Compression of dataset-array.
def compress(dataset_array, dataset_id, compression_type=None, compression_param=None, path_to_save='compressed.bin'):

    # Retrieve dataset parameters
    num_dim = dataset_params[dataset_id]['num_dim']
    #len_ts = dataset_params[dataset_id]['len_ts'] not needed, only for decompression
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
            array_flatdim[start_idx:end_idx, i] = do_compression(array_flatdim[start_idx:end_idx, i], compression_type, compression_param)

            # Adjust indices for next block
            start_idx = end_idx
            end_idx += block_size


        if path_to_save==None:
            return array_flatdim
        
        
        # Flattten on 2d-array puts rows behind each other! -> to reconstruct only need number of columns(=num_dim)
        array_totally_flat = array_flatdim.flatten()

        # Save to file. (Adjust with npy in case. Then no flattening needed.) Keep in mind. Cant transfer .bin to other machines!
        array_totally_flat.tofile(path_to_save)

        

# Load from datafile in wrapper function then!





# Decompress from File!
# Return array now, but adjust later to save to file!
def decompress(compr_dataset_array, dataset_id, compression_type=None, compression_param=None):
    
    # Retrieve dataset parameters
    num_dim = dataset_params[dataset_id]['num_dim']
    len_ts = dataset_params[dataset_id]['len_ts']
    #block_size = dataset_params[dataset_id]['block_size'] -> not needed, only for compression

    # Load compressed data
    array_totally_flat = np.fromfile(path_to_compressed)

    # Calculat the number of datapoints in the dataset. Saver than passing it as an argument. Should always be whole number./integer.
    num_dp = array_totally_flat.shape[0] / (len_ts * num_dim)

    # Reshape the data. Works as intended and puts array after num_dim values under it and so on.
    array_flatdim = array_totally_flat.reshape(-1, num_dim)



    decomp_dataset_array = np.zeros((num_dp,len_ts, num_dim))

    # Go for each dim over all ts and decompress
    for i in range(num_dim):
        for j in range(num_dp):
            start_idx = 0
            end_idx = len_ts
            decomp_dataset_array[j,:,i] = do_decompression(array_flatdim[j,:,i], compression_type, compression_param)

            start_idx = end_idx
            end_idx += len_ts

    
    return decomp_dataset_array


    
   