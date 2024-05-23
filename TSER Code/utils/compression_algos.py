import pandas as pd
import numpy as np

from scipy.fft import dct, idct, fft, ifft
import pywt



def apply_compression(data_x_p, compression_type, dropout_ratio):
    # Apply compression to the data

    if compression_type == 'dwt':
        for i in range(data_x_p.shape[0]):
            for j in range(data_x_p.shape[2]):
                data_x_p[i,:,j] = dwt_compress(data_x_p[i,:,j], dropout_ratio)

    elif compression_type == 'dct':
        for i in range(data_x_p.shape[0]):
            for j in range(data_x_p.shape[2]):
                data_x_p[i,:,j] = dct_compress(data_x_p[i,:,j], dropout_ratio)
        
    elif compression_type == 'dft':
        for i in range(data_x_p.shape[0]):
            for j in range(data_x_p.shape[2]):
                data_x_p[i,:,j] = dft_compress(data_x_p[i,:,j], dropout_ratio)
        
    else:
        raise ValueError("Invalid compression type. Must be one of 'dwt', 'dct', or 'dft'.")

    return data_x_p


# Compression Methods:
# For all 3:
# Internal logic is checked with print statments
# External logic:
# Checked for iee and appliances rmse per datapoint, gets higher with more compression
# Checked edge cases 0 and 1 dropout ratio
# Checked roughly values after compression are approximations of the original signal


def dct_compress(signal, dropout_ratio):
    # Apply DCT to the signal
    signal_transformed = dct(signal, type=2, norm=None)

    # Calculate the number of coefficients to zero out
    num_coeffs = int((dropout_ratio) * len(signal_transformed))
    
    # Sort the coefficients by magnitude and cut of the smallest ones
    sorted_indices = np.argsort(np.abs(signal_transformed))
    indices_to_zero = sorted_indices[:num_coeffs]

    # Zero out selected coefficients
    signal_transformed[indices_to_zero] = 0

    # Reconstruct the signal using inverse DCT
    compressed_signal = idct(signal_transformed, type=2, norm=None)

    return compressed_signal


# DFT with thresholding to simply handle hermetian symmetry problem.
# This results in effect, that sometimes compression i a little weaker than the dropout_ratio demands.
# Thresholding introducs small probability of cutting other coeff pairs, if by chance 2 frequencies have exactly the same amplitude.
def dft_compress(signal, dropout_ratio):

    norm = "ortho"
    # Ensure valid dropout_ratio, with threshold and my implementation dropout of 1 doesn't work
    if not 0 <= dropout_ratio < 1:
        raise ValueError("Dropout ratio must be between 0 and 1 (exclusive).")

    # Perform DFT
    dft_coeffs = np.fft.fft(signal, norm= norm)

    # Calculate threshold based on sorted magnitudes
    sorted_magnitudes = np.sort(np.abs(dft_coeffs))
    threshold_index = int(len(dft_coeffs) * dropout_ratio)
    threshold = sorted_magnitudes[threshold_index]

    # Zero out coefficients below the threshold
    
    dft_coeffs[np.abs(dft_coeffs) < threshold] = 0

    # Reconstruct the signal using inverse DFT
    compressed_signal = np.fft.ifft(dft_coeffs, norm= norm)

    return compressed_signal.real  # Return only the real part



# My decisions:
# wavelet = db4
# level = max_level / 2 and then to round up
# compress both, detail and approximation coefficients
# global and hard-cut thresholding


def dwt_compress(signal, dropout_ratio):
    """
    Args:
        signal: The input signal (1D numpy array).
        dropout_ratio: The ratio of coefficients to be zeroed out.
        
    Returns:
        Compressed signal (1D numpy array).
    """
    # My parameters
    wavelet = 'db4'
    max_level = pywt.dwt_max_level(len(signal), wavelet) 
    
    
    # guarantees max_level / 2 and then to round up!
    level = (max_level +1 ) // 2
  
    # Decompose the signal -> coeffs is an list of arrays!
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Concatenate detail coefficients only -> for only detail coeff change to one
    concat_coeffs = np.concatenate(coeffs[0:])
 
    # Sort indices of concatenated coefficients
    sorted_indices = np.argsort(np.abs(concat_coeffs))
    
    # Determine number of coefficients to zero out
    num_values_to_zero = int(dropout_ratio * len(concat_coeffs))
    
    # Indices to zero out -> the first elements of array are the smallest
    zero_indices = sorted_indices[:num_values_to_zero]
   
    # Zero out selected coefficients
    concat_coeffs[zero_indices] = 0
   
    # Reconstruct individual detail coefficients -> for only detail coeff change range(1, len(coeffs))
    start_idx = 0
    for i in range(0, len(coeffs)):
        # Length of current coefficient
        coeff_len = len(coeffs[i])
        # End index of current coefficient
        end_idx = start_idx + coeff_len
        # Assign values to the coefficient
        coeffs[i] = concat_coeffs[start_idx:end_idx]
        # Update start index for next coefficient
        start_idx = end_idx
        
    # Reconstruct the signal
    compressed_signal = pywt.waverec(coeffs, wavelet)


    return compressed_signal