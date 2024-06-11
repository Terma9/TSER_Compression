import pandas as pd
import numpy as np

from scipy.fft import dct, idct, fft, ifft
import pywt



def apply_compression(signal, compression_type, comp_param, andDecompress:bool):
    # Apply compression to the data


    if compression_type == 'dwt':
        return dwt_compress(signal, comp_param, andDecompress)

    elif compression_type == 'dct':
         return dct_compress(signal, comp_param, andDecompress)

    elif compression_type == 'dft':
        return dft_compress(signal, comp_param, andDecompress)
            
    elif compression_type == 'cpt':
        return cpt_compress(signal, comp_param, andDecompress)


    else:
        raise ValueError("Invalid compression type. Must be one of 'dwt', 'dct', or 'dft'.")




# Compression Methods:
# For all 3:
# Internal logic is checked with print statments
# External logic:
# Checked for iee and appliances rmse per datapoint, gets higher with more compression
# Checked edge cases 0 and 1 dropout ratio
# Checked roughly values after compression are approximations of the original signal


def dct_compress(signal, dropout_ratio, andDecompress:bool):
    # Apply DCT to the signal
    dct_coeffs = dct(signal, type=2, norm=None)

    # Calculate the number of coefficients to zero out
    num_coeffs = int((dropout_ratio) * len(dct_coeffs))
    
    # Sort the coefficients by magnitude and cut off the smallest ones
    sorted_indices = np.argsort(np.abs(dct_coeffs))
    indices_to_zero = sorted_indices[:num_coeffs]

    # Zero out selected coefficients
    dct_coeffs[indices_to_zero] = 0

    if andDecompress == False:
        return dct_coeffs
    else:
        decompressed_signal = idct(dct_coeffs, type=2, norm=None)
        return decompressed_signal








# DFT with thresholding to simply handle hermetian symmetry problem.
# This results in effect, that sometimes compression i a little weaker than the dropout_ratio demands.
# Thresholding introducs small probability of cutting other coeff pairs, if by chance 2 frequencies have exactly the same amplitude.
def dft_compress(signal, dropout_ratio, andDecompress:bool):

    norm = "ortho"
    # Ensure valid dropout_ratio, with threshold and my implementation dropout of 1 doesn't work
    #if not 0 <= dropout_ratio < 1:
    #    raise ValueError("Dropout ratio must be between 0 and 1 (exclusive).")

    # Perform DFT
    dft_coeffs = np.fft.fft(signal, norm= norm)

    # Calculate threshold based on sorted magnitudes
    sorted_magnitudes = np.sort(np.abs(dft_coeffs))
    threshold_index = int(len(dft_coeffs) * dropout_ratio)

    # Check if threshold_index is valid, if not return zeroed out coefficients
    if threshold_index == len(dft_coeffs):
        return np.zeros_like(dft_coeffs)

    threshold = sorted_magnitudes[threshold_index]

    # Zero out coefficients below the threshold
    
    dft_coeffs[np.abs(dft_coeffs) < threshold] = 0


    if andDecompress == False:
        return dft_coeffs
    else:
        decompressed_signal = np.fft.ifft(dft_coeffs, norm= norm)
        return decompressed_signal.real # Return only the real part


#-> Chance to zero out one coefficient less than on dct, dwt -> to contain hermetian symmetry
#-> Unlikely case of 2 frequencies with same amplitude, and threshold index is between them none aret -> how high probability?



# My decisions:
# wavelet = db4
# level = max_level / 2 and then to round up
# compress both, detail and approximation coefficients
# global and hard-cut thresholding
# Signal Extension Mode = Periodic! If not, the len of coefficients is not equal to the len of the signal

# -> couls use pywt coeff functions for less code!
def dwt_compress(signal, dropout_ratio, andDecompress:bool):
    """
    Args:
        signal: The input signal (1D numpy array).
        dropout_ratio: The ratio of coefficients to be zeroed out.
        
    Returns:
        Decompressed signal (1D numpy array).
    """
    # My parameters
    wavelet = 'db4'

    # Add manual max level depending on wavelet! + Also add max_level if hardcoded level is too high!
    max_level = pywt.dwt_max_level(len(signal), wavelet) 
    # guarantees max_level / 2 and then to round up!
    level = (max_level +1 ) // 2
  

    # Decompose the signal -> coeffs is a list of arrays!
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode='per')
    
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

    if andDecompress == False:
        return concat_coeffs
    
    else:
        # Reconstruct individual coefficients -> for only detail coeff change range(1, len(coeffs))
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
        compressed_signal = pywt.waverec(coeffs, wavelet, mode='per')


        return compressed_signal
    








from numpy.polynomial.chebyshev import chebfit, chebval


# Chebyshev Transoform Compression Code -WIP
def cpt_compress(signal, dropout_ratio, andDecompress:bool):
    
    # Later: Try with fitting into Interval -1,1
    t = np.arange(0, len(signal))

    # Get the coefficients. deg is the longest that is possible. Shouldn't take too long to calculate.
    #cpt_coeffs = chebfit(t, signal, deg=int((signal.size-1)))
    cpt_coeffs = chebfit(t, signal, deg=signal.size-1)

    # Calculate the number of coefficients to zero out
    num_coeffs = int((dropout_ratio) * len(cpt_coeffs))
    
    # Sort the coefficients by value and cut off the smallest ones
    sorted_indices = np.argsort(np.abs(cpt_coeffs))
    indices_to_zero = sorted_indices[:num_coeffs]

    # Zero out selected coefficients
    cpt_coeffs[indices_to_zero] = 0

    if andDecompress == False:
        return cpt_coeffs
    else:
        decompressed_signal = chebval(t, cpt_coeffs)
        return decompressed_signal




 


""" import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval

def min_max_normalize_np(data):

    min_val = np.min(data)
    max_val = np.max(data)
    norm_data = (2 * (data - min_val) / (max_val - min_val)) - 1
    return norm_data

def inverse_min_max_normalize_np(norm_data, min_val, max_val):

    orig_data = ((norm_data + 1) * (max_val - min_val) / 2) + min_val
    return orig_data

def cpt_compress(signal, dropout_ratio, andDecompress:bool):
    # Normalize the signal to the range [-1, 1]
    t = min_max_normalize_np(signal)
    
    # Create the time vector for Chebyshev fitting
    #t = np.linspace(-1, 1, signal_norm.size)

    # Fit Chebyshev polynomial
    cpt_coeffs = chebfit(t, signal, deg=signal.size-1)

    # Calculate the number of coefficients to zero out
    num_coeffs = int((dropout_ratio) * len(cpt_coeffs))
    
    # Sort the coefficients by value and cut off the smallest ones
    sorted_indices = np.argsort(np.abs(cpt_coeffs))
    indices_to_zero = sorted_indices[:num_coeffs]

    # Zero out selected coefficients
    cpt_coeffs[indices_to_zero] = 0

    if andDecompress == False:
        return cpt_coeffs
    else:
        # Decompress and inverse normalize the signal
        decompressed_signal_norm = chebval(t, cpt_coeffs)
        decompressed_signal = inverse_min_max_normalize_np(decompressed_signal_norm, np.min(signal), np.max(signal))
        return decompressed_signal """