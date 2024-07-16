import pandas as pd
import numpy as np
from scipy.optimize import linprog
from scipy.fft import dct, idct, fft, ifft
import pywt



def apply_compression(signal, compression_type, comp_param, andDecompress:bool, level, wavelet, quantization_level):
    # Apply compression to the data


    if compression_type == 'dwt':
        return dwt_compress(signal, comp_param, andDecompress, level, wavelet, quantization_level)

    elif compression_type == 'dct':
         return dct_compress(signal, comp_param, andDecompress, quantization_level)

    elif compression_type == 'dft':
        return dft_compress(signal, comp_param, andDecompress, quantization_level)
            
    elif compression_type == 'cpt':
        return cpt_compress(signal, comp_param, andDecompress, quantization_level)


    else:
        raise ValueError("Invalid compression type. Must be one of 'dwt', 'dct', or 'dft'.")




# Compression Methods:
# For all 3:
# Internal logic is checked with print statments
# External logic:
# Checked for iee and appliances rmse per datapoint, gets higher with more compression
# Checked edge cases 0 and 1 dropout ratio
# Checked roughly values after compression are approximations of the original signal


def dct_compress(signal, dropout_ratio, andDecompress:bool, quantization_level):

    # Apply DCT to the signal
    dct_coeffs = dct(signal, type=2, norm=None)

    # Calculate the number of coefficients to zero out
    num_coeffs = int((dropout_ratio) * len(dct_coeffs))
    
    # Sort the coefficients by magnitude and cut off the smallest ones
    sorted_indices = np.argsort(np.abs(dct_coeffs))
    indices_to_zero = sorted_indices[:num_coeffs]

    # Zero out selected coefficients
    dct_coeffs[indices_to_zero] = 0

     # Add quantization. Round to qlevel number of decimal places
    dct_coeffs = np.round(dct_coeffs, quantization_level)

    if andDecompress == False:
        return dct_coeffs
    else:
        decompressed_signal = idct(dct_coeffs, type=2, norm=None)
        return decompressed_signal








# DFT with thresholding to simply handle hermetian symmetry problem.
# This results in effect, that sometimes compression is a little weaker than the dropout_ratio demands. ( We only apply it on 1000er blocks, so it has not a big effect.)
# Thresholding introducs small probability of cutting other coeff pairs, if by chance 2 frequencies have exactly the same amplitude.
def dft_compress(signal, dropout_ratio, andDecompress:bool, quantization_level):

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

     # Add quantization. Round to qlevel number of decimal places
    dft_coeffs = np.round(dft_coeffs, quantization_level)


    if andDecompress == False:
        return dft_coeffs
    else:
        decompressed_signal = np.fft.ifft(dft_coeffs, norm= norm)
        return decompressed_signal.real # Return only the real part


#-> Chance to zero out one coefficient less than on dct, dwt -> to contain hermetian symmetry
#-> Unlikely case of 2 frequencies with same amplitude, and threshold index is between them none aret -> how high probability?



# My decisions:
# wavelet = db4 and haar
# compress both, detail and approximation coefficients
# global and hard-cut thresholding
# Signal Extension Mode = periodization! If not, the len of coefficients is not equal to the len of the signal.
# -> could use pywt coeff functions for less code!

# extend for level and type for testing! -> remove after decision is made
def dwt_compress(signal, dropout_ratio, andDecompress:bool, level, wavelet, quantization_level):
    """
    Args:
        signal: The input signal (1D numpy array).
        dropout_ratio: The ratio of coefficients to be zeroed out.
        
    Returns:
        Decompressed signal (1D numpy array).
    """

    # Add manual max level depending on wavelet! + Also add max_level if hardcoded level is too high!
    max_level = pywt.dwt_max_level(len(signal), wavelet) 

    if(max_level < level):
        level = max_level

        
    # Decompose the signal -> coeffs is a list of arrays!
    # Periodization is the right signal, not periodic!
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode='periodization')
    
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


    # Add quantization. Round to qlevel number of decimal places
    concat_coeffs = np.round(concat_coeffs, quantization_level)



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
        compressed_signal = pywt.waverec(coeffs, wavelet, mode='periodization')


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





# PPA Subfuncitions





# PMR-Midrange for degree 0, returns the error and the approximation
# maybe adapt to return only one number instead of signal

def pmr_midrange(signal):

    min_val = np.min(signal)
    max_val = np.max(signal)
    midrange_value = (min_val + max_val) / 2
    pmr_signal = np.full_like(signal, midrange_value)
    

    # Calculate the error
    error = np.abs(signal - pmr_signal)

    # Compute the uniform norm of the error
    uniform_norm_error = np.max(error)

    return uniform_norm_error, pmr_signal

    




# Handles everything up from degree one
def approx_deg_p(signal, degree):


    # will later see how to handle it with index, what exactly is saved in signal! IMPORTANT


    x = np.arange(signal.size)
    y = signal  

    # Degree of the polynomial 
    degree

    # Number of data points
    n = len(x)

    # Coefficients for the linear programming problem
    # There will be degree + 1 coefficients for the polynomial plus one for the error term
    c = np.zeros(degree + 2)
    c[-1] = 1  # Coefficient for the error term

    # Inequality constraints
    A_ub = []
    b_ub = []


    # Look at photo for calculations and rearrangements!
    for i in range(n):
        row_pos = []
        row_neg = []
        for j in range(degree + 1):
            row_pos.append(x[i]**j)
            row_neg.append(-x[i]**j)
        
        row_pos.append(-1)  # -t term
        row_neg.append(-1)  # -t term
        
        A_ub.append(row_pos)  # f(x_i) - g(x_i) <= t
        b_ub.append(y[i])
        
        A_ub.append(row_neg)  # g(x_i) - f(x_i) <= t
        b_ub.append(-y[i])

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds for the variables
    bounds = [(None, None)] * (degree + 1) + [(0, None)]  # No bounds on polynomial coefficients, t >= 0

    # Solve the linear programming problem
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    # Check if the optimization was successful
    if result.success:
        # Extract the solution
        coeffs = result.x[:-1]  # The polynomial coefficients
        min_error = result.x[-1]  # The minimum error t
        #print(f"Optimal polynomial coefficients: {coeffs}")
        #print(f"Minimum error t: {min_error}")

        return min_error, coeffs
    
    else:
        print("Optimization failed.")
        print(f"Status: {result.status}")
        print(f"Message: {result.message}")
    



