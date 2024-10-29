import numpy as np


def get_polyfit(arr, deg=10):

# Identify discontinuity
    differences = np.diff(arr)
    threshold = 0.1  # Define a threshold to identify significant jumps
    discontinuity_index = np.where(np.abs(differences) > threshold)[0]

    # Compute polynomial fit excluding discontinuous region
    if discontinuity_index.size > 0:
        discontinuity_index = discontinuity_index[0]  # Take the first discontinuity index
        x = np.arange(len(arr))
        x_fit = np.delete(x, np.arange(discontinuity_index, discontinuity_index + 2))
        y_fit = np.delete(arr, np.arange(discontinuity_index, discontinuity_index + 2))
    else:
        x = np.arange(len(arr))
        x_fit = x
        y_fit = arr

    # Fit a polynomial
    poly_coeff = np.polyfit(x_fit, y_fit, deg)
    poly_fit = np.poly1d(poly_coeff)
    
    return poly_fit(x)

def joint_fit(arr, logrhoarr, deg=5, logrho_thresh1=-3, logrho_thresh2=0.0):
    
    # fit3 = get_polyfit(arr[logrhoarr > logrho_thresh2], deg=3)
    fit2 = get_polyfit(arr[(logrhoarr >= logrho_thresh1) & (logrhoarr <= logrho_thresh2)], deg=5)
    # fit1 = get_polyfit(arr[logrhoarr < logrho_thresh1], deg=deg)

    # return np.concatenate([fit1, fit2, fit3])

    #fit2 = get_polyfit(arr[logrhoarr >= logrho_thresh], deg=deg)
    fit1 = get_polyfit(arr[logrhoarr < logrho_thresh1], deg=deg)

    return np.concatenate([fit1, fit2, arr[logrhoarr > logrho_thresh2]])

# def gauss_smooth(data, base_sigma=5, base_window=10):
#     smoothed_data = np.zeros_like(data)
#     differences = np.abs(np.diff(data, prepend=data[0]))  # prepend to match the length
#     median_difference = np.median(differences)
    
#     for i in range(len(data)):
#         # Adjust sigma based on local difference information
#         local_sigma = base_sigma * (1 + (median_difference - differences[i]) / (median_difference + 1e-6))
        
#         # Determine the window size dynamically based on the position within the array
#         start_index = max(0, i - base_window // 2)
#         end_index = min(len(data), i + base_window // 2 + 1)
#         actual_window_size = end_index - start_index
        
#         # Generate the Gaussian kernel for the actual window size
#         x = np.linspace(-(actual_window_size // 2), actual_window_size // 2, actual_window_size)
#         gauss_kernel = np.exp(-0.5 * (x / local_sigma) ** 2)
#         gauss_kernel /= gauss_kernel.sum()  # Normalize the kernel
#         #print(actual_window_size)
        
#         # Apply the kernel to the data segment
#         smoothed_data[i] = np.dot(data[start_index:end_index], gauss_kernel)

#     return smoothed_data

def gauss_smooth(data, base_sigma=5, base_window=10, min_sigma=0.1, max_sigma=10):
    # Calculate differences and normalize
    differences = np.abs(np.diff(data, prepend=data[0]))
    median_difference = np.median(differences) + 1e-6  # Avoid division by zero
    normalized_diff = (median_difference - differences) / median_difference

    # Calculate local sigmas and clip to [min_sigma, max_sigma]
    local_sigmas = base_sigma * (1 + normalized_diff)
    local_sigmas = np.clip(local_sigmas, min_sigma, max_sigma)

    # Pre-compute Gaussian kernels for unique sigma values
    unique_sigmas = np.unique(local_sigmas)
    kernel_dict = {}
    for sigma in unique_sigmas:
        # Define window size based on sigma
        window_size = int(base_window * sigma / base_sigma)
        window_size = max(window_size, 3)  # Minimum window size
        if window_size % 2 == 0:
            window_size += 1  # Ensure window size is odd
        x = np.linspace(-window_size // 2, window_size // 2, window_size)
        gauss_kernel = np.exp(-0.5 * (x / sigma) ** 2)
        gauss_kernel /= gauss_kernel.sum()
        kernel_dict[sigma] = gauss_kernel

    # Apply smoothing using the kernels
    smoothed_data = np.zeros_like(data)
    for i, (value, sigma) in enumerate(zip(data, local_sigmas)):
        kernel = kernel_dict[sigma]
        window_size = len(kernel)
        start_index = max(0, i - window_size // 2)
        end_index = min(len(data), i + window_size // 2 + 1)
        data_segment = data[start_index:end_index]

        # Adjust kernel if we're at the edges
        if len(data_segment) != window_size:
            kernel = kernel[(window_size // 2 - (i - start_index)):(window_size // 2 + (end_index - i))]
            kernel /= kernel.sum()  # Renormalize

        smoothed_data[i] = np.dot(data_segment, kernel)

    return smoothed_data
