import numpy as np
from mGrangerdDTF import *
from scipy.stats import ttest_1samp


# This code is in an implementation of steps to implement the statistical tests described in the work of Liu et al, 2012.

class mGrangerStats2:
    def __init__(self, original_data):
        self.original_data = original_data

    def generate_surrogate(self, num_sets=2500):

        """
        Inputs:
        original_data: 2D array with time series data of the independent components (samples X ICs)
        num_sets: number of sets of surrogate data to be computed

        Output: 3D array (num_sets X samples X ICs) with surrogate time series for each IC that shares
        spectral properties with the original data but lacks the specific temporal relationships.
        """
        num_time_points, num_ICs = self.original_data.shape
        surrogate_data = np.zeros((num_sets, num_time_points, num_ICs))
        for s in range(num_sets):
            for ic in range(num_ICs):
                # Fourier transform for each IC
                original_fft = np.fft.fft(self.original_data[:, ic])

                # Parameters
                n = len(original_fft)  # length of the fourier transform
                half_len = (n - 1) // 2  # half the length of the FT
                idx1 = np.arange(1, half_len + 1, dtype=int)  # 1st half
                idx2 = np.arange(half_len + 1, n, dtype=int)  # 2nd half

                # Generate random phases for each half
                phases1 = np.exp(2.0 * np.pi * 1j * np.random.rand(half_len))
                phases2 = np.exp(2.0 * np.pi * 1j * np.random.rand(half_len))

                # Apply random phases to each half
                surrogate_fft = np.zeros_like(original_fft, dtype=complex)
                surrogate_fft[0] = original_fft[0] # Symmetry! 
                surrogate_fft[idx1] = original_fft[idx1] * phases1  # 1st half
                phases2 = np.exp(2.0 * np.pi * 1j * np.random.rand(len(idx2))) # Phases for 2nd half with its specific length
                surrogate_fft[idx2] = original_fft[idx2] * phases2  # 2nd half, corrected for dimensions. 2nd correction after problems with ADwAwD group

                # Inverse Fourier Transform
                surrogate_data[s, :, ic] = np.real(np.fft.ifft(surrogate_fft))

        return surrogate_data

    def significance_testing(self, surrogate_series, my_pvalue):
        """
        Inputs: 
        original_series, from which to generate data (dimensions samples X ICs)
        my_pvalue: significance value (for convenience p=0.05)

        Output:
        effective_connectivity_network, a matrix where the non-significant connections (p>=0.05, meaning that the measured value is 
        smaller than that obtained from the null distribution) are suppressed.
        """
        num_sets, num_time_points, num_ICs = surrogate_series.shape
        # Initiate classes
        og = mGrangerdDTF(data=self.original_data)  # original
        dDTF_og = np.abs(og.dDTF_ij)  # dDTF matrix of the original series
        p_values = np.zeros((num_ICs, num_ICs))

        # Computing the dDTF for each set in surrogate_series
        su = np.zeros((num_sets, num_ICs, num_ICs))
        for s in range(num_sets):
            temporal_surrogate = surrogate_series[s, :, :]
            temporal_su = mGrangerdDTF(data=temporal_surrogate)
            # Compute dDTF matrix for each set
            su[s, :, :] = np.abs(temporal_su.dDTF_ij)

        # Compare Original dDTF with Null Distribution for all elements
        for i in range(num_ICs):
            for j in range(num_ICs):
                original_value = dDTF_og[i, j]
                surrogate_values = su[:, i, j]
                t_statistic, p_value = ttest_1samp(surrogate_values, original_value)
                p_values[i, j] = p_value / 2  # One-tailed test

        # Identify Significant Connections
        significance_levels = p_values < my_pvalue
        significant_connections = np.zeros((num_ICs, num_ICs), dtype=bool)
        for i in range(num_ICs):
            for j in range(num_ICs):
                if significance_levels[i, j]:
                    significant_connections[i, j] = True

        # Network where the significanct connections are preserved
        effective_connectivity_network = dDTF_og * significant_connections

        return effective_connectivity_network
