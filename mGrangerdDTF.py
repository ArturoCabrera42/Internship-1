import numpy as np
from statsmodels.tsa.api import VAR

# This code is in an implementation of the steps described in the work of Liu et al, 2012.

class mGrangerdDTF:
    def __init__(self, data, order=1, num_frequencies=10, lower_frequency_limit=0.01, upper_frequency_limit=0.1):
        # Regression and Fourier transform variables
        self.data = data
        self.order = order
        self.num_frequencies = num_frequencies
        self.lower_frequency_limit = lower_frequency_limit
        self.upper_frequency_limit = upper_frequency_limit

        # Variables to store intermediate results
        self.A_n = None # Regression coefficients, order 1
        self.errors = None # Errors vector of the MVAR model
        self.A_f = None # Fourier transform of the regression coefficients, for each frequency
        self.H = None # H(f) = A^{-1}(f)
        self.H_h = None # H^H Hermitian conjugate
        self.errors_f = None # Fourier transform of the Errors vector, for each frequency
        self.V = None # Variances
        self.S_f = None # Cross-spectra
        self.theta_ij = None # Partial coherence
        self.dDTF_ij = None #dDTF - Complex values!
        self.abs_dDTF = None # Absolute value of the dDTF to compute in/out/in+out-degrees

        # Execution, for its use across notebooks
        self.execute_sequence()


    def compute_A_n(self,data, order):
        """
        Compute the A(n) coefficient matrix of a vector autoregressive (VAR) model. 
        Inputs.
        data = subject's fMRI data. Must be ordered so it has dimensions (samples x ROIs). 
        order = for the regression coefficient. Computed with AIC.

        Output
        matrix_coefficients = the A(n) matrix of order n as specified in the input
        """ 

        model = VAR(data)
        result = model.fit(order)
        matrix_coefficients = np.stack(result.coefs, axis = 0)
        self.A_n = matrix_coefficients[0, :, :]# this directly extracts the first regression, order is 1
        return self.A_n

    def compute_E(self,data, A):
        """
        Compute the errors vector for a Multivariate AutoRegressive (MVAR) model.

        Parameters:
        - data (numpy.ndarray): The input time series data matrix with dimensions (samples x IC),
                            where IC is the number of observed variables.
        - A (numpy.ndarray): Coefficients matrix obtained from the MVAR model. It represents
                            the lagged terms in the MVAR equation.

        Returns:
        - errors (numpy.ndarray): The errors vector calculated based on the MVAR model.

        MVAR Equation:
        X(t) = \Sum_{n=1}^p A(n)X(t-n) + E(t)

        Algorithm:
        1. Extract the lag order (n) from the shape of the coefficients matrix A.
        2. Extract the coefficients (A matrices) from the VAR model for every n value.
        3. Extract the observed variables from the data matrix, excluding the first n rows.
        4. Initialize an array to store errors.
        5. Compute the errors using the MVAR model equation.

        Note:
        - The errors vector represents the difference between the observed variables and
        the predicted values based on the MVAR model.
        """
        # Lag order
        lag_order = A.shape[0]

        # extracts the coefficients (A matrices)
        coefficients = [A[i] for i in range(lag_order)]

        
        observed_variables = data[lag_order:, :]

        
        errors = np.zeros_like(observed_variables)

        
        for i in range(lag_order, len(observed_variables)):
            errors[i - lag_order] = observed_variables[i]
            for lag in range(1, lag_order + 1):
                errors[i - lag_order] -= np.dot(coefficients[lag - 1], observed_variables[i - lag])
        
        self.errors = errors
        return self.errors

    def compute_A_f(self,A_n, num_frequencies, lower_frequency_limit, upper_frequency_limit):
        """
        Compute frequency slices of the 2D Fourier Transform of a coefficients matrix.

        Parameters:
        - A_n (numpy.ndarray): Coefficients matrix obtained from the VAR model.
        - num_frequencies (int): Number of frequencies to analyze.
        - lower_frequency_limit (float): Lower limit of the frequency range.
        - upper_frequency_limit (float): Upper limit of the frequency range.

        Returns:
        - a_ij_f_slices (numpy.ndarray): Array containing frequency slices of the coefficients matrix.
        """

    
        A_fft = np.fft.fft2(A_n)

        # stores the results for each frequency
        a_ij_f_slices = np.zeros((num_frequencies,) + A_n.shape, dtype=np.complex128)

        
        for idx, f in enumerate(np.linspace(lower_frequency_limit, upper_frequency_limit, num_frequencies)):
            
            exp_term = np.exp(-2j * np.pi * f * np.fft.fftfreq(A_n.shape[0])[:, np.newaxis] * np.arange(A_n.shape[0]))

            
            a_ij_f = A_fft * exp_term

            # Dirac's delta
            i, j = np.indices(A_n.shape)
            a_ij_f -= np.eye(A_n.shape[0])[i, j]

            # Store the result for the current frequency in the corresponding slice
            a_ij_f_slices[idx] = a_ij_f

        self.A_f = a_ij_f_slices

        return self.A_f
    
    def compute_H_f(self,a_ij_f_slices):
        """
        Compute the inverses of frequency slices of a coefficients matrix.

        Parameters:
        - a_ij_f_slices (numpy.ndarray): Array containing frequency slices of the coefficients matrix A(f).

        Returns:
        - h (numpy.ndarray): Array containing the inverses of the frequency slices (H(f) = A^-1(f)).
        """

        
        h = np.zeros_like(a_ij_f_slices, dtype=np.complex128)

        
        for i in range(a_ij_f_slices.shape[0]):
            
            h[i] = np.linalg.inv(a_ij_f_slices[i])

        self.H = h
        return self.H
    
    def compute_H_h(self,matrix):
        """
        Computes the Hermitian conjugate of the matrix H, H^H, defined as the
        conjugate of the transposed.
        """

        H_h = np.empty_like(matrix, dtype=np.complex128) # H^H(f)

        for i in range(matrix.shape[0]): 
            H_h[i, :, :] = np.conjugate(matrix[i, :, :].T)

        self.H_h = H_h

        return self.H_h

    def compute_E_f(self,errors, lower_frequency_limit, upper_frequency_limit, num_frequencies):
        """
        Compute the Fourier transform of each IC for a range of frequencies.

        Parameters:
        - errors (numpy.ndarray): The errors matrix obtained from the compute_E function with dimensions (samples x IC).
        - lower_frequency_limit (float): Lower limit of the frequency range.
        - upper_frequency_limit (float): Upper limit of the frequency range.
        - num_frequencies (int): Number of frequencies to compute.

        Returns:
        - fourier_matrix (numpy.ndarray): 3D array containing Fourier transforms for each frequency and each IC.
        """

        
        frequencies = np.linspace(lower_frequency_limit, upper_frequency_limit, num_frequencies)

        # stores Fourier transforms
        fourier_matrix = np.zeros((num_frequencies, errors.shape[0], errors.shape[1]), dtype=np.complex64)

        # Compute Fourier transform for each IC and each frequency
        for i in range(errors.shape[1]):  # Loop over ICs
            for j in range(num_frequencies):  # Loop over frequencies
                fourier_matrix[j, :, i] = np.fft.fft(errors[:, i])

        self.errors_f = fourier_matrix

        return self.errors_f

    def compute_V(self,errors):
        """
        Input: E(f), matrix of errors in the frequency domain
        Output: 1D array, each element is the variance at a specific frequency
        """
        self.V = np.var(errors, axis=(1, 2))
        return self.V

    def compute_S_f(self,H, H_h, V):
        """
        Compute the cross-spectra
        """

        S = np.zeros_like(H, dtype=complex)
        num_frequencies = H.shape[0]
        for i in range(num_frequencies):
            S[i,:,:] = H[i,:,:] * V[i] * H_h[i,:,:]

        self.S_f = S
        return self.S_f
    
    def normalize_complex_arr(self,a): #offsets a matrix to help with the normalization to range [0,1]
        a_oo = a - a.real.min() - 1j * a.imag.min()  # origin offsetted
        return a_oo / np.abs(a_oo).max()

    def compute_partial_coherence(self,matrix):
        slices = matrix.shape[0]
        cof = np.zeros_like(matrix, dtype=complex)
        
        for s in range(slices):
            cof[s, :, :] = np.linalg.inv(matrix[s, :, :]).T * np.linalg.det(matrix[s, :, :]) # M - cofactors matrix for each slice
        theta = np.zeros_like(matrix, dtype=complex)

        for s in range(slices):
            for i in range(matrix.shape[1]):
                for j in range(matrix.shape[2]):
                    theta[s, i, j] = cof[s, i, j] ** 2 / (cof[s, i, i] * cof[s, j, j])

        # Normalize the theta values for both real and imaginary parts
        theta_normalized = self.normalize_complex_arr(theta)
        theta_ij = np.abs(theta_normalized)
        self.theta_ij = theta_ij
        return self.theta_ij
    
    def compute_dDTF(self,h_ij, theta_ij):
        """
        Compute the dDTF (direct directed transfer function).

        Parameters:
        - theta_ij: 3D array representing theta values (complex)
        - h_ij: 3D array representing h values (range [0,1])

        Returns:
        - dDTF_ij: 2D array representing dDTF values
        """
        slices, rows, cols = theta_ij.shape
        dDTF_ij = np.zeros((rows, cols), dtype=complex)

        for i in range(rows):
            for j in range(cols):
                for f in range(slices):
                    dDTF_ij[i, j] += h_ij[f, i, j] * theta_ij[f, i, j]

        self.dDTF_ij = dDTF_ij
        return self.dDTF_ij
    
    def compute_abs_dDTF(self, matrix):
        """
        Compute the absolute value of the dDTF function to obtain the in/out/in+out - degrees
        """

        self.abs_dDTF = np.abs(matrix)
        return self.abs_dDTF

    def in_out_degree(self, ddtf_matrix):
        """
        Compute in-degree and out-degree measures based on the dDTF matrix.

        Returns:
        - in_degree (numpy.ndarray): 1D array representing in-degree for independent component.
        - out_degree (numpy.ndarray): 1D array representing out-degree for independent component.
        - in_and_out (numpy.ndarrat): 1D array, in+out degree for independent component.
        """
        in_degree = np.sum(np.abs(ddtf_matrix), axis=1)
        out_degree = np.sum(np.abs(ddtf_matrix), axis=0)
        in_and_out = in_degree + out_degree

        return in_degree, out_degree, in_and_out
    

    def execute_sequence(self):
        self.A_n = self.compute_A_n(data=self.data, order=self.order)
        self.errors = self.compute_E(data=self.data, A=self.A_n)
        self.A_f = self.compute_A_f(A_n=self.A_n, num_frequencies=self.num_frequencies,
                                    lower_frequency_limit=self.lower_frequency_limit,
                                    upper_frequency_limit=self.upper_frequency_limit)
        self.H = self.compute_H_f(a_ij_f_slices=self.A_f)
        self.H_h = self.compute_H_h(matrix=self.H)
        self.errors_f = self.compute_E_f(errors=self.errors, lower_frequency_limit=self.lower_frequency_limit,
                                        upper_frequency_limit=self.upper_frequency_limit,
                                        num_frequencies=self.num_frequencies)
        self.V = self.compute_V(errors=self.errors_f)
        self.S_f = self.compute_S_f(H=self.H, H_h=self.H_h, V=self.V)
        self.theta_ij = self.compute_partial_coherence(matrix=self.S_f)
        self.dDTF_ij = self.compute_dDTF(h_ij=self.H, theta_ij=self.theta_ij)
        self.abs_dDTF = self.compute_abs_dDTF(matrix = self.dDTF_ij)
