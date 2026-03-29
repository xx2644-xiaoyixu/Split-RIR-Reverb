import numpy as np
import soundfile as sf
import time
from IPython.display import display, Audio
import matplotlib.pyplot as plt
import pyroomacoustics as pra

# split full IR into ir_early and ir_late
def split(split_time, h, fs):
    split_sample = int(fs * split_time)
    early = h[0:split_sample]
    late = h[split_sample:]
    return early, late

# FFT Conv
def fast_conv(x, h):
    '''Convolves two 1-dimensional signals
    together using the fast method.
    
    Parameters
    ----------
    x : (np.ndarray)
        The first input signal in an array
        
    h : (np.ndarray)
        The second input signal in an array
        
    Returns
    -------
    y : (np.ndarray)
        The resulting signal from convolving x and h
        using the fast method.
    '''
    # using full mode padding
    N = len(x)
    K = len(h)
    x_padded = np.pad(x, (0, K - 1))
    h_padded = np.pad(h, (0, N - 1))

    X = np.fft.fft(x_padded)
    H = np.fft.fft(h_padded)
    
    Y = X * H
    y = np.real(np.fft.ifft(Y))

    return y

# Block Processing FFT Conv
def running_convolver(x, h, buff_size):
    '''Convolves two 1-dimensional signals
    together using buffers and the overlap-add method.
    
    Parameters
    ----------
    x : (np.ndarray)
        The first input signal in an array
        
    h : (np.ndarray)
        The second input signal in an array

    buff_size : (int)
                The block or buffer size
        
    Returns
    -------
    y : (np.ndarray)
        The resulting signal from convolving x and h
        using running convolution.
    '''
    start_time = time.time()

    N = len(x)
    K = len(h)
    y = np.zeros(N + K - 1) # full mode conv

    num_buffers = int(np.ceil(N / buff_size))

    for i in range(num_buffers):
        start_time = time.time()
        
        start = i * buff_size
        end = min(start + buff_size, N)
        buffer = x[start:end]
        conv_buffer = fast_conv(buffer, h)
        y[start:start + len(conv_buffer)] += conv_buffer
        
        end_time = time.time()
        print(f"running conv took {end_time - start_time} seconds.")

    return y

