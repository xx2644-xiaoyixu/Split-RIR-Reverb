import numpy as np
import time
from scipy.signal import resample_poly

# =================================================
# utility
# =================================================
def resample(sr, fs, x):
    ''' change sample rate to fs 

    Params
    --------------------
    sr: sample rate of input signal
    x : (np.ndarray)
        input signal
    fs: system sample rate

    Returns
    --------------------
    y: (np.ndarray)
       output signal
    '''
    if sr != fs:
        y = resample_poly(x, fs, sr)
    else:
        y = x

    return y

def separate_channel(x):
    '''separate stereo audio to mono
    input: 2D array, [channels, samples]
    '''
    if x.shape[0] != 2:
        raise ValueError("x is not stereo input.")
    x_L = x[0, :]
    x_R = x[1, :]

    return x_L, x_R
        
def stereo_stack(x_L, x_R):
    '''inverse of separate_channel()
    output: 2D array, [channels, samples]
    '''
    y = np.column_stack((x_L, x_R))
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return y.T

def normalization(x):
    peak = np.max(np.abs(x))
    y = x * (0.99 / peak)
    return y

# ===================================================
# Acoustic Simulation
# ===================================================
    
# calculate arrival time of direct sound
def estimate_t0(x, fs, win=32, noise_ms=3.0, k=6.0):
    '''
    Estimate the time (seconds) of direct sound.

    Params
    ------------------------------------------
    x: (np.ndarray) 1D array, input ir signal
    fs: sample rate
    win: window size, must be small for precise locating
    noise_ms: assumed duration of noise before direct sound
    k: multiply coefficient, at least k times greater than mean noise level can be seen as the direct sound

    Returns:
    ------------------------------------------
    t0: arrival time of direct sound
    '''
    
    # normalize x
    x_norm = x / (np.max(np.abs(x)) + 1e-12)

    def moving_rms_win(x, win):
        ''' get the RMS of each window of input sig with hop size = 1, sliding across the whole sig

        Params:
        ----------------------------------------------------
        x: 1D array, input sig
        win: window size

        Returns:
        ----------------------------------------------------
        rms: (np.ndarray) rms of each windowed sig
        '''
        # the whole detection process is under energy domain
        energy = 0
        x2 = x ** 2
        # protect edge if len(x) < win
        win = min(win, len(x))
        # using valid mode of windowing method++
        rms = np.zeros(len(x) - win + 1)
        
        for i in range(len(x)):
            # get sum energy of windowed sig with hop size = 1
            energy += x2[i]
            # if i exceeds window size, each time slide in 1 sample at the end, detach the first sample in window
            if i >= win:
                energy -= x2[i-win]
            # if the window is full, calculate mean energy of samples in each window
            if i >= win-1:
                rms[i-win+1] = np.sqrt(max(energy / win, 0.0))
        
        return rms

    rms_x_win = moving_rms_win(x_norm, win)

    # estimate 
    noise_len = int(noise_ms * 1e-3 * fs)
    # to protect the edge
    noise_len = min(noise_len, len(x))
    # take the first noise_len samples of sig as noise
    noise = x[:noise_len]
    
    rms_noise = moving_rms_win(noise, win)
    # calculate mean rms of noise and standard deviation
    mu = np.mean(rms_noise)
    sigma = np.std(rms_noise)
    # threshold of detected direct sound
    T = mu + k * sigma

    # get sample idx of all rms of the whole sig and save the first one
    idx = np.where(rms_x_win > T)[0]
    n0 = int(idx[0]) if len(idx) > 0 else 0
    n0_center = n0 + win // 2
    t0 = n0_center / fs

    return t0

def estimate_ER_time(x, fs, t0, win_ms=30, hop_ms=3):
    '''
    Estimate the time of early reflection (excludes direction sound) using sliding window kurtosis method.

    Params
    ---------------------------------------------------
    x: 1D array, input whole IR signal.
    fs: sample rate.
    t0: time of direction sound (seconds).
    win_ms: window size for sliding through the whole sig.
    hop_ms: hop size between each window.

    Returns
    -----------------------------------------------------
    t: timeline (seconds) for kurt function of early reflection.
    kurt_smooth: smooth Kurtosis curve. 
                 First point the absolute value of slope achieves the maximum or the curve approaches zero.
    t_early: estimated duration of early reflection.
    t_split: estimated split time (includes direct sound).
    '''
    # remove direct_ir
    s0 = int(t0 * fs)
    x = x[s0:]
    # Normalize input sig
    x = x / (np.max(np.abs(x)) + 1e-12)

    # prepare for sliding windowing method
    win_len = min(len(x), int(win_ms * 1e-3 * fs))
    hop_len = min(win_len, int(hop_ms * 1e-3 *fs))
    num_frames = len(x) // hop_len

    win = np.zeros(num_frames)
    mu = np.zeros(num_frames)
    sigma = np.zeros(num_frames)
    kurt = np.zeros(num_frames)
    t = np.zeros(num_frames)
    
    # calculate kurt function
    for i in range(num_frames):
        win = x[i: i + win_len]
        mu[i] = np.mean(win)
        sigma[i] = max(0.0, np.std(win))
        kurt[i] = np.mean((win - mu[i]) ** 4) / (sigma[i] ** 4) - 3

        # create time axis: each at the center of each window
        t[i] = ((i + win_len) // 2) / fs

    # exp moving average for smoothness. lower alpha leads to soomther curve
    def exp_moving_average(x, alpha=0.2):
        y = np.zeros_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
        return y

    kurt_smooth = exp_moving_average(kurt, alpha=0.2)

    # using the sharpest transition as split point
    slope = np.diff(kurt_smooth)
    idx_1 = np.argmin(slope) # cuz slope is negative

    t_early_1 = t[idx_1 + 25] # plus 10 is for avoiding losing edge

    # using the first point when kurt curve approaches min threshold
    # use the last 20% of kurt curve as basic noise
    tail = kurt_smooth[int(0.8 * len(kurt_smooth)):]
    k_base = np.mean(tail)
    k_std  = np.std(tail)
    threshold = k_base + 1.5 * k_std

    idx_2 = np.where(kurt_smooth < threshold)[0][0]
    t_early_2 = t[idx_2 + 25]

    t_early = min(t_early_1, t_early_2)
    # final split time, add 20ms to keep more transition.
    t_split = t_early + t0 + 20 * 1e-3
        
    return t, kurt_smooth, t_early, t_split

# split full IR into ir_direct, ir_early and ir_late using cross-fade method
def split(time_direct, time_split, fade_time, h, fs):
    '''split full IR into ir_direct, ir_early and ir_late using cross-fade method

    Params
    -----------------------------------------------------
    time_direct: arrival time of direct sound (seconds).
    time_split: split time between ER (include direct) and Late reverb (seconds).
    fade_time: fade in/out time between ER and Late (seconds).
    h: 1D array, full IR.
    fs: sample rate.

    Returns
    -----------------------------------------------------
    early: early ir (include direct sound but without the silence at the beginning).
    late: late ir
    '''

    # calculate split sample, with overlap and fade method
    split_sample_direct = int(fs * time_direct)
    split_sample_early = int(fs * time_split) # fade out point for early
    fade_len = int(fade_time * fs)
    split_sample_late = max(split_sample_direct, (split_sample_early - fade_len)) # fade in point for late

    # linear crossfade
    fade_out = np.linspace(1.0, 0.0, fade_len)
    fade_in  = 1 - fade_out

    # split by fade in and fade out point
    early = np.copy(h[split_sample_direct:split_sample_early])
    late  = np.copy(h[split_sample_late:]) 
    
    # apply fade to early and late
    if fade_len > 0:
        if h.ndim == 1:
            early[-fade_len:] *= fade_out
            late[:fade_len]   *= fade_in
        else:
            early[-fade_len:] *= fade_out[:, None]
            late[:fade_len]   *= fade_in[:, None]
    
    return early, late

# =================================================================
# Convolution
# =================================================================
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

    X = np.fft.rfft(x_padded)
    H = np.fft.rfft(h_padded)
    
    Y = X * H
    y = np.real(np.fft.irfft(Y))

    return y

# Block Processing RFFT Conv
def running_convolver_stereo(x, h_L, h_R, buff_size):
    '''Convolves 1-dimensional signals with stereo IR by block processing.
    
    Parameters
    ----------
    x : (np.ndarray)
        1D array, the input signal.
        
    h_L : (np.ndarray)
        1D array, left channel of ir
    h_R: (np.ndarray)
        1D array, right channel of ir
    buff_size: int, block processing size (samples).
    
    Returns
    -------
    y : (np.ndarray)
        2D array, resulting stereo signal from convolving x and h.
    '''

    start_time_total = time.time()

    # force mono input
    if x.ndim != 1:
        raise ValueError("x must be mono, shape (N,)")
    if h_L.ndim != 1:
        raise ValueError("h_L must be mono, shape (N,)")
    if h_R.ndim != 1:
        raise ValueError("h_R must be mono, shape (N,)")
    # to make sure two split early ir are in same length
    if len(h_L) != len(h_R):
        L = max(len(h_L), len(h_R))
        h_L = np.pad(h_L, (0, L - len(h_L)))
        h_R = np.pad(h_R, (0, L - len(h_R)))

    N = len(x)
    K = len(h_L)

    y_L = np.zeros((N + K - 1), dtype=float)
    y_R = np.zeros((N + K - 1), dtype=float)

    num_buffers = int(np.ceil(N / buff_size))

    for i in range(num_buffers):
        start = i * buff_size
        end = min(start + buff_size, N)
        buffer = x[start:end]

        conv_L = fast_conv(buffer, h_L)
        conv_R = fast_conv(buffer, h_R)

        y_L[start:start + len(conv_L)] += conv_L
        y_R[start:start + len(conv_R)] += conv_R

    y = stereo_stack(y_L, y_R)
    y_norm = normalization(y)

    end_time_total = time.time()
    block_time = (end_time_total - start_time_total) / num_buffers
    print(f"running conv took {end_time_total - start_time_total:.6f} seconds total.")
    print(f"average time per block: {block_time:.6f} seconds")

    return y_norm, block_time

    
