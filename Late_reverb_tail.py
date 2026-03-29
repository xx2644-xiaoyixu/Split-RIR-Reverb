import numpy as np
import soundfile as sf
import time
from IPython.display import display, Audio
import matplotlib.pyplot as plt
import pyroomacoustics as pra

# compute estimated RT60 by late IR
def compute_edc_rt60(h_late, fs):
    """
    Compute Energy Decay Curve (EDC) and estimate RT60 from late reverberation using Schroeder's method.

    Parameters
    ----------
    h_late : np.ndarray
        Late part of impulse response (1D array)
    fs : int or float
        Sampling rate (Hz)

    Returns
    -------
    rt60 : float
        Estimated RT60 (seconds)
    """

    # h(t) -> h^2(t), Convert waveform to energy domain
    h2 = h_late ** 2
    # E[n] = sum_{k=n}^{N-1} h^2[k] using cumsum method, which saves time from O(N**2) to O(N)
    edc = np.cumsum(h2[::-1])[::-1]
    # EDC_dB = 10 * log10(E)
    edc_db = 10 * np.log10(edc + 1e-12)
    # Normalize so the curve starts near 0 dB
    edc_db -= np.max(edc_db)

    N = len(h_late)
    t = np.arange(N) / fs

    # This is not the standard T20 range; it is a custom late-tail fitting window
    # -10~-0dB is "initial" reverb, -30~-10 is closer to "late" reverb
    mask = (edc_db <= -10) & (edc_db >= -30)

    if np.sum(mask) < 2:
        raise ValueError("Not enough points in fitting range.")

    # Fit: edc_db = slope * t + intercept <- EDCdB(t)=−kt+C
    p = np.polyfit(t[mask], edc_db[mask], 1)
    slope = p[0]
    intercept = p[1]

    # Assume attenuation is linear between 0 ~ -60 dB
    rt60 = -60.0 / slope

    return rt60

# lowpass and highpass for IR
def onepole_allpass(x, fc, fs):
    k = np.tan(np.pi * fc / fs)
    c = (k - 1) / (k + 1)
    y = np.zeros(len(x))
    y_1 = 0.0
    x_1 = 0.0
    for n in range(len(x)):
        y[n] = c * x[n] + x_1 - c * y_1
        x_1 = x[n]
        y_1 = y[n]
    return y
def onepole_lowpass(x, fc, fs):
    y = np.zeros(len(x))
    ap = onepole_allpass(x, fc, fs)
    y= 1 / 2 * (x + ap)
    return y
def onepole_highpass(x, fc, fs):
    y = np.zeros(len(x))
    ap = onepole_allpass(x, fc, fs)
    y= 1 / 2 * (x - ap)
    return y

# get damping for LPF based on the difference btw rt60_low and rt60_high
def calculate_d(low_fc, high_fc, d_min, d_max, ir_late, fs):
    # apply filter to late ir
    ir_late_low = onepole_lowpass(ir_late, low_fc, fs)
    ir_late_high = onepole_highpass(ir_late, high_fc, fs)
    # estimate rt60 for diff frequency bands
    esti_rt60_low = compute_edc_rt60(ir_late_low, fs)
    esti_rt60_high = compute_edc_rt60(ir_late_high, fs)
    
    delta_rt60 = esti_rt60_low - esti_rt60_high
    # calculate damping with clipping and exp
    d = 1 - np.exp(-delta_rt60)
    d = d_min + (d_max - d_min) * d
    
    return d

# delay line
class Delay:
    def __init__(self, max_delay):
        self.delay_length = max_delay
        self.delay_buffer = np.zeros(max_delay, dtype=float)
        self.rw_pointer = 0
        
    def next(self, in_sample):
        # delay_input = self.delay_buffer[self.rw_pointer]
        out_sample = self.delay_buffer[self.rw_pointer]
        self.delay_buffer[self.rw_pointer] = in_sample
        self.rw_pointer = (self.rw_pointer + 1) % self.delay_length
        return out_sample
    
    def tap(self, n): # read n samples before
        index = (self.rw_pointer - n) % self.delay_length
        return self.delay_buffer[index]

class LPF_Damping:
    def __init__(self, d):
        self.damping = d
        self.y_1 = 0.0

    def next(self, in_sample):
        out_sample =  self.y_1 * self.damping + in_sample * (1 -self.damping)
        self.y_1 = out_sample
        return out_sample

class IIR_Comb:
    def __init__(self, max_delay, d, n, Fs, esti_rt60):
        self.delay = Delay(max_delay)
        self.damping = LPF_Damping(d)
        self.n = n
        self.g = 10 ** ((- 3 * n) / (esti_rt60 * Fs))

    def next(self, in_sample):
        damping_sample = self.damping.next(self.delay.tap(self.n))
        out_sample = in_sample + damping_sample * self.g
        self.delay.next(out_sample)
        return out_sample
        return self.delay_buffer[index]

class APF:
    def __init__(self, g, max_delay, n):
        self.delay = Delay(max_delay)
        self.g = g
        self.n = n
    def next(self, in_sample):
        delayed_sample = self.delay.tap(self.n)
        v = in_sample + self.g * delayed_sample
        out_sample = - self.g * v + delayed_sample
        self.delay.next(v)
        return out_sample

def tail_block_processing(x, buff_size, IIR_Combs, APFs):
    N = len(x)
    num_buffers = int(np.ceil(N / buff_size))
    y = np.zeros(N, dtype=float)

    for i in range(num_buffers):
        start_time = time.time()
        
        start = i * buff_size
        end = min(start + buff_size, N)
        x_buff = x[start:end]

        y_buff = np.zeros_like(x_buff, dtype=float)
        
        for i in range(len(x_buff)):
            IIR_Comb1_out = IIR_Combs[0].next(x_buff[i])
            IIR_Comb2_out = IIR_Combs[1].next(x_buff[i])
            IIR_Comb3_out = IIR_Combs[2].next(x_buff[i])
            IIR_Comb4_out = IIR_Combs[3].next(x_buff[i])
            IIR_Comb_out = (IIR_Comb1_out + IIR_Comb2_out + IIR_Comb3_out + IIR_Comb4_out) * 0.25
            APF1_out = APFs[0].next(IIR_Comb_out)
            APF2_out = APFs[1].next(APF1_out)
    
            y_buff[i] = 0.2 * IIR_Comb_out + 0.3 * APF1_out + 0.5 * APF2_out
        
        y[start:end] = y_buff
        end_time = time.time()
        print(f"reverb tail block takes {end_time - start_time} seconds.")

    return y

# mixing early reflection and reverb tail
def mix_early_late(late_time, early, tail, early_gain, tail_gain, fs):
    late_offset = int(late_time * fs)
    tail_shifted = np.pad(tail, (late_offset, 0))
    L = max(len(early), len(tail_shifted))
    early_p = np.pad(early, (0, L - len(early)))
    tail_p  = np.pad(tail_shifted, (0, L - len(tail_shifted)))

    y = early_gain * early_p + tail_gain * tail_p
    y = y / np.max(np.abs(y))
    return y