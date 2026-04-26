import numpy as np
import time
import matplotlib.pyplot as plt

def normalization(x):
    peak = np.max(np.abs(x))
    y = x * (0.99 / peak)
    return y

def stereo_stack(x_L, x_R):    
    '''inverse of separate_channel()    
    output: 2D array, [channels, samples]    
    '''    
    y = np.column_stack((x_L, x_R))    
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    return y.T

# =============================================================
# Acoustic Simulation
# =============================================================
# compute estimated RT60 by Energy Decay Curve
def compute_edc_rt60(h, fs, fit_plt=False):
    """
    Compute Energy Decay Curve (EDC) and estimate RT60 from IR using Schroeder's method.

    Parameters
    ----------
    h : np.ndarray
        mono impulse response (1D array)
    fs : int
        Sampling rate
    fit_plt: bool
        Whether to draw the fit plot or not.

    Returns
    -------
    rt60_esti : float
        Estimated RT60 (seconds)
    edc_db: 1D array.
        Remaining energy of IR in each sample.
    """
    
    # Convert waveform to energy domain
    h2 = h ** 2
    # using Schroeder's 后向积分法 calculate Energy Decay Curve
    # sing cumsum method, O(N**2) to O(N)
    edc = np.cumsum(h2[::-1])[::-1]
    # EDC_dB = 10 * log10(E)
    edc_db = 10 * np.log10(edc + 1e-12)
    # Normalize so the curve starts at 0 dB
    edc_db -= np.max(edc_db)

    N = len(h)
    # time axis for EDC curve
    t = np.arange(N) / fs

    # selecting T30 range
    mask = (edc_db <= -5) & (edc_db >= -35)

    if np.sum(mask) < 2:
        raise ValueError("Not enough points in fitting range.")

    # Fit in linear : edc_db = slope * t + intercept <- EDCdB(t)∝−kt+C
    p = np.polyfit(t[mask], edc_db[mask], 1)
    slope = p[0]
    intercept = p[1]

    # Assume attenuation is linear between 0 ~ -60 dB
    rt60_esti = -60.0 / slope

    t_fit = t[mask]
    fit_line = slope * t_fit + intercept

    if fit_plt:
        plt.figure(figsize=(9, 5))
        plt.plot(t, edc_db, label="Full EDC", linewidth=2)
        plt.plot(t_fit, edc_db[mask], '.', label="Fit region (-5 to -35 dB)")
        plt.plot(t_fit, fit_line, '--', linewidth=2,
                 label=f"Linear fit, RT60 = {rt60_esti:.3f} s")

        plt.axhline(-5, color='gray', linestyle='--', linewidth=1)
        plt.axhline(-35, color='gray', linestyle='--', linewidth=1)

        plt.xlabel("Time (s)")
        plt.ylabel("EDC (dB)")
        plt.title("EDC and T30 Fitting Segment")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return rt60_esti, edc_db

# construct lowpass and highpass by one-pole allpass
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

# get damping for LPF based on the difference btw rt60_low and rt60_high of late ir
def calculate_d(low_fc, high_fc, d_min, d_max, ir_late, fs, esti_rt60_low, esti_rt60_high):
    # use the difference between high freq's decay and low freq's decay
    delta_rt60 = max(0.0, esti_rt60_low - esti_rt60_high)
    # calculate damping with exp. higher delta leads to higher damping
    d = 1 - np.exp(-delta_rt60)
    # clipped by d_min and d_max
    d = d_min + (d_max - d_min) * d
    
    return d


# =========================
# Algorithm for reverb tail
# =========================
class Delay:
    def __init__(self, delay_length):
        self.delay_length = max(1, int(delay_length))
        self.delay_buffer = np.zeros(self.delay_length)
        self.rw_pointer = 0
        
    def next(self, in_sample):
        out_sample = self.delay_buffer[self.rw_pointer]
        self.delay_buffer[self.rw_pointer] = in_sample
        self.rw_pointer = (self.rw_pointer + 1) % self.delay_length
        return out_sample
    
    # just in case we need to use tap method
    def tap(self, n):
        index = (self.rw_pointer - n) % self.delay_length
        return self.delay_buffer[index]

# LPF by exponential moving average
class LPF_Band:
    def __init__(self, bw):
        self.bandwidth = bw
        self.y_1 = 0.0

    def next(self, in_sample):
        out_sample = self.y_1 * (1 - self.bandwidth) + in_sample * self.bandwidth
        self.y_1 = out_sample
        return out_sample

# IIR Allpass Filter with delay d and gain g
class APF:
    def __init__(self, delay_length, g):
        self.delay_length = max(1, int(delay_length))
        self.g = g
        self.delay_buffer = np.zeros(self.delay_length)
        self.rw_pointer = 0

    def next(self, in_sample):
        d = self.delay_buffer[self.rw_pointer]
        xh = in_sample + self.g * d
        out_sample = -self.g * xh + d
        self.delay_buffer[self.rw_pointer] = xh
        self.rw_pointer = (self.rw_pointer + 1) % self.delay_length
        return out_sample

    def tap(self, n):
        idx = (self.rw_pointer - n) % self.delay_length
        return self.delay_buffer[idx]

# Modulated IIR Allpass Filter (by sine wave)
class MAPF:
    def __init__(self, d0, g, mod_depth, mod_rate, fs):
        self.base_delay = d0
        self.g = g
        self.mod_depth = mod_depth
        self.mod_rate = mod_rate
        self.fs = fs
        self.phase = 0
        self.rw_pointer = 0
        self.buffer_length = int(np.ceil(self.base_delay + self.mod_depth)) + 2 # + 2 to protect the edge for frac
        self.delay_buffer = np.zeros(self.buffer_length)

    def next(self, in_sample):
        modulated_delay = self.base_delay + self.mod_depth * np.sin(
            2 * np.pi * self.mod_rate * self.phase / self.fs  # phase[sample] is time index, t = phase / fs
        )

        k = int(np.floor(modulated_delay))
        f = modulated_delay - k

        idx1 = (self.rw_pointer - k) % self.buffer_length
        idx2 = (self.rw_pointer - k - 1) % self.buffer_length

        s1 = self.delay_buffer[idx1]
        s2 = self.delay_buffer[idx2]

        # linear interpolation
        delayed_sample = (1 - f) * s1 + f * s2

        # modulated allpass
        xh = in_sample - self.g * delayed_sample
        out_sample = delayed_sample + self.g * xh

        self.delay_buffer[self.rw_pointer] = xh
        self.rw_pointer = (self.rw_pointer + 1) % self.buffer_length
        self.phase += 1
        return out_sample

    def tap(self, n):
        idx = (self.rw_pointer - n) % self.buffer_length
        return self.delay_buffer[idx]


class LPF_Damping:
    def __init__(self, d):
        self.damping = d
        self.y_1 = 0.0

    def next(self, in_sample):
        out_sample = self.y_1 * self.damping + in_sample * (1 - self.damping)
        self.y_1 = out_sample
        return out_sample

def decay(esti_rt60, n, fs):
    '''
    Params:
    ---------------------------
    esti_rt60: float
    n: length of current delay line in FDN
    fs: sample rate

    Returns:
    ----------------------------
    g: float, decay gain of each delay line
    '''
    g= 10 ** ((-3.0 * n) / (esti_rt60 * fs))

    return g

def stereo_width(ir_L, ir_R, eps=1e-12):
    """
    Compute stereo width using Mid-Side energy ratio.

    Parameters
    ----------
    ir_L : 1D array
        Left channel impulse response.
    ir_R : 1D array
        Right channel impulse response.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    width : float
        Stereo width in [0, 1].
    """
    N = min(len(ir_L), len(ir_R))
    L = ir_L[:N]
    R = ir_R[:N]

    # energy
    E_L = np.sum(L**2)
    E_R = np.sum(R**2)

    # balance: -1 = left, 0 = center, +1 = right
    balance = (E_R - E_L) / (E_L + E_R + eps)

    # mid-side
    M = 0.5 * (L + R)
    S = 0.5 * (L - R)

    E_M = np.sum(M**2)
    E_S = np.sum(S**2)

    raw_width = E_S / (E_M + E_S + eps)

    # penalize strong left/right imbalance
    centeredness = 1 - abs(balance)

    width_centered = raw_width * centeredness

    return width_centered

# =========================
# Main processing loop
# =========================

class Stereo_FDN_Reverb:
    def __init__(self, fs, d_L, d_R, esti_rt60_L, esti_rt60_R, width):
        
        self.fs = fs
        self.esti_rt60_L = esti_rt60_L
        self.esti_rt60_R = esti_rt60_R
        self.width = width
        
        # =========================
        # Input diffusion
        # =========================
        self.Predelay_L = Delay(85)
        self.Predelay_R = Delay(90)

        self.LPF0 = LPF_Band(0.5)

        self.APF0 = APF(210, 0.75)
        self.APF1 = APF(158, 0.75)
        self.APF2 = APF(561, 0.625)
        self.APF3 = APF(410, 0.625)

        # =========================
        # Delay parameters
        # =========================
        self.mapf_dl = np.array([1343, 995, 1097, 1241])
        self.delay_dl = np.array([
            6241, 4681, 6590, 5505,
            6141, 4691, 6597, 5305
        ])
        self.apf_dl = np.array([3931, 2684, 3831, 2677])

        # =========================
        # MAPF
        # =========================
        self.MAPF0 = MAPF(self.mapf_dl[0], 0.7, 8, 0.20, self.fs)
        self.MAPF1 = MAPF(self.mapf_dl[1], 0.7, 7, 0.23, self.fs)
        self.MAPF2 = MAPF(self.mapf_dl[2], 0.8, 7, 0.25, self.fs)
        self.MAPF3 = MAPF(self.mapf_dl[3], 0.8, 8, 0.21, self.fs)

        # =========================
        # Delay lines (8 total)
        # =========================
        self.Delay0 = Delay(self.delay_dl[0])
        self.Delay1 = Delay(self.delay_dl[1])
        self.Delay2 = Delay(self.delay_dl[2])
        self.Delay3 = Delay(self.delay_dl[3])
        self.Delay4 = Delay(self.delay_dl[4])
        self.Delay5 = Delay(self.delay_dl[5])
        self.Delay6 = Delay(self.delay_dl[6])
        self.Delay7 = Delay(self.delay_dl[7])

        # =========================
        # Feedback damping (per branch)
        # =========================
        self.LPF1 = LPF_Damping(d_L)
        self.LPF2 = LPF_Damping(d_L)
        self.LPF3 = LPF_Damping(d_R)
        self.LPF4 = LPF_Damping(d_R)

        # =========================
        # APF in tank
        # =========================
        self.APF4 = APF(self.apf_dl[0], 0.5)
        self.APF5 = APF(self.apf_dl[1], 0.5)
        self.APF6 = APF(self.apf_dl[2], 0.5)
        self.APF7 = APF(self.apf_dl[3], 0.5)

        # =========================
        # Path lengths
        # =========================
        self.n0 = self.mapf_dl[0] + self.delay_dl[0]
        self.n1 = self.mapf_dl[1] + self.delay_dl[2]
        self.n2 = self.mapf_dl[2] + self.delay_dl[4]
        self.n3 = self.mapf_dl[3] + self.delay_dl[6]

        if self.width > 0.3:
            self.n0 += self.apf_dl[0]
            self.n1 += self.apf_dl[1]
            self.n2 += self.apf_dl[2]
            self.n3 += self.apf_dl[3]
        if self.width > 0.6:
            self.n0 += self.delay_dl[1]
            self.n1 += self.delay_dl[3]
            self.n2 += self.delay_dl[5]
            self.n3 += self.delay_dl[7]

        # =========================
        # Feedback gains (RT60 → decay gain)
        # =========================
        self.g0 = decay(esti_rt60_L, self.n0, self.fs)
        self.g1 = decay(esti_rt60_L, self.n1, self.fs)
        self.g2 = decay(esti_rt60_R, self.n2, self.fs)
        self.g3 = decay(esti_rt60_R, self.n3, self.fs)

        # sort by size for stable output
        gains = np.array([self.g0, self.g1, self.g2, self.g3])
        order = np.argsort(gains)
        self.left_idx = [order[0], order[3]]
        self.right_idx = [order[1], order[2]]

        # =========================
        # Feedback states (FDN state)
        # =========================
        self.v0_prev = 0.0
        self.v1_prev = 0.0
        self.v2_prev = 0.0
        self.v3_prev = 0.0

    def tail_block_processing(self, x, buff_size):

        rt60_esti = max(self.esti_rt60_L, self.esti_rt60_R)
        # pre-pad the tail by rt60
        x_p = np.concatenate([x, np.zeros(int(2.5 * rt60_esti * self.fs))])
        
        num_buffers = int(np.ceil(len(x_p) / buff_size))
        N = len(x_p)
        
        tail_L = np.zeros(N, dtype=float)
        tail_R = np.zeros(N, dtype=float)
        
        start_time = time.time()

        for i in range(num_buffers):

            start = i * buff_size
            end = min(start + buff_size, N)

            x_buff = x_p[start:end]

            y_buff_L = np.zeros_like(x_buff, dtype=float)
            y_buff_R = np.zeros_like(x_buff, dtype=float)

            for n in range(len(x_buff)):

                # =========================
                # Input diffusion (mono → L&R)
                # =========================
                l0 = self.Predelay_L.next(x_buff[n])
                l0 = self.LPF0.next(l0)
                l0 = self.APF0.next(l0)
                l0 = self.APF1.next(l0)
                l0 = self.APF2.next(l0)
                l0 = self.APF3.next(l0)

                l1 = self.Predelay_R.next(x_buff[n])
                l1 = self.LPF0.next(l1)
                l1 = self.APF0.next(l1)
                l1 = self.APF1.next(l1)
                l1 = self.APF2.next(l1)
                l1 = self.APF3.next(l1)

                # =========================
                # Input of FDN
                # =========================
                in0 =  l0
                in1 = -l0
                in2 =  l1
                in3 = -l1

                # =========================
                # 4x4 FDN (Hadamard Matric)
                # =========================
                u0 = (self.v0_prev + self.v1_prev + self.v2_prev + self.v3_prev) * 0.5 + in0
                u1 = (self.v0_prev - self.v1_prev + self.v2_prev - self.v3_prev) * 0.5 + in1
                u2 = (self.v0_prev + self.v1_prev - self.v2_prev - self.v3_prev) * 0.5 + in2
                u3 = (self.v0_prev - self.v1_prev - self.v2_prev + self.v3_prev) * 0.5 + in3

                # Tank 1
                l2 = self.MAPF0.next(u0)
                l2 = self.Delay0.next(l2)
                l2 = self.LPF1.next(l2)
                if self.width > 0.3:
                    l2 = self.APF4.next(l2)
                if self.width > 0.6:
                    l2 = self.Delay1.next(l2)
                v0 = l2 * self.g0

                # Tank 2
                l3 = self.MAPF1.next(u1)
                l3 = self.Delay2.next(l3)
                l3 = self.LPF2.next(l3)
                if self.width > 0.3:
                    l3 = self.APF5.next(l3)
                if self.width > 0.6:
                    l3 = self.Delay3.next(l3)
                v1 = l3 * self.g1

                # Tank 3
                l4 = self.MAPF2.next(u2)
                l4 = self.Delay4.next(l4)
                l4 = self.LPF3.next(l4)
                if self.width > 0.3:
                    l4 = self.APF6.next(l4)
                if self.width > 0.6:
                    l4 = self.Delay5.next(l4)
                v2 = l4 * self.g2

                # Tank 4
                l5 = self.MAPF3.next(u3)
                l5 = self.Delay6.next(l5)
                l5 = self.LPF4.next(l5)
                if self.width > 0.3:
                    l5 = self.APF7.next(l5)
                if self.width > 0.6:
                    l5 = self.Delay7.next(l5)
                v3 = l5 * self.g3

                # =========================
                # Update state
                # =========================
                self.v0_prev = v0
                self.v1_prev = v1
                self.v2_prev = v2
                self.v3_prev = v3

                # =========================
                # Output: correlation by values of gains
                # =========================
                v = [v0, v1, v2, v3]
                
                y_buff_L[n] = 0.5 * (v[self.left_idx[0]] + v[self.left_idx[1]])
                y_buff_R[n] = 0.5 * (v[self.right_idx[0]] + v[self.right_idx[1]])

            tail_L[start:end] = y_buff_L
            tail_R[start:end] = y_buff_R

        end_time = time.time()
        block_time = (end_time - start_time) / num_buffers

        print(f"Tail processing took {end_time - start_time:.6f} seconds total.")
        print(f"Average time per block: {block_time:.6f} seconds")

        return tail_L, tail_R, block_time

def trim_leading_silence_stereo(x, threshold=1e-6):
    if x.shape[0] != 2:
        raise ValueError("Input must in shape (2,N).")
    # bool value of samples whether bigger than threshold
    mask = np.max(np.abs(x), axis=0) > threshold
    if not np.any(mask):
        return x
    start_idx = np.argmax(mask)
    return x[:, start_idx:]
    
# mixing early reflection and reverb tail
def mix(t0_L, t0_R, early, tail, fs, ratio_L, ratio_R, threshold=1e-6, eps=1e-12):

    tail_cut = trim_leading_silence_stereo(tail, threshold=threshold)

    # keep early and tail in same length
    N = max(early.shape[1], tail_cut.shape[1])
    early_p  = np.pad(early, ((0, 0), (0, N - early.shape[1])))
    tail_p   = np.pad(tail_cut, ((0, 0), (0, N - tail_cut.shape[1])))

    E_early_p_L = np.sum(early_p[0] ** 2)
    E_early_p_R = np.sum(early_p[1] ** 2)
    E_tail_p_L = np.sum(tail_p[0] ** 2)
    E_tail_p_R = np.sum(tail_p[1] ** 2)

    # np.sqrt is to convert energy-domain to amplitude gain
    g_early_L = np.sqrt(ratio_L / (E_early_p_L + eps))
    g_tail_L  = np.sqrt((1 - ratio_L) / (E_tail_p_L + eps))
    g_early_R = np.sqrt(ratio_R / (E_early_p_R + eps))
    g_tail_R  = np.sqrt((1 - ratio_R) / (E_tail_p_R + eps))

    print([g_early_L, g_tail_L], [g_early_R, g_tail_R])
    
    y_L = g_early_L * early_p[0] + g_tail_L * tail_p[0]
    y_R = g_early_R * early_p[1] + g_tail_R * tail_p[1]
    y = np.stack([y_L, y_R], axis=0)

    # add arrival time of direct sound at the beginning
    predelay_L = np.zeros(int(t0_L * fs))
    predelay_R = np.zeros(int(t0_R * fs))

    y_predelayed_L = np.concatenate([predelay_L, y[0]])
    y_predelayed_R = np.concatenate([predelay_R, y[1]])
    
    # padding to keep the same length
    M = max(len(y_predelayed_L), len(y_predelayed_R))
    y_predelayed_L_p = np.pad(y_predelayed_L, (0, M - len(y_predelayed_L)))
    y_predelayed_R_p = np.pad(y_predelayed_R, (0, M - len(y_predelayed_R)))

    y_predelayed = stereo_stack(y_predelayed_L_p, y_predelayed_R_p)

    y_delayed_norm = normalization(y_predelayed)
    
    return y_delayed_norm
