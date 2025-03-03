import numpy as np
import soundfile as sf
from scipy.signal import resample as scipy_resample
from time import perf_counter as pf
import random
import pydub
import math

def update_duration(method):
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self.duration_ = len(self.arr) / self.sr
        return result
    return wrapper

def t_normalized(x, sr):
    # x += 0.0000000000001
    # frac = 1/sr
    # dur = x
    # return (((dur * sr)*frac//frac))/sr
    samples = round(x * sr)  # Convert to sample count
    normalized_time = samples / sr  # Convert back to time
    return normalized_time

def rand_dur(x, y, sr=8000):
    FRACTION_SECOND = 1/8000
    dur = random.uniform(x, y)
    return ((dur * sr)*FRACTION_SECOND//FRACTION_SECOND)/sr

def pydub_m4a_read(path) -> tuple[np.ndarray, int]:
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0]. 
    Returns tuple (audio_np_array, sample_rate).
    """
    audio = pydub.AudioSegment.from_file(path, format='m4a')
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate

class Ses():
    def __init__(self, path) -> None:
        self.initial_path = path if path else None
        self.arr, self.sr = sf.read(path) if path and not path.endswith('.m4a') else (pydub_m4a_read(path) if path.endswith('.m4a') else None)
        if len(self.arr.shape) > 1: self.arr = np.mean(self.arr, axis=1)
        self.duration_ = (len(self.arr)/self.sr) if self.arr.any() and self.sr else None

    def duration(self):
        arr, sr = self.arr, self.sr
        return (len(arr)/sr)
    
    def duration_to_idx(self, t):
        t = t_normalized(t, self.sr)
        x = int((t * self.sr))
        # print(f"Duration {t} converted into the idx: {x}")
        return x
    
    def random_timestamp(self, max_=None):
        try:
            if max_ == None: max_ = self.duration_
        except ValueError as e:
            print(max_, e)
            raise ValueError(e)
        return t_normalized(random.uniform(0, max_), self.sr)
    
    def write(self, path):
        sf.write(path, self.arr, self.sr)

    @update_duration
    def audio_from_array(self, arr, sr):
        self.arr, self.sr = arr, sr
        self.duration_ = self.duration()
    
    def trimmed(self, ss, t):
        ss, t = t_normalized(ss, self.sr), t_normalized(t, self.sr)
        start_idx, stop_idx = self.duration_to_idx(ss), self.duration_to_idx(ss+t)
        if not len(self.arr) >= stop_idx:
            raise OverflowError(f"Kesilemedi. 't' değişkeni sesi aşıyor.\nss: {ss} | t: {t}")
        return SesFromArray(self.arr[start_idx:stop_idx], self.sr)
    
    @update_duration
    def trim(self, ss, t):
        t = t_normalized(t, self.sr)
        start_idx, stop_idx = self.duration_to_idx(ss), self.duration_to_idx(ss+t)
        if not len(self.arr) > stop_idx:
            raise OverflowError("Kesilemedi. 't' değişkeni sesi aşıyor.")
        copy = self.arr.copy()
        self.arr = copy[start_idx:stop_idx]

    @update_duration
    def random_trim(self, t:float, overlook=False):
        t = t_normalized(t, self.sr)
        duration_ = self.duration()
        if duration_ < t:
            if overlook:
                t = duration_
            else:
                raise OverflowError('`t` is longer than the audio. Set `overlook=True` to set `t` to self.duration_.')
        max_offset_idx = int((duration_ - t) * self.sr)
        offset_idx = random.randint(0, max_offset_idx)
        self.arr = self.arr[offset_idx:offset_idx+self.duration_to_idx(t)]

    def random_trimmed(self, t:float, overlook=False, max_out_naturally_short_ones=True):
        t = t_normalized(t, self.sr)
        edited_arr = self.arr.copy()
        duration_ = self.duration()
        if duration_ < t:
            if overlook:
                t = duration_
            else:
                if isinstance(self.initial_path, str) and max_out_naturally_short_ones:
                    return self
                raise OverflowError(f'`t` {t} is longer than the audio({self.initial_path}). Set `overlook=True` to set `t` to self.duration_.')
        max_offset_time = (self.duration_ - t)
        max_offset_idx = max_offset_time * self.sr
        max_offset_idx = int((duration_ - t) * self.sr)
        offset_idx = random.randint(0, max_offset_idx)
        new_arr = edited_arr[offset_idx:offset_idx+self.duration_to_idx(t)]
        return SesFromArray(new_arr, self.sr)

    def resampled(self, new_sr):
        resampled_arr_length = int(new_sr*self.duration())
        return SesFromArray(scipy_resample(self.arr.copy(), resampled_arr_length), new_sr)
    
    @update_duration
    def resample(self, new_sr):
        t1 = pf()
        resampled_arr_length = int(new_sr*self.duration())
        self.arr, self.sr = scipy_resample(self.arr.copy(), resampled_arr_length), new_sr
        print(f"{self.initial_path} resampled in {pf()-t1}")

    def distorted(self, factor:float=1.1, iterations:int=1, *args):
        new_arr = self.arr.copy()
        if factor > 20:
            raise IndexError("Factor can't be greater than 20.")
        # Multiplicative Noise
        for _ in range(iterations):
            random_factors = np.random.uniform(2/(2**factor), (2**factor)-1, size=new_arr.shape) if 'logarithmic' in args else np.random.uniform(2-factor, factor, size=new_arr.shape)
            new_arr *= random_factors
        return SesFromArray(new_arr, self.sr)
    
    def faded(self, in_, out_):
        in_ = t_normalized(in_, self.sr)
        out_ = t_normalized(out_, self.sr)
        # Copy the array
        new_array = self.arr.copy()
        
        # Declare fade in
        idx_in_ = self.duration_to_idx(in_)
        expanded_array = np.linspace(0, 1, idx_in_)
        new_array[:idx_in_] = new_array[:idx_in_] * expanded_array

        # Declare fade out
        idx_out_ = self.duration_to_idx(out_)
        # print(idx_out_)
        expanded_array = np.linspace(1, 0, idx_out_)
        #print(expanded_array.shape)
        new_array[-idx_out_:] = new_array[-idx_out_:] * expanded_array

        return SesFromArray(new_array, self.sr)
    
    def set_volume(self, volume:float = 1):
        return SesFromArray(self.arr.copy()*volume, self.sr)
    
    def reverbed(self, factor:float = 1):
        delay_multiplier = random.uniform(0.5, 3) * factor
        attenuation_multiplier = random.uniform(0.3, 2) * factor
        # Reverb parameters
        delay_times = [0.0002, 0.0004, 0.0008]  # Delay times in seconds
        delay_times = [delay * delay_multiplier for delay in delay_times]
        # print(delay_times)
        attenuation_factors = [0.6, 0.4, 0.3]  # Attenuation factors for each delay
        delay_times = [attnt * attenuation_multiplier for attnt in attenuation_factors]

        # Convert delay times to sample indices
        delay_samples = [int(self.sr * t) for t in delay_times]

        # Initialize the output array with the input signal
        output_wav = np.copy(self.arr.copy())

        # Add delayed and attenuated versions of the signal
        for delay, attenuation in zip(delay_samples, attenuation_factors):
            delayed_signal = np.zeros_like(self.arr.copy())
            delayed_signal[delay:] = self.arr[:-delay] * attenuation
            output_wav += delayed_signal

        # Normalize the output to prevent clipping
        output_wav = output_wav / np.max(np.abs(output_wav))

        return SesFromArray(output_wav, self.sr)
    
    def put_audio_on_top(self, audio, t=0, multipliers:tuple[float, float]=(1, 1), trim_overflow=False, trim_overflow_if_minute=False):
        '''Multiplier list should have 2 float elements. The first one for the larger and the second one for the smaller. Pass [2, 2] to put one on top rather than calculating the average. (x+y)/2 is used for each array on the overlapping segment, hence using [2, 2] as multipliers will equate to x+y
        '''
        t = t_normalized(t, self.sr)
        if self.sr != audio.sr:
            raise TypeError('Sampling rates are not matching.')
        tiny:Ses = audio
        large:Ses = SesFromArray(self.arr.copy(), self.sr)

        if t>large.duration():
            raise OverflowError('t parameter overflows both audios.')

        tiny_ = tiny

        overflowing_length = t + tiny.duration() - large.duration()
        if overflowing_length > 0:
            if not trim_overflow:
                if trim_overflow_if_minute and overflowing_length < 0.005:
                        tiny_ = tiny.trimmed(0, tiny.duration_-overflowing_length)
                else: raise OverflowError(f"{tiny.duration_} sesi, {large.duration_} sesini aştığı için({overflowing_length}sn) üste koyulamadı. `trim_over = True` yapın.")
            else:
                tiny_ = tiny.trimmed(0, tiny.duration_-overflowing_length)

        t_idx = large.duration_to_idx(t)
        try:overlapping = (large.arr[t_idx:t_idx+len(tiny_.arr)]*multipliers[0] + (tiny_.arr)*multipliers[1]) / 2
        except ValueError as e:
            print(large.initial_path, tiny_.initial_path, t, len(tiny_.arr), t_idx)
            raise e
        large.arr *= multipliers[0]
        large.arr[t_idx:t_idx+len(tiny_.arr)] = overlapping

        return SesFromArray(large.arr, self.sr)
    
    def put_audio_somewhere_random(self, audio, multipliers:tuple[float, float]=(1, 1), trim_overflow=False, trim_overflow_if_minute=True):
        '''Multiplier list should have 2 float elements. The first one for the larger and the second one for the smaller. Pass [2, 2] to put one on top rather than calculating the average. (x+y)/2 is used for each array on the overlapping segment, hence using [2, 2] as multipliers will equate to x+y
        '''
        return self.put_audio_on_top(audio, t=self.random_timestamp(self.duration_ - audio.duration_), multipliers=multipliers, trim_overflow=trim_overflow, trim_overflow_if_minute=trim_overflow_if_minute)
    
    def fragments(self, split_seconds=1):
        return [SesFromArray(self.arr.copy(), self.sr).trimmed(i, split_seconds) for i in range(0, math.floor(self.duration_), split_seconds)]
    
    def dismember_audio_randomly(self, iters, split_seconds = 1):
        if self.duration_ < split_seconds: raise OverflowError("Split seconds can't be larger than the audio duration.")
        iters = iters if iters else math.floor(self.duration_)
        trimmed = [SesFromArray(self.arr.copy(), self.sr).random_trimmed(split_seconds) for _ in range(iters)]
        return trimmed
    
    def energy(self):
        '''Returns the average energy of an audio per any signal float multiplied by 100.'''
        return np.sum(self.arr**2)/(self.duration_ * self.sr)*100
    
    def copy(self):
        return SesFromArray(self.arr.copy(), self.sr)

class SesFromArray(Ses):
    def __init__(self, arr:np.ndarray, sr) -> None:
        if arr.dtype.name != 'float64': raise TypeError(f"Array must be of type 'float64', not {arr.dtype.name}")
        self.arr, self.sr = arr, sr
        self.duration_ = self.duration()
        self.initial_path = (self.sr, self.duration_)


if __name__ == '__main__':
    pass