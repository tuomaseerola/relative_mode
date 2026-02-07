# relative mode: A estimation of relative mode from audio.
# part of the initial code has been derived from:
# https://github.com/jackmcarthur/musical-key-finder
#
# Full documentation of this model is available from
# [https://github.com/tuomaseerola/relative_mode](https://github.com/tuomaseerola/relative_mode)
#
# Minor improvements 1.1 (January 2026, TE):
# - output of the RM_segments has onset time in seconds
# - interpolation method can be selected: 'cubic', 'linear', or 'none'
# - dynamic tick step calculation to ensure 5-7 ticks on x-axis


#### libraries ####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import math
from scipy import spatial
import heapq
from scipy.interpolate import interp1d

#### import helper functions ####

def calculate_cosine_distance(a, b):
    cosine_distance = float(spatial.distance.cosine(a, b))
    return cosine_distance

def calculate_cosine_similarity(a, b):
    cosine_similarity = 1 - calculate_cosine_distance(a, b)
    return cosine_similarity

def calculate_euclidean_distance(a, b):
    euclidean_distance = float(spatial.distance.euclidean(a, b))
#    distance = np.sqrt(np.sum(np.square(a - b)))
    return euclidean_distance

def calculate_euclidean_similarity(a, b):
    euclidean_similarity = 1 - calculate_euclidean_distance(a, b)
    return euclidean_similarity

#### main function ####

class Tonal_Fragment(object):
    def __init__(self, waveform, sr, tstart=None, tend=None, chromatype='CQT', profile='krumhansl', octaves=7, thres=0.0, bins_octave=36,wlensmooth=41, f_min=65.4, distance='pearson'):
        self.waveform = waveform
        self.sr = sr
        self.tstart = tstart
        self.tend = tend
        self.chromatype = chromatype
        self.profile = profile
        self.octaves = octaves
        self.thres = thres
        self.bins_octave = bins_octave
        self.win_len_smooth = wlensmooth
        self.distance = distance

        if self.tstart is not None:
            self.tstart = librosa.time_to_samples(self.tstart, sr=self.sr)
        if self.tend is not None:
            self.tend = librosa.time_to_samples(self.tend, sr=self.sr)
        self.y_segment = self.waveform[self.tstart:self.tend]

        if self.chromatype == 'CENS':
          #  print('CENS')
            self.chromograph = librosa.feature.chroma_cens(y=self.y_segment, sr=self.sr, fmin=f_min, n_octaves=octaves, bins_per_octave=36, hop_length=8192,win_len_smooth=wlensmooth) # , n_chroma=12, fmin=50, n_octaves=5,tuning=True
        if self.chromatype == 'STFT':
          #  print('STFT')
            self.chromograph = librosa.feature.chroma_stft(y=self.y_segment, sr=self.sr) #,n_chroma=12, fmin=50, n_octaves=5,threshold=0.1
        if self.chromatype == 'CQT':
            #print('CQT with filter')
            self.chromograph = librosa.feature.chroma_cqt(y=self.y_segment, sr=self.sr, n_octaves=octaves, threshold=thres, fmin=f_min, bins_per_octave=36,cqt_mode='hybrid',hop_length=8192) # n_chroma=12, fmin=50, n_octaves=5,threshold=0.1
            #print(octaves)
        if self.chromatype == 'CQHC':
            #print('CQHC')
            step_length = int(pow(2, int(np.ceil(np.log2(0.04 * self.sr)))) / 2)
            minimum_frequency = 65.4
            octave_resolution = 12
            self.cqt_spectrogram = cqhc.cqtspectrogram(self.y_segment, self.sr, step_length, minimum_frequency, octave_resolution)
            self.spectral_component, self.pitch_component = cqhc.cqtdeconv(self.cqt_spectrogram)
            self.CQT = np.abs(self.pitch_component)
            self.chroma_map = librosa.filters.cq_to_chroma(self.CQT.shape[0])
            self.chromograph = self.chroma_map.dot(self.CQT)
            # Max-normalize each time step (this is perhaps optional
            #print(sum(self.chromograph))
            #self.chromograph = librosa.util.normalize(self.chromograph, axis=0) # DISABLED BY TE

            #self.chromograph = librosa.feature.chroma_cqt(y=self.y_segment, sr=self.sr, n_octaves=5, threshold=0.07, fmin=65.4, bins_per_octave=36,cqt_mode='full',hop_length=8192) # n_chroma=12, fmin=50, n_octaves=5,threshold=0.1
        #self.chromograph = [i/sum(self.chromograph+1) for i in self.chromograph+1]

        # chroma_vals is the amount of each pitch class present in this time interval
        self.chroma_vals = []
        for i in range(12):
            self.chroma_vals.append(np.sum(self.chromograph[i]))
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        # dictionary relating pitch names to the associated intensity in the song
        self.keyfreqs = {pitches[i]: self.chroma_vals[i] for i in range(12)}

        keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]

        # use of the Krumhansl-Schmuckler key-finding algorithm, which compares the chroma
        # data above to typical profiles of major and minor keys:
        if self.profile == 'krumhansl':
            maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
            min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        # http://extras.humdrum.org/man/keycor/
        # Aarden-Essen continuity profiles
        if self.profile == 'aarden':
            maj_profile = [17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587, 0.291248, 22.062, 0.145624, 8.15494, 0.232998, 4.95122]
            min_profile = [18.2648, 0.737619, 14.0499, 16.8599, 0.702494, 14.4362, 0.702494, 18.6161, 4.56621, 1.93186, 7.37619, 1.75623]

        # Bellman - Budge
        if self.profile == 'bellman':
            maj_profile = [16.80, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 20.28, 1.80, 8.04, 0.62, 10.57]
            min_profile = [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49, 1.53, 0.92, 10.21]

        # Simple pitch profiles (Sapp)
        if self.profile == 'simple':
            maj_profile = [2, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 1]
            min_profile = [2, 0, 1, 1, 0, 1, 0, 2, 1, 0, 0.5, 0.5]
        # Albrecht and Shanahan profiles
        if self.profile == 'albrecht':
            maj_profile = [0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081]
            min_profile = [0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.061, 0.052]

        # Temperley-Kostka-Payne chord-based profiles
        if self.profile == 'temperley':
            maj_profile = [0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400]
            min_profile = [0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330]
        # normalise all sets of profiles
        #maj_profile = [i/sum(maj_profile) for i in maj_profile]
        #min_profile = [i/sum(min_profile) for i in min_profile]

# finds correlations between the amount of each pitch class in the time interval and the above profiles,
        # starting on each of the 12 pitches. then creates dict of the musical keys (major/minor) to the correlation
        self.min_key_corrs = []
        self.maj_key_corrs = []
        for i in range(12):
            key_test = [self.keyfreqs.get(pitches[(i + m) % 12]) for m in range(12)]
            # correlation coefficients (strengths of correlation for each key)
            if self.distance == 'pearson':
                self.maj_key_corrs.append(np.corrcoef(maj_profile, key_test)[1, 0])
                self.min_key_corrs.append(np.corrcoef(min_profile, key_test)[1, 0])
                output_weight = 3.0
            if self.distance == 'cosine':
                self.maj_key_corrs.append(calculate_cosine_similarity(maj_profile, key_test)) #  *-2+1   to make this between -1 and +1
                self.min_key_corrs.append(calculate_cosine_similarity(min_profile, key_test))
                output_weight = 6.5
            if self.distance == 'euclidean':
                maj_profile = [i/sum(maj_profile) for i in maj_profile] # normalise!
                min_profile = [i/sum(min_profile) for i in min_profile]
                key_test = [i/sum(key_test) for i in key_test]
                output_weight = 10.0

                self.maj_key_corrs.append(calculate_euclidean_similarity(maj_profile, key_test)) #  uses Euler's constant
                self.min_key_corrs.append(calculate_euclidean_similarity(min_profile, key_test))

        # names of all major and minor keys
        self.key_dict = {**{keys[i]: self.maj_key_corrs[i] for i in range(12)},
                         **{keys[i + 12]: self.min_key_corrs[i] for i in range(12)}}

        # this attribute represents the key determined by the algorithm
        self.key = max(self.key_dict, key=self.key_dict.get)
        self.bestcorr = max(self.key_dict.values())

        # TE Addition: Mode: best key in major vs best in minor
        # this attribute represents the key determined by the algorithm

        # method 1: max of major and minor
        self.max_maj = max(self.maj_key_corrs)
        self.max_min = max(self.min_key_corrs)
        self.maj_min_delta_max = (self.max_maj - self.max_min) * output_weight

        # method 2: max of parallel keys
        self.combined = np.concatenate((self.maj_key_corrs, self.min_key_corrs))
        self.overall_max = max(self.combined)  # find best key
        self.overall_max_index = self.overall_max == self.combined
        self.overall_max_indexN=np.where(self.overall_max_index)[0]

        if self.overall_max_indexN[0] > 11:  # if in minor, take major key - 12 pcs
            self.same_opposite = self.combined[self.overall_max_indexN[0] - 12]
            self.maj_min_delta_parallel = self.same_opposite - self.overall_max # NOTE: MAJOR - MINOR
        else:                                    # if in major, take minor key + 12 pcs
            self.same_opposite = self.combined[self.overall_max_indexN[0] + 12]
            self.maj_min_delta_parallel = self.overall_max - self.same_opposite # NOTE: MAJOR - MINOR

        # method 3: max relative keys
        if self.overall_max_indexN[0] < 12:   # in major, take minor + 9 pcs
            if self.overall_max_indexN[0] < 3:
                self.same_opposite = self.combined[self.overall_max_indexN[0] + 12 + 9]
            else:
                self.same_opposite = self.combined[self.overall_max_indexN[0] + 9]
            self.maj_min_delta_relative = self.overall_max - self.same_opposite
        else:                            # in minor, take major + 3 pcs
            self.same_opposite = self.combined[self.overall_max_indexN[0] - 12 + 3]
            self.maj_min_delta_relative = self.same_opposite - self.overall_max

        # method 4: two highest major - two highest minor
        self.max_maj_double = sum(heapq.nlargest(2,self.maj_key_corrs))
        self.max_min_double = sum(heapq.nlargest(2,self.min_key_corrs))
        self.maj_min_delta_max_double = self.max_maj_double - self.max_min_double
        # this attribute represents the second-best key determined by the algorithm,
        # if the correlation is close to that of the actual key determined
        self.altkey = None
        self.altbestcorr = None

        for key, corr in self.key_dict.items():
            if corr > self.bestcorr * 0.9 and corr != self.bestcorr:
                self.altkey = key
                self.altbestcorr = corr

    # prints the relative prominence of each pitch class            
    def print_chroma(self):
        self.chroma_max = max(self.chroma_vals)
        for key, chrom in self.keyfreqs.items():
            print(key, '\t', f'{chrom / self.chroma_max:5.3f}')

    # prints the correlation coefficients associated with each major/minor key
    def corr_table(self):
        for key, corr in self.key_dict.items():
            print(key, '\t', f'{corr:6.3f}')

    # printout of the key determined by the algorithm; if another key is close, that key is mentioned
    def print_key(self):
        print("likely key: ", max(self.key_dict, key=self.key_dict.get), ", correlation: ", self.bestcorr, sep='')
        if self.altkey is not None:
            print("also possible: ", self.altkey, ", correlation: ", self.altbestcorr, sep='')

    # prints a chromagram of the file, showing the intensity of each pitch class over time
    def chromagram(self, title=None):
        C = librosa.feature.chroma_cqt(y=self.waveform, sr=self.sr, bins_per_octave=12, fmin=50, n_octaves=5, threshold=0.1)
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(C, sr=self.sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1,cmap='gray_r')
        if title is None:
            plt.title('Chromagram')
        else:
            plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.show()


#### Other functions ####
# This is a meta-function that runs the tonal_fragment across the windows

def relative_mode(y, sr, winlen=3, hoplen=3, cropfirst=0, croplast=0, distance='cosine', profile='albrecht', chromatype='CENS', remove_percussive = False):
    """
    Run relative mode estimation across analysis windows. 

    Arguments:
        y: audio
        sr: sampling rate
        winlen: analysis window length in seconds
        hoplen: analysis window hop length in seconds
        cropfirst: number of seconds to crop from the beginning of the audio
        croplast: number of seconds to crop from the end of the audio
        distance: distance measure (string), either cosine, pearson, or euclidean
        profile: key profile used (string), either krumhansl, albrecht, simple, aarden
        chromatype: type of chroma extraction (string), either CQT, CENS, HPCC
        remove_percussive (bool): Whether to remove percussive elements from the audio. Default is False.
     
    Returns:
        df: relative mode scalar value (-1 to +1).
        df_segment: relative mode scalar value (-1 to +1) for each window
    """
    # Crop the audio based on cropfirst and croplast
    total_duration = librosa.get_duration(y=y, sr=sr)
    start_sample = int(cropfirst * sr)
    end_sample = int((total_duration - croplast) * sr)
    y = y[start_sample:end_sample]
    # Handle tuning
    t = librosa.estimate_tuning(y=y, sr=sr)
    y440 = librosa.effects.pitch_shift(y, sr=sr, n_steps=-t)
    # remove percusive elements if requested
    if remove_percussive:
        y440_stft = librosa.stft(y440)
        y440_stft_harmonic, y440_stft_percussive = librosa.decompose.hpss(y440_stft)
        y440 = librosa.istft(y440_stft_harmonic, length=len(y440))

    df = pd.DataFrame(columns=['tonmaxmaj', 'tonmaxmin', 'tondeltamax'])

    df_segments = pd.DataFrame(columns=['onset', 'tonmaxmaj', 'tonmaxmin', 'tonkey', 'tondeltamax'])
    frames = librosa.util.frame(y440, frame_length=int(sr * winlen), hop_length=int(sr * hoplen))
    N = int(frames.shape[1])
    for ii in range(0, N):
        ton = Tonal_Fragment(frames[:, ii], sr, distance=distance, profile=profile, chromatype=chromatype)
        # Update onset to reflect actual time in seconds
        onset_time = ii * hoplen
        df_segments.loc[len(df_segments)] = [onset_time, ton.max_maj, ton.max_min, ton.key, ton.maj_min_delta_max]

    df.loc[len(df)] = [np.mean(df_segments['tonmaxmaj']), np.mean(df_segments['tonmaxmin']),
                       np.mean(df_segments['tondeltamax'])]
    return df, df_segments


# This is to plot the results across time

#import numpy as np
#import pandas as pd
#import librosa
#import librosa.display
#import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#from src.relative_mode import Tonal_Fragment

# function to plot CME across time
def RME_across_time(filename=None, winlen=3, hoplen=3, cropfirst=0, croplast=0, chromatype='CQT', profile='krumhansl',
                    octaves=7, thres=0, bins_octave=36, wlensmooth=41, distance='pearson', plot=False, interpolation='cubic',remove_percussive=False):
    """
     Calculate the chroma movement energy (CME) of an audio signal across time and plot it.

     Parameters:
     audio_filename (str): The filename of the audio file to process.
     win_len (int): The length of the window in seconds.
     hop_len (int): The hop length between windows in seconds.
     crop_first (int): The number of seconds to crop from the beginning of the audio file.
     crop_last (int): The number of seconds to crop from the end of the audio file.
     chroma_type (str): The type of chroma representation to use.
     profile (str): The tonal profile to use.
     num_octaves (int): The number of octaves to use.
     threshold (float): The threshold value for determining the tonal center.
     bins_per_octave (int): The number of bins per octave.
     window_len_smooth (int): The length of the smoothing window.
     distance (str): The distance metric to use.
     interpolation (str): Interpolation method for time ('cubic', 'linear', or 'none'). Default is 'cubic'.
     remove_percussive (bool): Whether to remove percussive elements from the audio. Default is False.

     Returns:
     fig (matplotlib.figure.Figure): The resulting plot.
     """

    # handle file reading and cropping (if necessary)
    y, sr = librosa.load(filename)
    d = librosa.get_duration(y=y, sr=sr)  #
    y, sr = librosa.load(filename, offset=0 + cropfirst, duration=d - croplast)

    # handle tuning
    t = librosa.estimate_tuning(y=y, sr=sr)
    y440 = librosa.effects.pitch_shift(y, sr=sr, n_steps=-t)
    # remove percusive elements if requested
    if remove_percussive:
        y440_stft = librosa.stft(y440)
        y440_stft_harmonic, y440_stft_percussive = librosa.decompose.hpss(y440_stft)
        y440 = librosa.istft(y440_stft_harmonic, length=len(y440))
    # create timeline for segments
    df_segments = pd.DataFrame(columns=['onset', 'tonmaxmaj', 'tonmaxmin','tonkey', 'tondeltamax'])
    frames = librosa.util.frame(y440, frame_length=int(sr * winlen), hop_length=int(sr * hoplen))
    N = int(frames.shape[1])
    seg = np.arange(0, N) * hoplen
    for i in range(0, N):
        ton = Tonal_Fragment(frames[:, i], sr, chromatype=chromatype, profile=profile, distance=distance)
        df_segments.loc[len(df_segments)] = [seg[i], ton.max_maj, ton.max_min, ton.key, ton.maj_min_delta_max]

    # Correct the `onset` column to reflect actual frame start times
    df_segments['onset'] = np.arange(0, N) * (hoplen * sr) / sr

    #################
    # create figure
    if plot:

        # interpolate time
        if interpolation != 'none':
            f_linear = interp1d(df_segments['onset'], df_segments['tondeltamax'], kind=interpolation)
            xnew = np.linspace(0, max(df_segments['onset']), num=200, endpoint=True)
            interpolated_values = f_linear(xnew)
        else:
            xnew = df_segments['onset']
            interpolated_values = df_segments['tondeltamax']

        # Ensure `fig` and `ax` are defined before plotting
        fig, ax = plt.subplots(figsize=(9, 3.5))

        # interpolated RME
        ax.plot(xnew + winlen / 2, interpolated_values, linewidth=4, color='blue')

        # segment starts
        ax.vlines(df_segments['onset'], -0.5, 0.5, color='black', linestyle='--',
                  linewidth=1, alpha=1.0)
        ax.hlines(y=0, xmin=0, xmax=max(df_segments['onset']) + winlen, color='white', linestyle='-', linewidth=2,
                  alpha=1.0)

        ax.set(xlim=(0, max(df_segments['onset']) + winlen), ylim=(-1.00, 1.00))

        # Dynamically calculate step size for ticks to ensure 5-7 ticks
        max_onset = max(df_segments['onset']) + winlen
        num_ticks = 7  # Target number of ticks
        step = max(1, round(max_onset / num_ticks))
        ax.set_xticks(np.arange(0, max_onset, step=step))

        # Add labels to the middle of windows
        for i in range(0, len(df_segments)):
            ax.text(np.array(df_segments['onset'][i] + winlen / 2), df_segments['tondeltamax'][i] + 0.0,
                    np.array(round(df_segments['tondeltamax'][i], 2)), color='white', size=10,
                    bbox=dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(0.0, 0.0, 0.0)))
            ax.text(np.array(df_segments['onset'][i] + winlen / 2), df_segments['tondeltamax'][i] + 0.08,
                    np.array(df_segments['tonkey'][i]), color='white', size=10,
                    bbox=dict(boxstyle="round", ec=(1.0, 1.0, 1.0)))
        fig.tight_layout()
    # outputfilename = filename + '.png'
    # print(outputfilename)
    # fig.savefig(outputfilename)
    if plot:
        return ax, df_segments 
    else:
        return df_segments

