"""
Disko_Sound: Acoustic feature extraction and analysis for whale sound classification
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal, stats
import warnings

warnings.filterwarnings('ignore')


class Disko_Sound:
    """
    A class for extracting and analyzing acoustic features from whale sound .wav files.
    
    Features extracted include:
    - Spectrograms and visualizations
    - Frequency domain: dominant frequency, bandwidth, spectral centroid, spectral rolloff
    - Temporal: call duration, inter-call intervals, rhythm patterns
    - Energy: signal-to-noise ratio, amplitude envelope
    - Advanced: MFCCs (Mel-frequency cepstral coefficients)
    """
    
    def __init__(self, wav_file_path, sr=None):
        """
        Initialize Disko_Sound with a .wav file.
        
        Parameters
        ----------
        wav_file_path : str
            Path to the .wav file
        sr : int, optional
            Sample rate. If None, uses the file's native sample rate
        """
        self.wav_file_path = wav_file_path
        self.y, self.sr = librosa.load(wav_file_path, sr=sr)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        self.n_samples = len(self.y)
        
    # ============================================================================
    # VISUALIZATION METHODS
    # ============================================================================
    
    def plot_spectrogram(self, figsize=(14, 6), cmap='viridis', vmin=None, vmax=None,
                         freq_range=None, title=None):
        """
        Create and display a spectrogram of the audio signal.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        cmap : str
            Colormap to use
        vmin, vmax : float, optional
            Min/max values for color scaling
        freq_range : tuple, optional
            Frequency range to display (min_freq, max_freq) in Hz
        title : str, optional
            Title for the plot
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        # Compute Short-Time Fourier Transform
        D = librosa.stft(self.y)
        S_db = librosa.power_to_db(np.abs(D)**2, ref=np.max)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Display spectrogram
        img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='hz',
                                       ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Set frequency range if specified
        if freq_range:
            ax.set_ylim(freq_range)
        
        ax.set_title(title or f'Spectrogram: {self.wav_file_path}')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        return fig, ax
    
    def plot_waveform(self, figsize=(14, 4)):
        """
        Plot the waveform of the audio signal.
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=figsize)
        times = np.arange(len(self.y)) / self.sr
        ax.plot(times, self.y, lw=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Waveform: {self.wav_file_path}')
        ax.grid(True, alpha=0.3)
        return fig, ax
    
    def plot_spectrogram_and_waveform(self, figsize=(14, 10), cmap='viridis'):
        """
        Create a figure with both waveform and spectrogram.
        
        Returns
        -------
        fig, axes : matplotlib figure and axes
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Waveform
        times = np.arange(len(self.y)) / self.sr
        axes[0].plot(times, self.y, lw=0.5)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'Waveform: {self.wav_file_path}')
        axes[0].grid(True, alpha=0.3)
        
        # Spectrogram
        D = librosa.stft(self.y)
        S_db = librosa.power_to_db(np.abs(D)**2, ref=np.max)
        img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='hz',
                                       ax=axes[1], cmap=cmap)
        axes[1].set_title('Spectrogram')
        fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
        
        return fig, axes
    
    # ============================================================================
    # FREQUENCY DOMAIN FEATURES
    # ============================================================================
    
    def get_dominant_frequency(self, freq_range=None):
        """
        Extract the dominant frequency (peak frequency) of the signal.
        
        Parameters
        ----------
        freq_range : tuple, optional
            Frequency range to search within (min_freq, max_freq)
        
        Returns
        -------
        float
            Dominant frequency in Hz
        """
        # Compute power spectrum
        D = librosa.stft(self.y)
        power_spectrum = np.abs(D) ** 2
        mean_power = np.mean(power_spectrum, axis=1)
        
        # Frequency bins
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # Apply frequency range if specified
        if freq_range:
            mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            mean_power = mean_power * mask
        
        # Find peak
        dominant_freq = freqs[np.argmax(mean_power)]
        return float(dominant_freq)
    
    def get_spectral_centroid(self):
        """
        Extract the spectral centroid (center of mass of the spectrum).
        
        Returns
        -------
        float
            Mean spectral centroid in Hz
        """
        spectral_centroids = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        return float(np.mean(spectral_centroids))
    
    def get_spectral_rolloff(self, percent=0.95):
        """
        Extract the spectral rolloff frequency (frequency below which 85% of energy is contained).
        
        Parameters
        ----------
        percent : float
            Energy percent threshold (default 0.95 = 95%)
        
        Returns
        -------
        float
            Mean spectral rolloff frequency in Hz
        """
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr,
                                                            roll_percent=percent)[0]
        return float(np.mean(spectral_rolloff))
    
    def get_bandwidth(self, freq_range=None):
        """
        Extract the bandwidth of the signal.
        
        Parameters
        ----------
        freq_range : tuple, optional
            Frequency range to consider
        
        Returns
        -------
        float
            Bandwidth in Hz (defined as rolloff - lowest significant freq)
        """
        D = librosa.stft(self.y)
        power_spectrum = np.abs(D) ** 2
        mean_power = np.mean(power_spectrum, axis=1)
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # Find frequencies with significant energy (above threshold)
        threshold = np.max(mean_power) * 0.05  # 5% threshold
        significant_freqs = freqs[mean_power > threshold]
        
        if len(significant_freqs) > 0:
            bandwidth = float(np.max(significant_freqs) - np.min(significant_freqs))
        else:
            bandwidth = 0.0
        
        return bandwidth
    
    def get_frequency_range(self):
        """
        Get the frequency range containing significant energy.
        
        Returns
        -------
        tuple
            (min_frequency, max_frequency) in Hz
        """
        D = librosa.stft(self.y)
        power_spectrum = np.abs(D) ** 2
        mean_power = np.mean(power_spectrum, axis=1)
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # Find frequencies with significant energy
        threshold = np.max(mean_power) * 0.05
        significant_freqs = freqs[mean_power > threshold]
        
        if len(significant_freqs) > 0:
            return float(np.min(significant_freqs)), float(np.max(significant_freqs))
        else:
            return 0.0, 0.0
    
    # ============================================================================
    # TEMPORAL FEATURES
    # ============================================================================
    
    def get_call_duration(self):
        """
        Get the total duration of the audio signal.
        
        Returns
        -------
        float
            Duration in seconds
        """
        return self.duration
    
    def detect_calls(self, threshold_db=-40, min_duration=0.05):
        """
        Detect individual calls/segments in the signal using energy thresholding.
        
        Parameters
        ----------
        threshold_db : float
            Energy threshold in dB (relative to max)
        min_duration : float
            Minimum duration of a call in seconds
        
        Returns
        -------
        list of tuples
            List of (start_time, end_time, duration) for each detected call
        """
        # Compute spectrogram and convert to dB
        S = librosa.stft(self.y)
        S_db = librosa.power_to_db(np.abs(S)**2, ref=np.max)
        
        # Average energy across frequencies
        energy = np.mean(S_db, axis=0)
        
        # Apply threshold
        above_threshold = energy > threshold_db
        
        # Find call boundaries
        calls = []
        in_call = False
        start_frame = 0
        
        for i, is_sound in enumerate(above_threshold):
            if is_sound and not in_call:
                # Start of a call
                start_frame = i
                in_call = True
            elif not is_sound and in_call:
                # End of a call
                start_time = librosa.frames_to_time(start_frame, sr=self.sr)
                end_time = librosa.frames_to_time(i, sr=self.sr)
                duration = end_time - start_time
                
                if duration >= min_duration:
                    calls.append((start_time, end_time, duration))
                
                in_call = False
        
        # Handle case where signal ends while in a call
        if in_call:
            start_time = librosa.frames_to_time(start_frame, sr=self.sr)
            end_time = librosa.frames_to_time(len(above_threshold), sr=self.sr)
            duration = end_time - start_time
            if duration >= min_duration:
                calls.append((start_time, end_time, duration))
        
        return calls
    
    def get_inter_call_intervals(self, threshold_db=-40, min_duration=0.05):
        """
        Calculate intervals between detected calls.
        
        Returns
        -------
        list
            List of inter-call intervals in seconds
        """
        calls = self.detect_calls(threshold_db=threshold_db, min_duration=min_duration)
        
        if len(calls) < 2:
            return []
        
        intervals = []
        for i in range(len(calls) - 1):
            interval = calls[i + 1][0] - calls[i][1]
            intervals.append(interval)
        
        return intervals
    
    def get_rhythm_pattern(self, threshold_db=-40, min_duration=0.05):
        """
        Analyze the rhythm pattern (call durations and inter-call intervals).
        
        Returns
        -------
        dict
            Dictionary with rhythm statistics
        """
        calls = self.detect_calls(threshold_db=threshold_db, min_duration=min_duration)
        intervals = self.get_inter_call_intervals(threshold_db=threshold_db,
                                                  min_duration=min_duration)
        
        rhythm_stats = {
            'num_calls': len(calls),
            'call_durations': [c[2] for c in calls],
            'mean_call_duration': np.mean([c[2] for c in calls]) if calls else 0,
            'std_call_duration': np.std([c[2] for c in calls]) if calls else 0,
            'min_call_duration': np.min([c[2] for c in calls]) if calls else 0,
            'max_call_duration': np.max([c[2] for c in calls]) if calls else 0,
            'inter_call_intervals': intervals,
            'mean_interval': np.mean(intervals) if intervals else 0,
            'std_interval': np.std(intervals) if intervals else 0,
            'call_rate': len(calls) / self.duration if self.duration > 0 else 0,
        }
        
        return rhythm_stats
    
    # ============================================================================
    # ENERGY FEATURES
    # ============================================================================
    
    def get_rms_energy(self):
        """
        Extract RMS (Root Mean Square) energy of the signal.
        
        Returns
        -------
        float
            RMS energy
        """
        rms = librosa.feature.rms(y=self.y)[0]
        return float(np.mean(rms))
    
    def get_energy_envelope(self):
        """
        Extract the amplitude envelope of the signal.
        
        Returns
        -------
        ndarray
            Normalized amplitude envelope
        """
        # Compute RMS energy in frames
        rms = librosa.feature.rms(y=self.y, frame_length=2048, hop_length=512)[0]
        
        # Normalize
        rms_normalized = rms / np.max(rms) if np.max(rms) > 0 else rms
        
        return rms_normalized
    
    def get_signal_to_noise_ratio(self, noise_duration=0.5):
        """
        Estimate Signal-to-Noise Ratio (SNR).
        
        Assumes the beginning of the signal contains noise.
        
        Parameters
        ----------
        noise_duration : float
            Duration in seconds of the signal assumed to be noise (default: first 0.5s)
        
        Returns
        -------
        float
            SNR in dB
        """
        # Extract noise from beginning
        noise_samples = int(noise_duration * self.sr)
        noise = self.y[:noise_samples]
        noise_power = np.mean(noise ** 2)
        
        # Get signal power (excluding noise region)
        signal = self.y[noise_samples:]
        signal_power = np.mean(signal ** 2)
        
        # Calculate SNR
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = np.inf
        
        return float(snr_db)
    
    def get_zero_crossing_rate(self):
        """
        Extract Zero Crossing Rate (ZCR) - useful for distinguishing voiced/unvoiced segments.
        
        Returns
        -------
        float
            Mean zero crossing rate
        """
        zcr = librosa.feature.zero_crossing_rate(self.y)[0]
        return float(np.mean(zcr))
    
    # ============================================================================
    # ADVANCED FEATURES - MFCCs
    # ============================================================================
    
    def get_mfcc(self, n_mfcc=13):
        """
        Extract Mel-Frequency Cepstral Coefficients (MFCCs).
        
        MFCCs are highly effective for audio classification and capture
        the perceptually relevant characteristics of sounds.
        
        Parameters
        ----------
        n_mfcc : int
            Number of MFCCs to extract (default: 13)
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'mfcc_features': (n_mfcc, T) array of MFCC values
            - 'mfcc_mean': mean MFCC across time
            - 'mfcc_std': standard deviation across time
            - 'mfcc_delta_mean': mean first derivative
            - 'mfcc_delta_std': std of first derivative
        """
        # Compute MFCCs
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=n_mfcc)
        
        # Compute derivatives (delta features)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        result = {
            'mfcc_features': mfcc,
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_std': np.std(mfcc, axis=1),
            'mfcc_delta_mean': np.mean(mfcc_delta, axis=1),
            'mfcc_delta_std': np.std(mfcc_delta, axis=1),
        }
        
        return result
    
    # ============================================================================
    # COMPREHENSIVE FEATURE EXTRACTION
    # ============================================================================
    
    def extract_all_features(self, n_mfcc=13):
        """
        Extract all available acoustic features.
        
        Parameters
        ----------
        n_mfcc : int
            Number of MFCCs to extract
        
        Returns
        -------
        dict
            Dictionary containing all extracted features
        """
        # Frequency domain features
        freq_range = self.get_frequency_range()
        
        # Temporal features
        rhythm = self.get_rhythm_pattern()
        
        # MFCC features
        mfcc_data = self.get_mfcc(n_mfcc=n_mfcc)
        
        all_features = {
            # File info
            'file': self.wav_file_path,
            'duration': self.duration,
            'sample_rate': self.sr,
            
            # Frequency domain
            'dominant_frequency': self.get_dominant_frequency(),
            'spectral_centroid': self.get_spectral_centroid(),
            'spectral_rolloff': self.get_spectral_rolloff(),
            'bandwidth': self.get_bandwidth(),
            'frequency_range_min': freq_range[0],
            'frequency_range_max': freq_range[1],
            
            # Temporal
            'num_calls': rhythm['num_calls'],
            'mean_call_duration': rhythm['mean_call_duration'],
            'std_call_duration': rhythm['std_call_duration'],
            'mean_inter_call_interval': rhythm['mean_interval'],
            'std_inter_call_interval': rhythm['std_interval'],
            'call_rate': rhythm['call_rate'],
            
            # Energy
            'rms_energy': self.get_rms_energy(),
            'snr_db': self.get_signal_to_noise_ratio(),
            'zero_crossing_rate': self.get_zero_crossing_rate(),
            
            # MFCCs
            'mfcc_mean': mfcc_data['mfcc_mean'],
            'mfcc_std': mfcc_data['mfcc_std'],
            'mfcc_delta_mean': mfcc_data['mfcc_delta_mean'],
            'mfcc_delta_std': mfcc_data['mfcc_delta_std'],
        }
        
        return all_features
    
    def print_feature_summary(self, n_mfcc=13):
        """
        Print a comprehensive summary of all extracted features.
        """
        features = self.extract_all_features(n_mfcc=n_mfcc)
        
        print(f"\n{'='*70}")
        print(f"ACOUSTIC ANALYSIS SUMMARY: {self.wav_file_path}")
        print(f"{'='*70}\n")
        
        print(f"FILE INFORMATION")
        print(f"  Duration: {features['duration']:.3f} seconds")
        print(f"  Sample Rate: {features['sample_rate']} Hz\n")
        
        print(f"FREQUENCY DOMAIN")
        print(f"  Dominant Frequency: {features['dominant_frequency']:.1f} Hz")
        print(f"  Spectral Centroid: {features['spectral_centroid']:.1f} Hz")
        print(f"  Spectral Rolloff: {features['spectral_rolloff']:.1f} Hz")
        print(f"  Bandwidth: {features['bandwidth']:.1f} Hz")
        print(f"  Frequency Range: {features['frequency_range_min']:.1f} - {features['frequency_range_max']:.1f} Hz\n")
        
        print(f"TEMPORAL CHARACTERISTICS")
        print(f"  Number of Calls: {features['num_calls']}")
        print(f"  Mean Call Duration: {features['mean_call_duration']:.3f} ± {features['std_call_duration']:.3f} seconds")
        print(f"  Mean Inter-call Interval: {features['mean_inter_call_interval']:.3f} ± {features['std_inter_call_interval']:.3f} seconds")
        print(f"  Call Rate: {features['call_rate']:.2f} calls/second\n")
        
        print(f"ENERGY CHARACTERISTICS")
        print(f"  RMS Energy: {features['rms_energy']:.4f}")
        print(f"  Signal-to-Noise Ratio: {features['snr_db']:.2f} dB")
        print(f"  Zero Crossing Rate: {features['zero_crossing_rate']:.4f}\n")
        
        print(f"MFCC STATISTICS")
        print(f"  Number of Coefficients: {len(features['mfcc_mean'])}")
        print(f"  Mean MFCC values: {features['mfcc_mean']}")
        print(f"  Std MFCC values: {features['mfcc_std']}\n")
        
        print(f"{'='*70}\n")


# ============================================================================
# UTILITY FUNCTION: Batch Analysis
# ============================================================================

def analyze_sample_directory(directory_path, n_mfcc=13):
    """
    Analyze all .wav files in a directory and return feature summaries.
    
    Parameters
    ----------
    directory_path : str
        Path to directory containing .wav files
    n_mfcc : int
        Number of MFCCs to extract
    
    Returns
    -------
    list of dict
        List of feature dictionaries for each file
    """
    import os
    
    wav_files = [f for f in os.listdir(directory_path) if f.endswith('.wav')]
    results = []
    
    for wav_file in wav_files:
        file_path = os.path.join(directory_path, wav_file)
        try:
            analyzer = Disko_Sound(file_path)
            features = analyzer.extract_all_features(n_mfcc=n_mfcc)
            results.append(features)
            print(f" Analyzed: {wav_file}")
        except Exception as e:
            print(f" Failed to analyze {wav_file}: {e}")
    
    return results
