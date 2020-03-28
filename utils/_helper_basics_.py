#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
if 1 : ## Imports
    import sys, os, datetime, argparse, traceback, pprint, pdb # pdb.set_trace()
    import subprocess, itertools, importlib , math, glob, time, random, shutil, csv, statistics
    from operator import itemgetter
    import numpy as np
    import pickle
    ## Plots
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    # import seaborn 
    # import pandas as pd
    ## Audio
    import wave, python_speech_features#, pyaudio
    import librosa, librosa.display
    import scipy, scipy.io, scipy.io.wavfile, scipy.signal
    import soundfile as sf
    import audiofile as af
    import resampy

##########################################################################################################################################################
## Saving/Loading 
def dump_load_pickle(file_Name, mode, a=None):
    if mode == 'dump':
        # open the file for writing
        fileObject = open(file_Name,'wb') 
        
        # this writes the object a to the file named 'testfile'
        pickle.dump(a,fileObject, protocol=2)   
        # cPickle.dump(a,fileObject, protocol=2)   
        
        # here we close the fileObject
        fileObject.close()
        b = 'dumped '+file_Name
    elif mode == 'load':
        # we open the file for reading
        fileObject = open(file_Name,'rb')  
        
        # load the object from the file into var b
        b = pickle.load(fileObject)  
        # b = cPickle.load(fileObject)  
        
        # here we close the fileObject
        fileObject.close()
    return b

##########################################################################################################################################################
## Audio
def read_audio(filename_in, mode="audiofile", sr=None, mean_norm=False):
    ## Reading the audio
    if mode=="librosa":
        # must define sr=None to get native sampling rate
        sound, sound_fs = librosa.load(filename_in,sr=sr)
        # sound *= 2**15
    elif mode=="soundfile":
        sound, sound_fs = sf.read(filename_in)
    elif mode=="audiofile":
        sound, sound_fs = af.read(filename_in)
    else:
        print('mode:{} is incorrect should be librosa/soundfile/audiofile'.format(mode))

    ## Resampling
    if sr and sr!=sound_fs: 
        sound = resampy.resample(sound, sound_fs, sr, axis=0)
        sound_fs=sr
    
    ## Zero-mean
    if mean_norm: sound -= sound.mean()
    
    return sound, sound_fs
def write_audio(filename_out,x_in,sr,mode="soundfile"):
    """
        Assume input is in the form np.int16, with range [-2**15,2**15]
    """
    curr_x_in_dtype=x_in.dtype
    if mode == "librosa":
        assert np.max(np.abs(x_in))<=1      , '{} is out of range'.format(filename_out)
        librosa.output.write_wav(filename_out, x_in, sr)
    if mode == "soundfile":
        assert np.max(np.abs(x_in))<=1      , '{} is out of range'.format(filename_out)
        sf.write(filename_out,x_in,sr)
    else:
        print('mode:{} is incorrect should be librosa/soundfile'.format(mode))

##########################################################################################################################################################
## Audio Feature
def get_paddedx_stft(x_in, n_fft):
    pad_left = n_fft//2
    pad_right = n_fft-pad_left
    
    ## This  pads both left and right end of the wave
    x_pad = np.append(x_in,np.zeros(pad_right))
    x_pad = np.append(np.zeros(pad_left),x_pad)
    return x_pad
def wav2Mel(x_in, 
    sr_in=None, mode='librosa', 
    pad_mode=True,
    n_fft=None, hop_length=None, n_mels=None):
    x_tmp = x_in+0.
    if pad_mode: x_tmp = get_paddedx_stft(x_tmp,n_fft)
    
    x_Mel=librosa.feature.melspectrogram(
        y=x_tmp, sr=sr_in, S=None, n_fft=n_fft, 
        hop_length=hop_length, n_mels=n_mels, 
        power=2.0, htk=True, fmin=0., fmax=None, norm='slaney')
    x_LogMel=np.log10( (x_Mel**2)/n_fft + 1e-32)
    x_LogMel=(x_LogMel+32)/(np.max(x_LogMel)+32)

    return x_LogMel

##########################################################################################################################################################
## Augmentation
# def calc_power(_x_in):
#     assert len(_x_in.shape)==1, 'it should be waveform i.e. vector '
#     power_out = np.linalg.norm(_x_in)**2/len(_x_in)

#     return power_out
# def power_norm(_wav_in, _des_power, _prevent_clipping=True):

#     wav_Power   = calc_power(_wav_in)

#     ## Calc output power
#     wav_pnorm = _wav_in*(_des_power/wav_Power)**.5
#     renorm_Pow=calc_power(wav_pnorm)

#     ## To prevent clipping
#     if _prevent_clipping:
#         max_val=np.max(np.abs(wav_pnorm))
#         if max_val>1:
#             print(max_val)
#             wav_pnorm /= max_val
    
#     return wav_pnorm
