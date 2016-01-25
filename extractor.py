# -*- coding: utf-8 -*-
"""
Mel extraction from WAV
"""
# Created by Anton Alekseev, December 2015
import os
import logging
import random
import struct
import matplotlib.pyplot as plt

import wave
import numpy as np
from scipy.fftpack import rfft, dct
from scipy.signal import hamming

logger = logging.getLogger(__name__)


class MelExtractor(object):
    """
    Extraction mfcc from arbitrary WAV file

    Parameters
    ----------
    :param speech_freq_max, speech_freq_min: int
        This is max, min frequency for detecting speech in frames.
    :param mel_cf: int
        Number of cepstral coefficients.
    :param recon_type: int
        Provide different type of frames division.
    :param num_filters: int
        Number of mel filters.
    :param show_sbs: bool
        If true you'll see every step in process mel extracting.
    :param dir_list: bool
        If true you'll work with directory of files, another you'll work with
        the specified file directly and depends on glob_path.
    :param glob_path: str
        It can be directory path or path to file.
    :param num_fft: int
        Number of samples after fft transform.
    :param glob_path_out: str
        This is a path to directory where you want to save mfcc array.
    """
    def __init__(self, speech_freq_max=8000, speech_freq_min=300, mel_cf=15, recon_type=3, num_filters=42,
                 show_sbs=False, dir_list=False, glob_path=None, num_fft=1024, glob_path_out=None, logger=None):
        self.speech_freq_max = speech_freq_max
        self.speech_freq_min = speech_freq_min
        self.mel_cf = mel_cf
        self.recon_type = recon_type
        self.num_filters = num_filters
        self.show_sbs = show_sbs
        self.dir_list = dir_list
        self.glob_path = glob_path
        self.num_fft = num_fft
        self.glob_path_out = glob_path_out
        self.logger = logger or logging.getLogger(__name__)

    def wave_analyze(self, path_in=None):
        """
        This block provide information about parameters of wav file and samples in it.

        Parameters
        ----------
        :param path_in: str
            You can put full path in it or path to directory.

        Returns
        -------
        :return num_channels: int
            Number of channels
        :return sampwidth: int
            Samples step.
        :return framerate: int
            Frequence of samples.
        :return num_frames: int
            Number of frames.
        :return comptype: int
            Type of compress
        :return compname: str
            Name of compress type
        :return samples: ndarray
            Samples of file.

        Examples
        --------

        >>> noise_out = wave.open('noise.wav', 'w')
        >>> noise_out.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
        >>> value = random.randint(-100, 100)
        >>> packed_value = struct.pack('h', value)
        >>> noise_out.writeframes(packed_value)
        >>> noise_out.close()
        >>> Ex=MelExtractor()
        >>> param_, samples = Ex.wave_analyze("noise.wav")
        >>> param_
        (1, 2, 44100, 1, 'NONE', 'not compressed')
        >>> len(samples)
        1

        """
        if path_in:
            types = {1: np.int8, 2: np.int16, 4: np.int32}
            wav = wave.open(path_in, mode="r")
            (num_channels, sampwidth, framerate, num_frames, comptype, compname) = wav.getparams()
            content = wav.readframes(num_frames)
            samples = np.fromstring(content, dtype=types[sampwidth])

            # There are correction for sample list if number of channels more than one.
            if num_channels != 1:
                samples = samples[::num_channels]
                num_frames = len(samples)
            self.logger.debug(u"Wav parameters is: %s %s %s %s %s %s" %
                          (num_channels, sampwidth, framerate, num_frames, comptype, compname))
            if self.show_sbs:
                plt.plot(np.linspace(0, num_frames, len(samples[::4])), samples[::4])
                plt.show()
            return (num_channels, sampwidth, framerate, num_frames, comptype, compname), samples

    @staticmethod
    def pre_emphases(samples):
        """
        Generally, pre-emphasis is performed for flattening the magnitude spectrum and
        balancing the high and low frequency components.

        Parameters
        ----------
        :param samples: ndarray or array
            Array of samples

        Returns
        -------
        :return samples: ndarray
            Array of samples
        """
        tmp = []
        for ss, x in enumerate(samples[:-1]):
            tmp.append(samples[ss+1] - 0.95*x)
        samples = np.array(tmp)
        return samples

    def frame_constructor(self, samples, param_, fix_count=30):
        """
        Remove zeros samples and return array of frames with different width.

        Parameters
        ----------
        :param samples: ndarray
            Array of samples
        :param param_: tuple
            Tuple of parameters
        :param fix_count: int
            Number of frames. Default is 30.

        Returns
        -------
        :return lst_of_frames: ndarray
            Array of samples for each frames.

        Examples
        --------

        >>> Ex=MelExtractor(recon_type=3)
        >>> lst_of_frames = Ex.frame_constructor(\
        np.random.rand(90), (1, 2, 44100, 90, 'NONE', 'not compressed'), fix_count=30)
        >>> len(lst_of_frames)
        30
        >>> len(lst_of_frames[0])
        5
        """

        if self.show_sbs:
            tmp = rfft(samples)[::4]
            tmp = tmp*hamming(len(tmp))
            self.logger.debug(u"Len samples before zeros pop %s" % len(samples))
            plt.plot(np.linspace(0, param_[2]/2, len(tmp)), tmp)
            plt.show()
        # Filtering and window function
        samples = filter(lambda x: x != 0, samples)
        samples *= hamming(len(samples))
        self.logger.debug(u"Len samples after zeros pop %s" % len(samples))

        # If recon_type is 1, we'll get fixed number of frames with adaptive width and 10ms step overflow.
        if self.recon_type == 1:
            tmp = int((len(samples) - (fix_count-1)*param_[2]*0.01))
            lst_of_frames = [samples[:tmp:]]
            samples = samples[int(0.01*param_[2])::]
            self.logger.debug(u"Frame size is - %s" % tmp)
            self.logger.debug(u"Step size is - %s" % len(samples[:int(0.01*param_[2]):]))
            for mel in xrange(1, fix_count):
                lst_of_frames = np.append(lst_of_frames, [samples[:tmp:]], axis=0)
                samples = samples[int(0.01*param_[2])::]
            self.logger.debug(u"We've got this number of frames - %s" % len(lst_of_frames))
            return lst_of_frames

        # If recon_type is 2, we'll get full number of frames 25ms width and 10ms step overflow.
        elif self.recon_type == 2:
            lst_of_frames = [samples[:int(0.025*param_[2]):]]
            samples = samples[int(0.01*param_[2])::]
            self.logger.debug(u"Frame size is - %s" % len(samples[:int(0.025*param_[2]):]))
            self.logger.debug(u"Step size is - %s" % len(samples[:int(0.01*param_[2]):]))
            for tmp in xrange(0, int(round((len(samples)-0.025*param_[2]) / (0.01*param_[2])))):
                lst_of_frames = np.append(lst_of_frames, [samples[:int(0.025*param_[2]):]], axis=0)
                samples = samples[int(0.01*param_[2])::]
            return lst_of_frames

        # If recon_type is 3, we'll get half-overlapping number of frames.
        elif self.recon_type == 3:
            tmp = int((len(samples) / (1+0.5*fix_count)))
            lst_of_frames = [samples[:tmp:]]
            samples = samples[tmp/2::]
            self.logger.debug(u"Frame size is - %s" % tmp)
            self.logger.debug(u"Step size is - %s" % len(samples[:tmp/2:]))
            for mel in xrange(1, fix_count):
                lst_of_frames = np.append(lst_of_frames, [samples[:tmp:]], axis=0)
                samples = samples[tmp/2::]
            self.logger.debug(u"We've got this number of frames - %s" % len(lst_of_frames))
            return lst_of_frames

    def bank(self, freq, smp_len):
        """
        This block involve in filtering of frame and return bank of filtered values.

        Parameters
        ----------
        :param freq: int
            Frequence of samples.
        :param smp_len: int
            Lenght of frame.

        Returns
        -------
        :return fbank: ndarray
            Array of filters.

        Examples
        --------

        >>> Ex=MelExtractor(num_filters=4)
        >>> Ex.bank(44100, 20)
        array([[ 0. ,  1. ,  0.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
               [ 0. ,  0. ,  0.5,  1. ,  0.8,  0.6,  0.4,  0.2,  0. ,  0. ,  0. ,
                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
               [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
               [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])
        """

        # Generation array of mel for num_filters
        fr2mel = np.linspace(1125*np.log(1 + self.speech_freq_min/700.0),
                             1125*np.log(1 + self.speech_freq_max/700.0),
                             self.num_filters)
        # Translate mel to freq scale
        mel2fr = 700*(np.exp(fr2mel/1125.0) - 1)
        self.logger.debug(u"Peak of filters is - %s" % mel2fr)
        if self.show_sbs:
            for x, y in enumerate(mel2fr):
                if x not in [0, self.num_filters-1]:
                    plt.plot([mel2fr[x-1], mel2fr[x], mel2fr[x+1]], [0, 1, 0])
                elif x == 0:
                    plt.plot([mel2fr[x+1] - 2*mel2fr[x], mel2fr[x], mel2fr[x+1]], [0, 1, 0])
                elif x == self.num_filters-1:
                    plt.plot([mel2fr[x-1], mel2fr[x], 2*mel2fr[x] - mel2fr[x-1]], [0, 1, 0])
            plt.show()
        lst_of_fbank = np.array([])
        for tmp in mel2fr:
            # Translate freq scale to samples scale
            lst_of_fbank = np.append(lst_of_fbank, np.round((smp_len+1) * tmp*2/freq))
        self.logger.debug(u"Number of samples for filter is - %s" % lst_of_fbank)

        # There are filtering directly
        fbank = np.zeros([self.num_filters, smp_len])
        for j in xrange(0, self.num_filters-2):
            for i in xrange(int(lst_of_fbank[j]), int(lst_of_fbank[j+1])):
                fbank[j, i] = (i-lst_of_fbank[j]) / (lst_of_fbank[j+1]-lst_of_fbank[j])
            for i in xrange(int(lst_of_fbank[j+1]), int(lst_of_fbank[j+2])):
                fbank[j, i] = (lst_of_fbank[j+2]-i) / (lst_of_fbank[j+2]-lst_of_fbank[j+1])
        return fbank

    def dc_transform(self, samples):
        """
        Just simple discrete cosine transform.

        Parameters
        ----------
        :param samples: ndarray
            Array of frames' samples.

        Returns
        -------
        :return c: ndarray
            Array of mfcc.

        """
        c = dct(samples, axis=1, norm='ortho')[:, :self.mel_cf]
        return c

    def save_file(self, samples, file_name):
        """
        Save file of numpy arrays.

        Parameters
        ----------
        :param samples: ndarray
            Array of frames' samples.
        :param file_name: str
            File name.
        """
        if self.glob_path_out:
            try:
                np.save(self.glob_path_out+str(file_name.split(".wav")[0]), samples)
            except IOError:
                self.logger.info(u'Creating new directory.')
                os.mkdir(self.glob_path_out)
                np.save(self.glob_path_out+str(file_name.split(".wav")[0]), samples)

    def viewer(self):
        """
        Example of using.
        You will see frames' images of all files in directory one after one.
        Look carefully and you'll see.
        """

        if self.dir_list and self.glob_path:
            try:
                os.listdir(self.glob_path)
            except WindowsError:
                self.logger.critical(u'Incorrect path to directory!')
            else:
                for ss, tmp in enumerate(os.listdir(self.glob_path)):
                    self.logger.info(self.glob_path+tmp)

                    try:
                        prm, smp = self.wave_analyze(self.glob_path+tmp)
                    except (wave.Error, IOError, EOFError):
                        self.logger.critical(u'Wrong type of file %s' % tmp)

                    else:
                        smp = self.pre_emphases(smp)
                        smp = self.frame_constructor(smp, prm)
                        # Fast Fourier Transform for real freq
                        smp = np.absolute(rfft(smp, self.num_fft))
                        # Find magnitude
                        smp = 1.0/self.num_fft * np.square(smp)
                        # Bank of filters' peaks
                        try:
                            fbank = self.bank(prm[2], len(smp[0]))
                        except IndexError:
                            self.logger.critical(u'Min frame rate of file should be more than 20000, %s has less.' % tmp)
                        else:
                            # Multiplication matrix
                            smp = np.dot(smp, fbank.T)
                            # Avoiding problem with 0 for logarithmic scale
                            smp = np.where(smp == 0, np.finfo(float).eps, smp)
                            smp = np.log(np.absolute(smp))
                            smp2 = np.zeros([len(smp), self.num_filters-2])
                            # Deleting empty filters
                            for x, y in enumerate(smp):
                                smp2[x] = y[:-2]
                            smp = self.dc_transform(smp2)
                            # for i in smp:
                            #     plt.plot(np.linspace(0, len(i)-1, len(i)-1), i[1:])
                            # plt.show()
                            # You can save numpy array there
                            self.save_file(samples=smp, file_name=tmp+str(ss))
        else:
            self.logger.debug(self.glob_path)
            try:
                prm, smp = self.wave_analyze(self.glob_path)
                if prm[2] < 20000:
                        self.logger.critical(u'Min frame rate of file should be more than 20000, %s has less'
                                             % self.glob_path)
                        raise IndexError(u'Min frame rate of file should be more than 20000, %s has less'
                                         % self.glob_path)
            except wave.Error:
                self.logger.critical(u'You should write correct global path for directory or WAV-file immediately')
            except IOError:
                self.logger.critical(u'You have to write full path to file!')
            else:
                smp = self.pre_emphases(smp)
                smp = self.frame_constructor(smp, prm)
                smp = np.absolute(rfft(smp, n=self.num_fft))
                smp = 1.0/self.num_fft * np.square(smp)
                fbank = self.bank(prm[2], len(smp[0]))
                smp = np.dot(smp, fbank.T)
                smp = np.where(smp == 0, np.finfo(float).eps, smp)
                smp = np.log(np.absolute(smp))
                smp_z = np.zeros([len(smp), self.num_filters-2])
                for x, y in enumerate(smp):
                    smp_z[x] = y[:-2]
                smp = self.dc_transform(smp_z)
                return smp

