# -*- coding: utf-8 -*-
"""
Generating and training ANN
"""
# Created by Anton Alekseev, January 2016
import os
import pickle
import logging

import extractor
import numpy as np
import neurolab as nl
from pybrain.structure import SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import RPropMinusTrainer

logger = logging.getLogger(__name__)


class AnnGenerator(object):
    """
    Creating Ann based on neurolab and pybrain library. Training on examples and test it.

    Parameters
    ----------
    :param lst_of_commands: list
        This is list of commands what need be recognized.
    :param exam_path: list
        This is paths to folders where will be examples.

    """
    def __init__(self, lst_of_commands=None, exam_path=None, logger=None):
        self.exam_path = exam_path
        self.lst_of_commands = lst_of_commands
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def ext_t(inform):
        """
        There are extracting mfcc list from array of numpy arrays or from file.

        Parameters
        ----------
        :param inform: ndarray or str
            Path to file with array or array immediately.

        Returns
        -------
        :return example: ndarray
            Lonely array of mfcc.

        Examples
        --------

        >>> pron = AnnGenerator()
        >>> pron.ext_t(np.zeros([5,5]))
        array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                0.,  0.,  0.,  0.,  0.,  0.,  0.])
        >>> pron.ext_t(list([1,2,3]))
        """
        if isinstance(inform, unicode) or isinstance(inform, str):
            example = np.load(inform)
        elif isinstance(inform, np.ndarray):
            example = inform
        else:
            return None
        example = example[:, 1:, ]
        example = np.resize(example, len(example[0])*len(example))
        return example

    def link(self, file_name):
        """
        Extracting command from filename.

        Parameters
        ----------
        :param file_name: str
            Name of file.

        Returns
        -------
        :return com: str
            Command in file.

        Examples
        --------
        >>> pron = AnnGenerator(lst_of_commands=[u"test1",u"test2",u"test3"])
        >>> os.mkdir(u"tmp")
        >>> f = open(u"tmp/test1.wav", "w")
        >>> f.write("test")
        >>> f.close()
        >>> pron.link(u"test1")
        u'test1'
        >>> os.remove(u"tmp/test1.wav")
        >>> os.rmdir(u"tmp")
        """
        for com in self.lst_of_commands:
            if file_name.startswith(com):
                return com
        self.logger.critical(u"We can't recognize what the command in this file.")
        return None

    def exam(self, dc, train_com, train_path):
        """
        Here you can train your networks.

        Parameters
        ----------
        :param dc: dict
            Dict of commands with values.
        :param train_com:
            Command what you want teach by ann to recognize.
        :param train_path:
            Path to folder with train examples.

        Returns
        -------
        :return:
            File with network.

        """
        num_hid = 1
        put, out = [], []

        ds = SupervisedDataSet(420, 1)
        nt = buildNetwork(420, 3, 1, bias=True, hiddenclass=SigmoidLayer, outclass=SigmoidLayer)

        for way in train_path:
            for i in os.listdir(way):
                lk = self.link(i)
                if lk:
                    self.logger.debug(u'File was added to training list %s' % i)
                    result = self.ext_t(way+i)
                    ds.addSample(result, (dc[lk],))
                    put.append(result)
                    out.append([dc[lk]])

        net = nl.net.newff([[np.min(put), np.max(put)]]*420, [num_hid, 1], [nl.trans.LogSig(), nl.trans.SatLinPrm()])
        net.trainf = nl.train.train_rprop
        trainer = RPropMinusTrainer(nt, dataset=ds, verbose=False)
        self.logger.info(u'Training brain...')
        trainer.trainUntilConvergence(maxEpochs=100, verbose=False, continueEpochs=100, validationProportion=1e-7)
        self.logger.info(u'Training neural...')
        error = net.train(put, out, epochs=500, show=500, goal=1e-4, lr=1e-10)

        while error[-1] > 1e-3:
            self.logger.info(u'Try to one more training, because MSE are little not enough!')
            net = nl.net.newff([[np.min(put), np.max(put)]]*420, [num_hid, 1], [nl.trans.LogSig(), nl.trans.SatLinPrm()])
            net.trainf = nl.train.train_rprop
            self.logger.info(u'Training neural...')
            error = net.train(put, out, epochs=500, show=500, goal=1e-4, lr=1e-10)
            num_hid += 1

        try:
            net.save(u'networks/%s_neurolab' % train_com)
            fl = open(u'networks/%s_brain' % train_com, 'w')
            pickle.dump(nt, fl)
            fl.close()

        except IOError:
            os.mkdir(u'networks')
            net.save(u'networks/%s_neurolab' % train_com)

    def train_res(self, train_path):
        """
        Involve exam module to creating and training ann

        Parameters
        ----------
        :param train_path: str or list
            Path to examples

        Returns
        -------
        :return:
            File with network.
        """
        dc = {}
        for cm in self.lst_of_commands:
            dc[cm] = 0
        for i in self.lst_of_commands:
            self.logger.info(u'Command is %s' % i)
            dc[i] = 1
            self.exam(dc, i, train_path)
            dc[i] = 0

    def test_res(self, path_for_testing):
        """
        Provide test of created ann for examples.

        Parameters
        ----------
        :param path_for_testing: str or unicode
            Path to files for testing.

        Returns
        -------
        :return:
            Return statistic of tests.
        """
        prc_sum, nm_sum = 0.0, 0.0
        for fls in os.listdir(path_for_testing):
            self.logger.info(u"Word %s" % fls)
            nm_sum += 1
            pron = extractor.MelExtractor(glob_path=path_for_testing+u"%s" % fls)
            ext_res = pron.viewer()

            prc_n, nm_n, prc_b, nm_b = 0.0, 0.0, 0.0, 0.0
            for i in self.lst_of_commands:
                try:
                    tmp = open(u'networks/%s_brain' % i)
                    nt = pickle.load(tmp)
                    tmp.close()
                    net = nl.load(u'networks/%s_neurolab' % i)

                    nm_n += 1
                    nm_b += 1
                    example = self.ext_t(ext_res)
                    if fls.startswith(i):
                        if round(net.sim([example])[0][0]) == 1.0:
                            prc_n += 1
                        if round(nt.activate(example)) == 1.0:
                            prc_b += 1
                    else:
                        if round(net.sim([example])[0][0]) == 0.0:
                            prc_n += 1
                        if round(nt.activate(example)) == 0.0:
                            prc_b += 1

                except IOError:
                    self.logger.critical(u"no created networks for %s" % i)
            if prc_n/nm_n*100 == 100.0:
                self.logger.info(u'Word was recognized by Neurolab')
                prc_sum += 1
            else:
                self.logger.debug(u'Neurolab %s' % unicode(prc_n/nm_n*100))
            if prc_b/nm_b*100 == 100.0:
                self.logger.info(u'Word was recognized by PyBrain')
            else:
                self.logger.debug(u'PyBrain %s' % unicode(prc_b/nm_b*100))

            self.logger.info(u'Result is %s' % unicode(prc_sum/nm_sum*100))
        print u"Result of recognition by Neurolab is %s percents" % unicode(prc_sum/nm_sum*100)

