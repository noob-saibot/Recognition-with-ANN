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
    def __init__(self, lst_of_commands=None, exam_path=None):
        self.exam_path = exam_path
        self.lst_of_commands = lst_of_commands

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
        """
        if isinstance(inform, str):
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

        :param file_name: str
            Name of file.

        Returns
        -------
        :return com: str
            Command in file.

        """
        for com in self.lst_of_commands:
            if file_name.startswith(com):
                return com
        logging.critical(u"We can't recognize what the command in this file.")
        return None

    def exam(self, dc, train_com, train_path):
        """
        Here you can train your networks.

        :param dc: dict
            Dict of commands with values.
        :param train_com:
            Command what you want teach by ann to recognize.
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
                    logging.debug(u'File was added to training list %s' % i)
                    result = self.ext_t(way+i)
                    ds.addSample(result, (dc[lk],))
                    put.append(result)
                    out.append([dc[lk]])

        net = nl.net.newff([[np.min(put), np.max(put)]]*420, [num_hid, 1], [nl.trans.LogSig(), nl.trans.SatLinPrm()])
        net.trainf = nl.train.train_rprop
        trainer = RPropMinusTrainer(nt, dataset=ds, verbose=False)
        trainer.trainUntilConvergence(maxEpochs=100, verbose=False, continueEpochs=100, validationProportion=1e-10)
        logging.debug(u'Training brain...')
        error = net.train(put, out, epochs=500, show=300, goal=1e-4, lr=1e-10)
        logging.debug(u'Training neural...')

        while error[-1] > 1e-3:
            logging.debug(u'Try to one more training, because MSE are little not enough!')
            net = nl.net.newff([[np.min(put), np.max(put)]]*420, [num_hid, 1], [nl.trans.LogSig(), nl.trans.SatLinPrm()])
            net.trainf = nl.train.train_rprop
            error = net.train(put, out, epochs=500, show=300, goal=1e-4, lr=1e-10)
            logging.debug(u'Training neural...')
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

        :param train_path: str or list
            Path to examples

        :return:
            File with network.
        """
        dc = {"back": 0, "dark": 0, "hight": 0, "light": 0, "low": 0, "next": 0, "stop": 0}
        for i in self.lst_of_commands:
            logging.debug(u'File is', i)
            dc[i] = 1
            self.exam(dc, i, train_path)
            dc[i] = 0

    def test_res(self):
        """
        Provide test of created ann for examples.

        :return:
            Return statistic of tests.
        """
        prc_sum, nm_sum = 0.0, 0.0
        for fls in os.listdir(u"C:/Python27/Neural/me/"):
            nm_sum += 1
            Ex = extractor.MelExtractor(glob_path=u"C:/Python27/Neural/me/%s" % fls, dir_list=False)
            ext_res = Ex.viewer()

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
                    logging.critical(u"no created networks for ", i)
            if prc_n/nm_n*100 == 100.0:
                logging.debug(u'Word was recognized by Neurolab')
                prc_sum += 1
            else:
                logging.debug(u'Neurolab', prc_n/nm_n*100)
            if prc_b/nm_b*100 == 100.0:
                logging.debug(u'Word was recognized by PyBrain')
            else:
                logging.debug(u'PyBrain', prc_b/nm_b*100)

            logging.debug(u'Result is', prc_sum/nm_sum*100)

if __name__ == '__main__':
    logging.basicConfig(format=u'%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s',
                        level=logging.DEBUG)
    commands = [u"back", u"dark", u"hight", u"light", u"low", u"next", u"stop"]
    path = [u"C:/Python27/Neural/tt2/", u"C:/Python27/Neural/Networks/stop/numpy30/",
            u"C:/Python27/Neural/Networks/back/numpy30/", u"C:/Python27/Neural/Networks/dark/numpy30/",
            u"C:/Python27/Neural/Networks/hight/numpy30/", u"C:/Python27/Neural/Networks/light/numpy30/",
            u"C:/Python27/Neural/Networks/low/numpy30/", u"C:/Python27/Neural/Networks/next/numpy30/"]
    ff = AnnGenerator(lst_of_commands=commands)
    # ff.train_res(path)
    ff.test_res()