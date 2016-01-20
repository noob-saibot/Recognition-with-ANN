from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.structure import *
import pickle
import neurolab as nl
import numpy as np
import re
import os
import extractor


class ANNgenerator(object):
    def __init__(self, path_to=None, file_to=None, dir_to=None, path_out_=None, file_out=None, dir_out=None, lst_of_commands=None, lst_path=None, tr_vs_ts=50):
        self.path_to = path_to
        self.file_to = file_out
        self.dir_to = dir_to
        self.path_out_ = path_out_
        self.file_out = file_out
        self.dir_out = dir_out
        self.tr_vs_ts = tr_vs_ts
        self.lst_path = lst_path
        self.lst_of_commands = lst_of_commands

    def ext(self, path):
        example = np.load(path)
        example = example[:, 1:, ]
        example = np.resize(example, len(example[0])*len(example))
        return example

    def link(self, file):
        for com in self.lst_of_commands:
            if file.startswith(com):
                return com

    def exam(self, dc, train_com):
        put, out = [], []
        ds = SupervisedDataSet(420, 1)
        nt = buildNetwork(420, 3, 1, bias=True, hiddenclass=SigmoidLayer, outclass=SigmoidLayer)

        for way in self.lst_path:
            for i in os.listdir(way):
                result = self.ext(way+i)
                ds.addSample(result, (dc[self.link(i)],))
                put.append(result)
                out.append([dc[self.link(i)]])
        num_hid = 1
        net = nl.net.newff([[np.min(put), np.max(put)]]*420, [num_hid, 1], [nl.trans.LogSig(), nl.trans.SatLinPrm()])
        net.trainf = nl.train.train_rprop
        trainer = RPropMinusTrainer(nt, dataset=ds, verbose=False)
        error = trainer.trainUntilConvergence(maxEpochs=100, verbose=True, continueEpochs=100, validationProportion=1e-10)
        error = net.train(put, out, epochs=300, show=300, goal=1e-4, lr=1e-10)

        while error[-1] > 0.01:
            net = nl.net.newff([[np.min(put), np.max(put)]]*420, [num_hid, 1], [nl.trans.LogSig(), nl.trans.SatLinPrm()])
            net.trainf = nl.train.train_rprop
            error = net.train(put, out, epochs=300, show=300, goal=1e-4, lr=1e-10)
            num_hid += 1

        try:
            net.save('networks/%s_neurolab' % train_com)
            fl = open('networks/%s_brain' % train_com, 'w')
            pickle.dump(nt, fl)
            fl.close()
        except IOError:
            os.mkdir('networks')
            net.save('networks/%s_neurolab' % train_com)

    def train_res(self):
        self.lst_of_commands = ["back", "dark", "hight", "light", "low", "next", "stop"]
        dc = {"back": 0, "dark": 0, "hight": 0, "light": 0, "low": 0, "next": 0, "stop": 0}
        for i in self.lst_of_commands:
            print i
            dc[i] = 1
            self.lst_path = ["C:/Python27/Neural/Networks/stop/numpy30/", "C:/Python27/Neural/Networks/back/numpy30/", "C:/Python27/Neural/Networks/dark/numpy30/", "C:/Python27/Neural/Networks/hight/numpy30/", "C:/Python27/Neural/Networks/light/numpy30/", "C:/Python27/Neural/Networks/low/numpy30/", "C:/Python27/Neural/Networks/next/numpy30/"]
            self.exam(dc, i)
            dc[i] = 0

    def test_res(self):
        self.lst_of_commands = ["back", "dark", "hight", "light", "low", "next", "stop"]
        for i in self.lst_of_commands:
            print i,
            try:
                file = open('networks/%s_brain' % i, 'r')
                nt = pickle.load(file)
                file.close()
                net = nl.load('networks/%s_neurolab' % i)
                for fls in os.listdir("C:/Python27/Neural/Networks/back/numpy30/"):
                    example = self.ext("C:/Python27/Neural/Networks/back/numpy30/%s" % fls)
                    print round(net.sim([example])[0][0]),
                    print round(nt.activate(example))
            except IOError:
                print "no created networks for %s" % i

if __name__ == '__main__':
    ff = ANNgenerator()
    ff.test_res()