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

    def exam(self, dc):
        put, out = [], []
        ds = SupervisedDataSet(420, 1)
        nt = buildNetwork(420, 3, 1, bias=True, hiddenclass=SigmoidLayer, outclass=SigmoidLayer)

        for way in self.lst_path:
            print way
            for i in os.listdir(way):
                print way+i, dc[self.link(i)]
                result = self.ext(way+i)
                ds.addSample(result, (dc[self.link(i)],))
                put.append(result)
                out.append([dc[self.link(i)]])

        net = nl.net.newff([[np.min(put), np.max(put)]]*420, [1, 1], [nl.trans.LogSig(), nl.trans.SatLinPrm()])
        net.trainf = nl.train.train_rprop
        print len(put), len(out), out
        trainer = RPropMinusTrainer(nt, dataset=ds, verbose=True)
        trainer.trainUntilConvergence(maxEpochs=20, verbose=None, continueEpochs=100, validationProportion=1e-10)
        net.train(put, out, epochs=10000, show=10, goal=1e-4, lr=1e-10)

    def show_res(self):
        dc = {"back": 1, "dark": 0, "hight": 0, "light": 0, "low": 0, "next": 0, "stop": 0}
        self.lst_of_commands = ["back", "dark", "hight", "light", "low", "next", "stop"]
        self.lst_path = ["C:/Python27/Neural/Networks/stop/numpy30/", "C:/Python27/Neural/Networks/back/numpy30/", "C:/Python27/Neural/Networks/dark/numpy30/", "C:/Python27/Neural/Networks/hight/numpy30/", "C:/Python27/Neural/Networks/light/numpy30/", "C:/Python27/Neural/Networks/low/numpy30/", "C:/Python27/Neural/Networks/next/numpy30/"]
        self.exam(dc)

if __name__ == '__main__':
    ff = ANNgenerator()
    ff.show_res()