import genann
import extractor
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
commands = [u"back", u"dark", u"hight", u"light", u"low", u"next", u"stop"]
path = [u"C:/Python27/Neural/tt2/", u"C:/Python27/Neural/Networks/stop/numpy30/",
            u"C:/Python27/Neural/Networks/back/numpy30/", u"C:/Python27/Neural/Networks/dark/numpy30/",
            u"C:/Python27/Neural/Networks/hight/numpy30/", u"C:/Python27/Neural/Networks/light/numpy30/",
            u"C:/Python27/Neural/Networks/low/numpy30/", u"C:/Python27/Neural/Networks/next/numpy30/"]
Mel = extractor.MelExtractor(glob_path=u"C:/Python27/Neural/Networks/back/speech/", glob_path_out=u"C:/Python27/Neural/Networks/next/numpy16/",
                             dir_list=True, logger=log, show_sbs=False)
Mel.viewer()
ff = genann.AnnGenerator(lst_of_commands=commands, logger=log)
ff.train_res(path)
ff.test_res(path_for_testing=u"C:/Python27/Neural/tt/")
