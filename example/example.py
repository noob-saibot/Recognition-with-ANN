import genann
import extractor
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
commands = ["List of commands"]
path = ["List of paths"]
Mel = extractor.MelExtractor(glob_path="YOURPATH", glob_path_out="YOURPATH",
                             dir_list=True, logger=log)
Mel.viewer()
ff = genann.AnnGenerator(lst_of_commands=commands, logger=log)
ff.train_res(path)
ff.test_res(path_for_testing="YOURPATH")
