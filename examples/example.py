from ANNrecon import *
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# Input commands for recognition.
commands = [u"List of commands"]
# Input list of directory to numpy arrays.
path = [u"List of paths"]
# Here you're extracting mfcc from files.
Mel = extractor.MelExtractor(glob_path="YOURPATH", glob_path_out="YOURPATH",
                             dir_list=True, logger=log, show_sbs=False)
Mel.viewer()
# Create ANN.
ann_example = genann.AnnGenerator(lst_of_commands=commands, logger=log)
# Train them.
ann_example.train_res(path)
# Test them.
ann_example.test_res(path_for_testing=u"ann_example")
