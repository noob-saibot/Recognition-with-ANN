#Recognition-with-ANN
Two modules what provide the opportunity to recognize izolated words.

####Firts module is "extractor.py".
1. This module extract mfcc from WAV files.
2. You can return array of mfcc for each frames.
3. You can save numpy arrays for each file.

####Second module is "genann.py".
1. This module generate Artifical Neural Network for each command.
2. Training each ANN.
3. Testing result of training.

####Example:
```
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
```
