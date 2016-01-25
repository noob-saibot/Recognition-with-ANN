#Recognition-with-ANN
Two modules what provide the opportunity to recognize izolated words.

####Firts module is "extractor.py".
1. This module extract mfcc from WAV files.
2. You can return array of mfcc for each frames.
3. You can save numpy arrays for each file.

####Second module is "genann.py".
1. This module generate Artificial Neural Network for each command.
2. Training each ANN.
3. Testing result of training.

####Example:
```
import genann
import extractor
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# Input commands for recognition.
commands = ["List of commands"]
# Input list of directory to numpy arrays.
path = ["List of paths"]
# Here you're extracting mfcc from files.
mel = extractor.MelExtractor(glob_path="YOURPATH", glob_path_out="YOURPATH",
                             dir_list=True, logger=log)
mel.viewer()

# Create ANN.
ann_example = genann.AnnGenerator(lst_of_commands=commands, logger=log)
# Train them.
ann_example.train_res(path)
# Test them.
ann_example.test_res(path_for_testing="YOURPATH")
```

