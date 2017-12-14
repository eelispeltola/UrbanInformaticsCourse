For Sustainable Design Studio / Urban Informatics course Fall 2017


Building a CNN to classify the UrbanSounds8K dataset:
* us8k_extract.py: log-scaled Mel-spectrogram feature extractor modified from Aqib Saeed's excellent tutorial http://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/
* us8k_salamon.py: builds, trains and evaluates CNN introduced by Salamon and Bello in https://arxiv.org/pdf/1608.04363.pdf
* salamon-cnn.h5: pre-built and saved model
* confusion_matrix.png: confusion matrix of results of the pre-built model
* Download the UrbanSounds8K dataset from https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html

Calculating dBA weighted SPL for a calibrated microphone:
* filterA.m: Matlab A-weighting filter by M.Sc. Eng. Hristo Zhivomirov
* license.txt: License for filterA.m
* dbA_avg.m: calculates dBA SPL of a .wav file
* calibrate_dbs.m: calculates calibration value for accurate SPL measurements based on recordings where a calibration device was used to read wanted SPL was present next to the microphone
* plot_dbs.m: plots dBA SPL values of Hämeenkatu recording excerpts
* 0109one_hkatu_recording: manually annotated data from Hämeenkatu on one Saturday morning, next to Stockmann. Move contents of this folder to same folder as plot_dbs.m to use
* dbatest1_sound1.wav...dbatest2_sound6.wav: calibration recordings for calibrate_dbs.m

Gender recognition from a csv file with voice frequency-space data:
* gender_recognition.py: builds, trains and evaluates model from voice.csv
* voice.csv: feature and label data from multiple voice datasets, by Kory Becker http://www.primaryobjects.com/2016/06/22/identifying-the-gender-of-a-voice-using-machine-learning/
* voicemodel.h5: pre-built and saved model

Manual annotation of Hämeenkatu data:
* Hämeenkatu manually classified data.txt: manually classified categories of all Hämeenkatu data