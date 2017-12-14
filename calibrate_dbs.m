% Urban informatics course
% Eelis Peltola, id:240286
% 12.12.2017
% This function calculates the decibel magnitude of calibration audio, and
% the difference to dB values fromm calibration device.

% Calibration test audio, in small audio room with 1 kHz sine wave sound
filenames = ['dba_test1_sound1.wav'; 'dba_test1_sound2.wav'; ...
    'dba_test1_sound3.wav'; 'dba_test1_sound4.wav'; ...
    'dba_test1_sound5.wav'; 'dba_test1_sound6.wav'];
% Calibration device dB values (A-weighted)
calib_device_db = [37, 38.2, 55, 67.1, 78, 84.4];
% Init arrays
averages_calib = [];
diffs_db = [];


for k = 1:size(filenames, 1)
    fn = filenames(k,:);
    [audio, fs] = audioread(fn, 'native');
    sz = size(audio);
    if sz(2) == 2
        audio = (audio(:,1)+audio(:,2))/2;
    end

    % A-weighting with filterA.m script from M.Sc. Eng. Hristo Zhivomirov
    audioA = filterA(audio, fs);
    % Convert to dB magnitude values
    dbA = 20*log10(abs(audioA));
    % Average dB
    avg_dbA = mean(dbA);
    averages_calib = [averages_calib, avg_dbA];
    % Difference of audio dB and calibration device dB
    diff_db = calib_device_db(k) - avg_dbA;
    diffs_db = [diffs_db, diff_db];
end

% Print output to command window
diffs_db
averages_calib
% Final average difference used as correction value for recordings
avg_diffs = mean(diffs_db)