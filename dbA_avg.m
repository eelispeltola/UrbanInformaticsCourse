% Urban informatics course
% Eelis Peltola, id:240286
% 12.12.2017
% This function calculates the calibrated decibel magnitude, its
% average and the length in seconds of a wav audio recording

function [dbA_audio, avg_dbA, t] = dbA_avg(fn)

% Read wav files in native format
[audio, fs] = audioread(fn, 'native');
% If stereo audio, average channels to mono
sz = size(audio);
if sz(2) == 2
    audio = (audio(:,1)+audio(:,2))/2;
end

% Uncomment for frequencies
% freq=[0:fs/length(audio):fs/2];

% A-weighting with filterA.m script from M.Sc. Eng. Hristo Zhivomirov
audioA = filterA(audio, fs);
% Convert to dB magnitude values
dbA_aud = 20*log10(abs(audioA));
% Add calibration value from calibration test
dbA_aud = dbA_aud+15.55;
% Calculate time in seconds for plotting
T = 1/fs;
t_end = (length(audio)*T)-T;
t = 0:T:t_end;

% Uncomment for individual plots
% figure
% plot(t,audio);
% plot(t, dbA_aud)
% xlabel('Time (seconds)')
% ylabel('Amplitude')

% Average A-weighted dB
avg_dbA = mean(dbA_aud);
% For output
dbA_audio = dbA_aud;

end
