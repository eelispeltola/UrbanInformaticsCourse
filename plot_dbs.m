% Urban informatics course
% Eelis Peltola, id:240286
% 12.12.2017
% For the recordings of 17.09.2017 0109: Stockmann

% Recordings of cars passing
carfiles = ['02-1.wav'; '03-1.wav'; '04-1.wav'; '05-1.wav'; ...
    '06-1.wav'; '11-1.wav'; '12-1.wav'; '13-1.wav'];

% Recordings of speech
speechfiles = ['01-2-1-0-1.wav'; '07-2-0-0-1.wav'; '08-2-1-0.wav  '; ...
    '09-2-1-0.wav  '; '10-2-1-0-1.wav'];

% Init arrays
car_avgs = [];
car_dbs = [];
speech_avgs = [];
speech_dbs = [];
t_car_tot = [];
t_speech_tot = [];

% Run dbA_avg() for all car files, push all values into arrays
for k = 1:size(carfiles, 1)
    fn = carfiles(k,:);
    [car_db, avg_db, t_car] = dbA_avg(fn);
    car_avgs = [car_avgs; avg_db];
    car_dbs = [car_dbs; car_db];
    t_car_tot = [t_car_tot, t_car];
end

% Run dbA_avg() for all speech files, push all values into arrays
for k = 1:size(speechfiles, 1)
    fn = speechfiles(k,:);
    [speech_db, avg_db, t_speech] = dbA_avg(fn);
    speech_avgs = [speech_avgs; avg_db];
    speech_dbs = [speech_dbs; speech_db];
    t_speech_tot = [t_speech_tot, t_speech];
end


% Plotting values into two subplots

figure
subplot(2,1,1)
plot(t_car_tot, car_dbs)
xlabel('Time (seconds)')
ylabel('Amplitude (dB)')
title('All car sounds')
car_avg_all = mean(car_avgs) % Output average of all dB values from cars

subplot(2,1,2)
plot(t_speech_tot, speech_dbs)
xlabel('Time (seconds)')
ylabel('Amplitude (dB)')
title('All speech')
speech_avg_all = mean(speech_avgs) % Output average of all dB values
                                   % from speech