% HMM exercise  
% -------------
clear;

% add your data
addpath('../data')

disp ('-------- reading signal and computing cepstra ----------');

%-----------reading in the training data----------------------------------
[training_data1_1,Fs1_1]=audioread('1_1.wav');
[training_data1_2,Fs1_2]=audioread('1_2.wav');
[training_data1_3,Fs1_3]=audioread('1_3.wav');
[training_data2_1,Fs2_1]=audioread('2_1.wav');
[training_data2_2,Fs2_2]=audioread('2_2.wav');
[training_data2_3,Fs2_3]=audioread('2_3.wav');
[training_data3_1,Fs3_1]=audioread('3_1.wav');
[training_data3_2,Fs3_2]=audioread('3_2.wav');
[training_data3_3,Fs3_3]=audioread('3_3.wav');
[training_data4_1,Fs4_1]=audioread('4_1.wav');
[training_data4_2,Fs4_2]=audioread('4_2.wav');
[training_data4_3,Fs4_3]=audioread('4_3.wav');
[training_data5_1,Fs5_1]=audioread('5_1.wav');
[training_data5_2,Fs5_2]=audioread('5_2.wav');
[training_data5_3,Fs5_3]=audioread('5_3.wav');

[testing_data1,Fs1t]=audioread('1t.wav');
[testing_data2,Fs2t]=audioread('2t.wav');
[testing_data3,Fs3t]=audioread('3t.wav');
[testing_data4,Fs4t]=audioread('4t.wav');
[testing_data5,Fs5t]=audioread('5t.wav');

[testing_datap,Fspt]=audioread('Peut.wav');

%-------------feature extraction, 12 coeff. ------------------------------------------
c1_1=melcepst(training_data1_1,Fs1_1)';
c1_2=melcepst(training_data1_2,Fs1_2)';
c1_3=melcepst(training_data1_3,Fs1_3)';

c2_1=melcepst(training_data2_1,Fs2_1)';
c2_2=melcepst(training_data2_2,Fs2_2)';
c2_3=melcepst(training_data2_3,Fs2_3)';

c3_1=melcepst(training_data3_1,Fs3_1)';
c3_2=melcepst(training_data3_2,Fs3_2)';
c3_3=melcepst(training_data3_3,Fs3_3)';

c4_1=melcepst(training_data4_1,Fs4_1)';
c4_2=melcepst(training_data4_2,Fs4_2)';
c4_3=melcepst(training_data4_3,Fs4_3)';

c5_1=melcepst(training_data5_1,Fs5_1)';
c5_2=melcepst(training_data5_2,Fs5_2)';
c5_3=melcepst(training_data5_3,Fs5_3)';

c1t=melcepst(testing_data1,Fs1t)';
c2t=melcepst(testing_data2,Fs2t)';
c3t=melcepst(testing_data3,Fs3t)';
c4t=melcepst(testing_data4,Fs4t)';
c5t=melcepst(testing_data5,Fs5t)';

cp=melcepst(testing_datap,Fspt)';

disp ('-------- data adaptation: array of cells ----------');
c1 = {c1_1,c1_2,c1_3};
c2 = {c2_1,c2_2,c2_3};
c3 = {c3_1,c3_2,c3_3};
c4 = {c4_1,c4_2,c4_3};
c5 = {c5_1,c5_2,c5_3};

disp ('-------- training ----------');
% HMM parameters
n_iter = 5;
K = [1 1 1 1 1];
N = [3 5 6 6 5];

disp ('-------- training model for 1 ----------');
A=inittran(N(1)); 
[MI,SIGMA,PCOMP] = initemis(N(1),K(1),c1); 
[NEWA, NEWMI, NEWSIGMA, NEWPCOMP,Ptot] = vit_reestim (A, MI, SIGMA, PCOMP, c1);
disp(Ptot);
for iter=1:n_iter
   [NEWA,NEWMI,NEWSIGMA,NEWPCOMP,Ptot] = vit_reestim (NEWA, NEWMI, NEWSIGMA,NEWPCOMP, c1);  
   disp(Ptot);
end
A1=NEWA; MI1=NEWMI; SIGMA1=NEWSIGMA; PCOMP1=NEWPCOMP;

disp ('-------- training model for 2 ----------');
A=inittran(N(2)); 
[MI,SIGMA,PCOMP]=initemis(N(2),K(2),c2); 
[NEWA, NEWMI, NEWSIGMA, NEWPCOMP,Ptot] = vit_reestim (A, MI, SIGMA, PCOMP, c2);
disp(Ptot);
for iter=1:n_iter 
   [NEWA,NEWMI,NEWSIGMA,NEWPCOMP,Ptot] = vit_reestim (NEWA, NEWMI, NEWSIGMA,NEWPCOMP,c2);
   disp(Ptot);
end
A2=NEWA; MI2=NEWMI; SIGMA2=NEWSIGMA; PCOMP2=NEWPCOMP;

disp ('-------- training model for 3 ----------');
A=inittran(N(3)); 
[MI,SIGMA,PCOMP]=initemis(N(3),K(3),c3); 
[NEWA, NEWMI, NEWSIGMA, NEWPCOMP,Ptot] = vit_reestim (A, MI, SIGMA,PCOMP,c3);
disp(Ptot);
for iter=1:n_iter  
   [NEWA,NEWMI,NEWSIGMA,NEWPCOMP,Ptot] = vit_reestim (NEWA, NEWMI, NEWSIGMA,NEWPCOMP,c3);  
   disp(Ptot);
end
A3=NEWA; MI3=NEWMI; SIGMA3=NEWSIGMA; PCOMP3=NEWPCOMP;

disp ('-------- training model for 4 ----------');
A=inittran(N(4)); 
[MI,SIGMA,PCOMP]=initemis(N(4),K(4),c4); 
[NEWA, NEWMI, NEWSIGMA,NEWPCOMP, Ptot] = vit_reestim (A, MI, SIGMA,PCOMP,c4);
disp(Ptot);
for iter=1:n_iter
   [NEWA,NEWMI,NEWSIGMA,NEWPCOMP,Ptot] = vit_reestim (NEWA, NEWMI, NEWSIGMA,NEWPCOMP,c4);  
   disp(Ptot);
end
A4=NEWA; MI4=NEWMI; SIGMA4=NEWSIGMA; PCOMP4=NEWPCOMP;

disp ('-------- training model for 5 ----------');
A=inittran(N(5)); 
[MI,SIGMA,PCOMP]=initemis(N(5),K(5),c5); 
[NEWA, NEWMI, NEWSIGMA, NEWPCOMP,Ptot] = vit_reestim (A, MI, SIGMA,PCOMP,c5);
disp(Ptot);
for iter=1:n_iter
   [NEWA,NEWMI,NEWSIGMA,NEWPCOMP,Ptot] = vit_reestim (NEWA, NEWMI, NEWSIGMA,NEWPCOMP,c5);  
   disp(Ptot);
end
A5=NEWA; MI5=NEWMI; SIGMA5=NEWSIGMA; PCOMP5=NEWPCOMP;


%disp ('====== now recognizing  =======') 
%format short e % this is to see correctly all elements of a vector

Pvit11 = viterbi_log (c1t, A1, MI1, SIGMA1,PCOMP1);
Pvit12 = viterbi_log (c1t, A2, MI2, SIGMA2,PCOMP2);
Pvit13 = viterbi_log (c1t, A3, MI3, SIGMA3,PCOMP3);
Pvit14 = viterbi_log (c1t, A4, MI4, SIGMA4,PCOMP4);
Pvit15 = viterbi_log (c1t, A5, MI5, SIGMA5,PCOMP5);
h = [Pvit11 Pvit12 Pvit13 Pvit14 Pvit15];
[~,ii] = max(h); 
disp(['testing for 1t, the best model is ' num2str(ii) ]);

Pvit11 = viterbi_log (c2t, A1, MI1, SIGMA1,PCOMP1);
Pvit12 = viterbi_log (c2t, A2, MI2, SIGMA2,PCOMP2);
Pvit13 = viterbi_log (c2t, A3, MI3, SIGMA3,PCOMP3);
Pvit14 = viterbi_log (c2t, A4, MI4, SIGMA4,PCOMP4);
Pvit15 = viterbi_log (c2t, A5, MI5, SIGMA5,PCOMP5);
h = [Pvit11 Pvit12 Pvit13 Pvit14 Pvit15];
[~,ii] = max(h); 
disp(['testing for 2t, the best model is ' num2str(ii) ]);

Pvit11 = viterbi_log (c3t, A1, MI1, SIGMA1,PCOMP1);
Pvit12 = viterbi_log (c3t, A2, MI2, SIGMA2,PCOMP2);
Pvit13 = viterbi_log (c3t, A3, MI3, SIGMA3,PCOMP3);
Pvit14 = viterbi_log (c3t, A4, MI4, SIGMA4,PCOMP4);
Pvit15 = viterbi_log (c3t, A5, MI5, SIGMA5,PCOMP5);
h = [Pvit11 Pvit12 Pvit13 Pvit14 Pvit15];
[~,ii] = max(h); 
disp(['testing for 3t, the best model is ' num2str(ii) ]);

Pvit11 = viterbi_log (c4t, A1, MI1, SIGMA1,PCOMP1);
Pvit12 = viterbi_log (c4t, A2, MI2, SIGMA2,PCOMP2);
Pvit13 = viterbi_log (c4t, A3, MI3, SIGMA3,PCOMP3);
Pvit14 = viterbi_log (c4t, A4, MI4, SIGMA4,PCOMP4);
Pvit15 = viterbi_log (c4t, A5, MI5, SIGMA5,PCOMP5);
h = [Pvit11 Pvit12 Pvit13 Pvit14 Pvit15];
[~,ii] = max(h); 
disp(['testing for 4t, the best model is ' num2str(ii) ]);

Pvit11 = viterbi_log (c5t, A1, MI1, SIGMA1,PCOMP1);
Pvit12 = viterbi_log (c5t, A2, MI2, SIGMA2,PCOMP2);
Pvit13 = viterbi_log (c5t, A3, MI3, SIGMA3,PCOMP3);
Pvit14 = viterbi_log (c5t, A4, MI4, SIGMA4,PCOMP4);
Pvit15 = viterbi_log (c5t, A5, MI5, SIGMA5,PCOMP5);
h = [Pvit11 Pvit12 Pvit13 Pvit14 Pvit15];
[nic,ii] = max(h); 
disp(['testing for 5t, the best model is ' num2str(ii) ]);