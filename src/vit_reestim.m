function [NEWA,NEWMI,NEWSIGMA,Ptot] = vit_reestim (O1,O2,O3, A, MI, SIGMA);
%
%Syntax: [NEWA,NEWMI,NEWSIGMA,Ptot] = vit_reestim (O1,O2,O3, A, MI, SIGMA);
%
% Reestimation of HMM using Viterbi criterion
%[P,T]=size(O);
N=size(A,1);

% if T<N,
%  error ('Not enough obs. vectors to reestim all states');
% end

[Pvit1,ALIGN1] = viterbi_log(O1,A,MI,SIGMA);
[Pvit2,ALIGN2] = viterbi_log(O2,A,MI,SIGMA);
[Pvit3,ALIGN3] = viterbi_log(O3,A,MI,SIGMA);

Ptot = (Pvit1 + Pvit2 + Pvit3)/3;

NEWA=zeros(size(A));
NEWMI=zeros(size(MI));
NEWSIGMA=zeros(size(SIGMA));

% compute new emission probabilities
for i=2:(N-1),
  ggg1=find(ALIGN1==i);
  ggg2=find(ALIGN2==i);
  ggg3=find(ALIGN3==i);
  O = [O1(:,ggg1)';O2(:,ggg2)';O3(:,ggg3)'];
  NEWMI(:,i) = mean (O)';
  NEWSIGMA(:,i) = std (O)';
end

% compute new transition probabilities
NEWA(1,2) = 1.0;
for i=2:(N-1),
  [notUsed,l1] = size(find(ALIGN1==i));
  [notUsed,l2] = size(find(ALIGN2==i));
  [notUsed,l3] = size(find(ALIGN3==i));
  numberOfVectorInState = l1 + l2 + l3;
  NEWA(i,i)   = (numberOfVectorInState-3) / numberOfVectorInState;
  NEWA(i,i+1) = 1 - NEWA(i,i);
end

