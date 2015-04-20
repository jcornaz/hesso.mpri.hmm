function [NEWA,NEWMI,NEWSIGMA,NEWPCOMP,Pvit_tot] = vit_reestim (A, MI, SIGMA, PCOMP, c)
%
%Syntax: [NEWA,NEWMI,NEWSIGMA,Ptot] = vit_reestim (A, MI, SIGMA, O1, O2, ....);
%
% Reestimation of HMM using Viterbi criterion
%[P,T]=size(O);
N=size(A,1);
nObs = length(c);
K = size(PCOMP,2);
% if T<N,
%  error ('Not enough obs. vectors to reestim all states');
% end
TransMatrix = zeros(N);
Pvit_tot = 0;
ALIGN_tot = {};
for j=1:nObs
    [Pvit,ALIGN] = viterbi_log(c{j},A,MI,SIGMA,PCOMP);
    ALIGN_tot{j} = ALIGN;
    % cumulative Matrix of transition, for computing the transition Matrix
    TransMatrix(1,ALIGN(1)) = TransMatrix(1, ALIGN(1)) + 1;
    for jj=2:length(ALIGN) 
        TransMatrix(ALIGN(jj-1),ALIGN(jj)) = TransMatrix(ALIGN(jj-1),ALIGN(jj))+1;
    end 
    TransMatrix(ALIGN(jj),N) = TransMatrix(ALIGN(jj),N) +1; 
    Pvit_tot = Pvit_tot + Pvit;
end
Pvit_tot = Pvit_tot/nObs;

NEWA=zeros(size(A));
NEWMI=zeros(size(MI));
NEWSIGMA=zeros(size(SIGMA));
NEWPCOMP = zeros(N,K);
% compute new emission probabilities
for i=2:(N-1),
    O = [];
    for j=1:nObs
        ind_align = ALIGN_tot{j}==i;
        O = [O; c{j}(:,ind_align)'];
    end
    try
        obj = gmdistribution.fit(O,K,'CovType','diagonal');   
    catch
        disp('Too few data:');
        disp('Add more data or reduce the number of states');
        break;
    end
    % create the emission matrix
    for k=1:K 
        NEWMI(:,i,k) = obj.mu(k,:); 
        NEWSIGMA(:,i,k) = sqrt(obj.Sigma(1,:,k));
        NEWPCOMP(i,k) = obj.PComponents(k);
    end
end

SumTM = sum(TransMatrix,2);
% cut the zero for avoid NaN in A
SumTM((SumTM(1:(N-1)) == 0)) = 1;
NEWA(1:(N-1),:) = bsxfun(@rdivide,TransMatrix(1:(N-1),:),SumTM(1:(N-1)));

end

