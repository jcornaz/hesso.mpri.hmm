function [MI,SIGMA,PCOMP]=initemis(N,K,c)
%
% Syntax: [MI,SIGMA]=initemis(O,N);
%
% initializes means and st. deviations of N state HMM. Pb. distributions
% are mono-gaussian with diag. cpvar matrix given by vectors of st. deviations
% O is matrix of observation vectors (P x T)
% N is nb. of states, including the first-entry and last-exit, which are
%   non-emitting.
% MI is matrix of means (each column=mean of one state), first and last columns
%   are dummy
% SIGMA is matrix of std (each column=std of one state), first and last columns
%   are dummy
 
nObs = length(c);
P=size(c{1},1);

MI=zeros(P,N,K);
SIGMA=zeros(P,N,K);
PCOMP = zeros(N,K);

for i=2:(N-1),
    O = [];
    for j=1:nObs
        [~,T]=size(c{j});
        X=round(linspace(2,N-1,T));
        O = [O c{j}(:,X==i)];
    end
    
    % call the K-means
    try
        idx = kmeans(O',K);
    catch
        disp('Problem with the K-Means');
    end
    
    % call the EM algorithm
    try
        obj = gmdistribution.fit(O',K,'Start',idx,'CovType','diagonal');    
    catch
        disp('Too few data:');
        disp('Add more data or reduce the number of states');
        break;
    end
     
    % create the emission matrix
    for k=1:K  
        MI(:,i,k) = obj.mu(k,:); 
        SIGMA(:,i,k) = sqrt(obj.Sigma(1,:,k));
        PCOMP(i,k) = obj.PComponents(k);
    end
end

end

