function [Pvit, X] = viterbi_nolog (O, A, MI, SIGMA);
%
%Syntax: [Pvit, X] = viterbi_nolog (O, A, MI, SIGMA);
%
% Viterbi decoding

[P,T]=size(O);
N=size(A,1);

FI=zeros(N,T);
XX=zeros(N,T);
X=zeros(1,N);

% init 
FI(1,1) = 1;
for j=2:(N-1),
  FI(j,1) = A (1,j) * normal (O(:,1), MI(:,j), SIGMA(:,j));
end

% cycle
for t=2:T,
  for j=2:(N-1),
    [mm,ii] =  max ( FI(2:(N-1),t-1).*A(2:(N-1),j) ) ;
    ii=ii+1;
    XX(j,t)=ii;
    emis =  normal (O(:,t), MI(:,j), SIGMA(:,j));
    FI(j,t) = mm * emis;
  end
end

% final
[mm,ii] =  max ( FI(2:(N-1),T).*A(2:(N-1),N) ) ;
ii=ii+1;
XX(N,T) = ii;
Pvit=mm;
 
%%% backtrace %%%
X(T) = XX(N,T);
for t=T-1:-1:1,
  X(t) = XX(X(t+1), t+1);
end 
