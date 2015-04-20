function [Pvit, X, PThr] = viterbi_log(O, A, MIMIX, SIGMAMIX,Pcomp)

% Viterbi decoding

[P,T]=size(O);
N = size(A,1);
K = size(Pcomp,2);
min_val = -1e8;
FI = zeros(N,T);
XX = zeros(N,T); 
X = zeros(1,N);
    
% init 
FI(1,1) = 1;
for j=2:(N-1)
    if (K > 2)
        emis = add_lns(logexp(O,MIMIX,SIGMAMIX,Pcomp,1:P,j,1,1),logexp(O,MIMIX,SIGMAMIX,Pcomp,1:P,j,2,1));
        for k=3:K
            emis = add_lns(emis,logexp(O,MIMIX,SIGMAMIX,Pcomp,1:P,j,k,1));
        end 
    elseif (K == 2)
        emis = add_lns(logexp(O,MIMIX,SIGMAMIX,Pcomp,1:P,j,1,1),logexp(O,MIMIX,SIGMAMIX,Pcomp,1:P,j,2,1));
    else % (K == 1) 
        emis = logexp(O,MIMIX,SIGMAMIX,Pcomp,1:P,j,1,1);
    end 
    if (A(1,j) == 0)
        FI(j,1) = min_val + emis;
    else
        FI(j,1) = log(A(1,j)) + emis; 
    end
end
    
% cycle
for t=2:T, 
    for j=2:(N-1),
        if (A(2:(N-1),j) == 0)
            [mm,ii] =  max ( FI(2:(N-1),t-1)+ min_val );
        else
            [mm,ii] =  max ( FI(2:(N-1),t-1)+log(A(2:(N-1),j)) );
        end
        ii=ii+1;
        XX(j,t)=ii;   
        if (K>2)
            emis = add_lns(logexp(O,MIMIX,SIGMAMIX,Pcomp,1:P,j,1,t),logexp(O,MIMIX,SIGMAMIX,Pcomp,1:P,j,2,t));
            for k=3:K
                emis = add_lns(emis,logexp(O,MIMIX,SIGMAMIX,Pcomp,1:P,j,k,t));
            end  
        elseif (K == 2)
            emis = add_lns(logexp(O,MIMIX,SIGMAMIX,Pcomp,1:P,j,1,t),logexp(O,MIMIX,SIGMAMIX,Pcomp,1:P,j,2,t));
        else % (K == 1)
            emis = logexp(O,MIMIX,SIGMAMIX,Pcomp,1:P,j,1,t);
        end 
        FI(j,t) = mm + emis;
    end
end

% final 
if (A(2:(N-1),N) == 0)
    [mm,ii] =  max ( FI(2:(N-1),T) + emis) ;
else
    [mm,ii] =  max ( FI(2:(N-1),T) + log(A(2:(N-1),N))) ;
end

ii=ii+1;
XX(N,T) = ii;
Pvit = mm; 

%%% backtrace %%%
X(T) = XX(N,T);     

% last value for threshold
if (A(X(T),N) == 0)
    PThr = min_val;
else
    PThr = log(A(X(T),N));
end

for t=T-1:-1:1,
    X(t) = XX(X(t+1), t+1);
    %%% backtrace threshold %%%
    if (A(X(t),X(t+1)) == 0)
        PThr = PThr + min_val;
    else
        PThr = PThr + log(A(X(t),X(t+1)));
    end
end 

end    

function [res]  = logexp(O,MIMIX,SIGMAMIX,Pcomp,p,j,k,t)
    res = log(Pcomp(j,k))-log(sqrt((2*pi).^length(p)))-log(prod(SIGMAMIX(p,j,k))) + ...
          sum(-((O(p,t) - MIMIX(p,j,k)).^2./(2*SIGMAMIX(p,j,k).^2)));
    % verified with: res = sum(log(Pcomp(j,k))+log(normpdf(O(p,t),MIMIX(p,j,k),(SIGMAMIX(p,j,k)))));
end    