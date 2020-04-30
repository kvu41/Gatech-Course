function [N] = noiselet(n)
% Author : Kamlesh Pawar
% Input  : 
%      n : size of the desired noiselet matrix (should be of form 2^k)
% Output : 
%      N : nxn noiselet matrix
if (n<2 || (log2(n)-floor(log2(n)))~=0)
    error('The input argument should be of form 2^k');
end
N = 0.5*[1-i 1+i; 
         1+i 1-i];
for indx = 2:log2(n)
    N1 = N;
    N = zeros(2^indx,2^indx);
    for k = 1:2:2^indx-1
        N(k,:) = 0.5*kron([1-i 1+i],N1((k+1)/2,:));
    end
    
    for k = 2:2:2^indx
        N(k,:) = 0.5*kron([1+i 1-i],N1(k/2,:));
    end
end
