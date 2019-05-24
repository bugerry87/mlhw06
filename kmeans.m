function [M, Y] = kmeans(X, M, yield)
%KMEANS Summary of this function goes here
%   Detailed explanation goes here

N = size(X,1);
K = size(M,1);
D = zeros(N,K);
Y = zeros(N,1);

while 1
    for k=1:K
        m = M(k,:);
        D(:,k) = vecnorm(X - m, 2, 2);
    end

    [~, newY] = min(D, [], 2);
    if any(newY ~= Y, 'all')
        Y = newY;
        yield(X, Y, M);
        for k=1:K
            M(k,:) = mean(X(Y==k,:));
            yield(X, Y, M);
        end
    else
        break;
    end
end

end

