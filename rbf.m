function [K,gamma] = rbf(X1, X2)
% X1 trainig
% X2 training or tuning or testing

% %% first 0/1 mean/var normalization
% [~,n1]=size(X1);
% [~,n2]=size(X2);
% avg=mean(X1,2);
% X1=X1-repmat(avg,1,n1);
% X2=X2-repmat(avg,1,n2);
% stdx=std(X1,[],2)+1e-8;
% X1=X1./repmat(stdx,1,n1);
% X2=X2./repmat(stdx,1,n2);

%% computing gaussian kernel
K=2*X1'*X2;
X1_norm=sum(X1.^2)';
X2_norm=sum(X2.^2);
K=bsxfun(@minus, K, X1_norm);
K=bsxfun(@minus, K, X2_norm);
gamma=-1/mean(mean(K));
% K=gamma*K;
% K=exp(K);
% K=(K+K')/2;