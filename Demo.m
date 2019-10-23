clear all;
addpath(genpath('../../sakpca')) % enter your root here

%% load the dataset
load('mnist.mat'); dataname='mnist'; gamma=.01; eta=40; X=X(:,1:20000);
[d,n]=size(X);

%% initialize parameters
numiter=2;
D=1000; % RFF parameter
m=100; % Nystrom parameter
k=10;
options.gamma=gamma; options.D=D; options.k=k; options.m=m;

%% make training, tuning and testing sets
ntr=ceil(n*3/8); ntun=ceil(n/8);
data.training=X(:,1:ntr);
data.tuning=X(:,ntr+1:ntr+ntun);
data.testing=X(:,ntr+ntun+1:end);

%% run different algorithms for kernel pca
SAKPCA(dataname,data,options,numiter,eta);
plotobjV(dataname,options,ntr,numiter,eta);
