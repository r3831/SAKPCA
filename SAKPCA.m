
%%SAKPCA(dataname,data,options,numiter,eta) runs various stochastic approximation
%  kernel PCA algs
%  The inputs are as follows
%  dataname - could be 'orthogonal', 'xsmnist', 'swissroll' or 'dbmoon'
%
%   k       - a positive integer, denotes the desired RANK
%   numiter - number of iterations
%   options - consists of these fields:
%       eta     - step size multiplier. The sequence of step size used by
%                 various algorithms is eta/t, where t is the iteration index.
%       gamma   - bandwidth of the rbf kernel k(x,y)=exp(-gamma*norm(x-y)^2)
%       D       - dimension of the explicit feature mapping
%
%  The output containing the objective on the dev set,population objective,
%  and the singular value decomposition (U,S,V) is written to a
%  MATFILE in ../page/profile/PCA/METHOD/DATANAME, where METHOD is one of
%  'rfsgd', 'dfsgd' 'rfmsg', 'dfmsg' or 'batch'.
%
%%
function SAKPCA(dataname,data,options,numiter,eta,startiter)

%% Default is 1
if(nargin<6)
    startiter=1;
end

%% Default is 1
if(nargin<5)
    eta=1;
end

%% Default is just one run
if(nargin<4)
    numiter=1;
end

gamma=options.gamma;
D=options.D;
k=options.k;
m=options.m;

%% Simulation parameters
[d, ntr]=size(data.training);
ntun=size(data.tuning,2);
ntst=size(data.testing,2);

%% constructing Gaussian kernel
tcounter=tic;
[K,~]=rbf(data.training,data.training);
K=exp(gamma*K);
kernel_time=toc(tcounter);

[Ktun,~]=rbf(data.tuning,data.training);
Ktun=exp(gamma*Ktun);

[Ktun2,~]=rbf(data.tuning,data.tuning);
Ktun2=exp(gamma*Ktun2);

[Ktst,~]=rbf(data.testing,data.training);
Ktst=exp(gamma*Ktst);

[Ktruth,~]=rbf(data.testing,data.testing);
Ktruth=exp(gamma*Ktruth);


%% Check if all the runs are done
flag=1;
for method={'batch','rfsgd','rfftl','nystrom'}
    %% IF reading from or writing to a report file
    if(sum(strcmp(method,{'batch'})))
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,gamma=%f,numiter=%d].mat'],...
            dataname,method{1},k,gamma,numiter);
    elseif(sum(strcmp(method,{'nystrom'})))
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,gamma=%f,m=%d,numiter=%d].mat'],...
            dataname,method{1},k,gamma,m,numiter);
    elseif(sum(strcmp(method,{'rfftl'})))
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,gamma=%f,D=%d,numiter=%d].mat'],...
            dataname,method{1},k,gamma,D,numiter);
    elseif(sum(strcmp(method,{'rfsgd'})))
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,gamma=%f,D=%d,eta=%f,numiter=%d].mat'],...
            dataname,method{1},k,gamma,D,eta,numiter);
    end
    flag=flag && (exist(reportfile,'file'));
end


if(~flag)
    for ITER=startiter:numiter
        
        %% set the random seed
        rng(ITER);
        
        %% Shuffle data
        rp=randperm(ntr);
        Xtr=data.training(:,rp);
        
        %% Randomized Fourier Features
        tcounter=tic;
        W2=mvnrnd(zeros(d,1),2*gamma*eye(d),D);
        b=rand(D,1)*2*pi;
        Ztr=sqrt(2/D)*cos(W2*Xtr+repmat(b,[1,ntr]));
        rff_time=toc(tcounter);
        Ztun=sqrt(2/D)*cos(W2*data.tuning+repmat(b,[1,ntun]));
        Crftun=Ztun*Ztun'/ntun;
        [Utun,~]=svds(Crftun,k);
        Ztst=sqrt(2/D)*cos(W2*data.testing+repmat(b,[1,ntst]));
        Crftst=Ztst*Ztst'/ntst;
        [Utst,~]=svds(Crftst,k);
        
        %% Sequence close to a uniform grid on semilog axis
        [seq,L]=equilogseq(ntr,1);
        
        for method={'batch','nystrom','rfsgd','rfftl'}
            
            %% Display the run
            fprintf('Starting run: (%s,%s,%s,%d,%d)\n',...
                dataname,'SAKPCA',method{1},k,ITER);
            
            %% Set PAGE directories
            pagepath=sprintf('../page/profile/SAKPCA/%s/%s/',...
                method{1},dataname);
            
            if(sum(strcmp(method,{'batch'})))
                pageprefix=@()[pagepath,...
                    sprintf('%s[rank=%d,gamma=%f,iter=%d].mat',...
                    method{1},k,gamma,ITER)];
            elseif(sum(strcmp(method,{'nystrom'})))
                pageprefix=@()[pagepath,...
                    sprintf('%s[rank=%d,m=%d,gamma=%f,iter=%d].mat',...
                    method{1},k,m,gamma,ITER)];
            elseif(sum(strcmp(method,{'rfftl'})))
                pageprefix=@()[pagepath,...
                    sprintf('%s[rank=%d,D=%d,gamma=%f,iter=%d].mat',...
                    method{1},k,D,gamma,ITER)];
            elseif(sum(strcmp(method,{'rfsgd'})))
                pageprefix=@()[pagepath,...
                    sprintf('%s[rank=%d,D=%d,gamma=%f,eta=%f,iter=%d].mat',...
                    method{1},k,D,gamma,eta,ITER)];
            end
            % Check if the PAGE directory is structured properly
            if(~exist(pagepath,'dir'))
                % If not create the desired directory structure
                flag=createpath(pagepath);
                % If the directory structure could not be created
                if(~flag)
                    % Display error message and quit
                    error('Could not create path for result files');
                end
            end
            
            %% Output filename
            fname=pageprefix();
            if(~exist(fname,'file'))
                %% Initialize the basis for sgd and incremental PLS
                S=0;
                if(sum(strcmp(method{1},{'rfsgd'})))
                    U=orth(randn(D,k));
                elseif(sum(strcmp(method{1},{'rfftl'})))
                    U=zeros(D,k);
                    evals=zeros(1,k);
                end
                
                %% Initialize objective value
                objV=zeros(L(2),1);
                objVtune=zeros(L(2),1);
                runtime=zeros(L(2),1);
                
                %% Check if we can start from a previous run
                initsamp=1;
                
                %% Loop over data
                for iter=L(1)+1:L(2)
                    fprintf('Sequence number %d...\t',seq(iter));
                    switch(method{1})
                        case 'batch'
                            %% BATCH KPCA
                            isamp=seq(iter);
                            tcounter=tic;
                            Ktr = K(rp(1:isamp),rp(1:isamp));
                            [U, S] = svds(Ktr,k);
                            runtime(iter)=toc(tcounter)+(2*isamp*kernel_time)/(seq(end)*(seq(end)+1));
                            
                        case 'nystrom'
                            %% Nystrom + KPCA
                            isamp=seq(iter);
                            if isamp<5
                                continue;
                            end
                            tcounter=tic;
                            Ktr = K(rp(1:isamp),rp(1:isamp));
                            rpi=randi(isamp,[1,m]);
                            K_nm=Ktr(:,rpi);
                            K_mm=Ktr(rpi,rpi);
                            [U_mm,S_mm]=svd(K_mm);
                            S_mm=diag(S_mm);
                            idx=S_mm>1e-6;
                            S_mm=S_mm(idx); U_mm=U_mm(:,idx);
                            S_mm_inv_half=diag(S_mm.^(-.5));
                            G_sqrt=K_nm*(U_mm*S_mm_inv_half);
%                             K_mm_inv_half=(pinv(K_mm))^(.5);
%                             G_sqrt=K_nm*K_mm_inv_half;
%                             G=K_nm*pinv(K_mm)*K_nm';                            
                            [U, ~] = svds(G_sqrt,k);
                            runtime(iter)=toc(tcounter)+(2*isamp*kernel_time)/(seq(end)*(seq(end)+1));
                            S = diag(diag(U'*Ktr*U));
                            
                        case 'rfftl'
                            %% FTL + RFF
                            isamp=seq(iter);
                            if isamp<5
                                continue;
                            end
                            tcounter=tic;
                            [U, S] = svds(Ztr(:,1:isamp),k);
                            runtime(iter)=toc(tcounter)+rff_time/seq(end);

                        case 'rfsgd'
                            %% Oja + RFF
                            tcounter=tic;
                            for isamp=initsamp:seq(iter)
                                modisamp=1+mod(isamp-1,ntr);
                                etax=eta/(isamp);
                                z=sqrt(2/D)*cos(W2*Xtr(:,modisamp)+b);
                                U=sgd(U,z,etax);
                                if(~mod(modisamp,100))
                                    U=Gram_Schmidt(U);
                                end
                            end
                            runtime(iter)=toc(tcounter)+rff_time/seq(end);
                    end
                    
                    if(sum(strcmp(method{1},{'rfsgd'})))
                        UU=Gram_Schmidt(U);
                    elseif(sum(strcmp(method{1},{'rfftl'})))
                        UU=Gram_Schmidt(U);
                    end
                    initsamp=seq(iter)+1;
                    if(sum(strcmp(method{1},{'batch','nystrom'})))
                        objVtune(iter)=norm(Ktun(:,rp(1:isamp))*U*(S^(-.5)),'fro')^2/(ntun);
                        objV(iter)=norm(Ktst(:,rp(1:isamp))*U*(S^(-.5)),'fro')^2/(ntst);
                    elseif(sum(strcmp(method{1},{'rfsgd','rfftl'})))
                        Ur=UU*UU'*Utun;
                        S=diag(diag(Ur'*Crftun*Ur));
                        UUU=Ztun'*Ur*S^(-.5)/sqrt(ntun);
                        objVtune(iter)=trace(UUU'*Ktun2*UUU/ntun);
                        
                        Ur=UU*UU'*Utst;
                        S=diag(diag(Ur'*Crftst*Ur));
                        UUU=Ztst'*Ur*S^(-.5)/sqrt(ntst);
                        objV(iter)=trace(UUU'*Ktruth*UUU/ntst);
                    end
                    fprintf('\t%d\t %g\t %g\n',k,objVtune(iter),objV(iter));
                end
                save(fname,'runtime','U','S','objV','objVtune','seq');
            end
        end
        
    end
end

%% Ground truth

%% Set PAGE directories
method={'truth'};
pagepath=sprintf('../page/profile/SAKPCA/%s/%s/',method{1},dataname);
% Check if the PAGE directory is structured properly
if(~exist(pagepath,'dir'))
    % If not create the desired directory structure
    flag=createpath(pagepath);
    % If the directory structure could not be created
    if(~flag)
        % Display error message and quit
        error('Could not create path for result files');
    end
end
fname=[pagepath,sprintf('truth[rank=%d,gamma=%f].mat',k,gamma)];
if(~exist(fname,'file'))
    Lambda = svds(Ktruth,k)/ntst;
    objV=sum(Lambda(1:k)); %#ok
    save(fname,'objV');
end
end
