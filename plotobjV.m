
%% PLOTOBJV(dataname,k,n,numiter,eta) plots
% PROGRESS-PER-ITERATION (empirical & population) and PROGRESS-PER-SECOND.
%
%  The inputs are as follows:
%
%  dataname is one of these strings: 'VidTIMIT', 'VidTIMIT2', 'SIM'
%  method is a cell of strings: e.g. {'sgd', 'brand', 'batch', 'truth'}
%
%  k a positive integer - denotes the desired RANK
%
%  numiter number of iterations to be averaged over
%
%  The output containing the report is written to ../REPORT as a MAT file
%  in the following format: reportPCA[method=sgd,rank=4,numiter=10].mat
%
%  If PLOTREPORT is set to 1, three plots are generated and written to
%  ../PLOTS as pdf files with names in the following formats:
%
%  iteration[dataname=VidTIMIT,rank=1,numiter=1,...
%  methods=sgd,brand]
%  convergence[dataname=VidTIMIT,rank=1,numiter=1,...
%  methods=sgd,brand].pdf
%
%%

function plotobjV(dataname,options,n,numiter,eta)

gamma=options.gamma;
D=options.D;
k=options.k;
m=options.m;


% Check if the PLOTS directory is structured properly
if(~exist('../plots','dir'))
    % If not create the desired directory structure
    flag=createpath('../plots');
    % If the directory structure could not be created
    if(~flag)
        % Display error message and quit
        error('Could not create path for the plots');
    end
end



%% PCA flag and method TAGS
% methods={'batch','rfsgd','dfsgd','rfmsg','dfmsg'};
methods={'batch','nystrom','rfsgd','rfftl'};

%% Setup Figures
col={[0 240 0],[240 0 0],[0 0 240],[0 240 240],...
    [0 0 0],[240 240 240],[160 80 80],[200 200 0]};
marker={'k^','ks','kd','ko','kv','kp','kp'};
fig11=figure(11); clf; set(fig11,'Position',[2 2 1200 800]);
fig22=figure(22); clf; set(fig22,'Position',[2 2 1200 800]);

%% File names of figures to be plotted
fnames=cell(2,1);
fnames{1}=sprintf(['../plots/iteration[data=%s,rank=%d,D=%d,gamma=%f,',...
    'numiter=%d,eta=%f].pdf'],dataname,k,D,gamma,numiter,eta);
fnames{2}=sprintf(['../plots/elapsedtime[data=%s,rank=%d,D=%d,gamma=%f,',...
    'numiter=%d,eta=%f].pdf'],dataname,k,D,gamma,numiter,eta);
LWIDTH=5;
MSIZE=18;
% LWIDTH=8;
% MSIZE=24;
maxobj=0; maxtime=0; maxruntime = 0; minruntime = 10000000;
%% Plot for each method
for imethod=1:length(methods)
    
    method=methods{imethod};
    
    %% IF reading from or writing to a report file
    if(sum(strcmp(method,{'batch'})))
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,gamma=%f,numiter=%d].mat'],...
            dataname,method,k,gamma,numiter);
    elseif(sum(strcmp(method,{'nystrom'})))
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,m=%d,gamma=%f,numiter=%d].mat'],...
            dataname,method,k,m,gamma,numiter);
    elseif(sum(strcmp(method,{'rfftl'})))
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,D=%d,gamma=%f,numiter=%d].mat'],...
            dataname,method,k,D,gamma,numiter);
    elseif(sum(strcmp(method,{'rfsgd'})))
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,D=%d,gamma=%f,eta=%f,numiter=%d].mat'],...
            dataname,method,k,D,gamma,eta,numiter);
    end
    
    %% Fetch data if plotting from a report file
    if(exist(reportfile,'file'))
        load(reportfile,'seq','progress','avgprogress','avgtime');
        maxobj=max(max(avgprogress),maxobj);
        maxtime=max(max(avgtime),maxtime);
    else
        %% Set PAGE path and PAGE prefix
        pagepath=sprintf('../page/profile/SAKPCA/%s/%s/',...
            method,dataname);
        
        if(sum(strcmp(method,{'batch'})))
            pageprefix=@(method,rank,iter)[pagepath,...
                sprintf('%s[rank=%d,gamma=%f,iter=%d].mat',...
                method,rank,gamma,iter)];
        elseif(sum(strcmp(method,{'nystrom'})))
            pageprefix=@(method,rank,iter)[pagepath,...
                sprintf('%s[rank=%d,m=%d,gamma=%f,iter=%d].mat',...
                method,rank,m,gamma,iter)];
        elseif(sum(strcmp(method,{'rfftl'})))
            pageprefix=@(method,rank,iter)[pagepath,...
                sprintf('%s[rank=%d,D=%d,gamma=%f,iter=%d].mat',...
                method,rank,D,gamma,iter)];
        elseif(sum(strcmp(method,{'rfsgd'})))
            pageprefix=@(method,rank,iter)[pagepath,...
                sprintf('%s[rank=%d,D=%d,gamma=%f,eta=%f,iter=%d].mat',...
                method,rank,D,gamma,eta,iter)];
        end
        
        %% Set sequence points based on dataset and datasize
        [seq,L]=equilogseq(n,1);
        
        %% Initialize performance metrics
        progress=zeros(L(2),length(numiter));
        comptime=zeros(L(2),length(numiter));
        
        %% Gather performance metrics
        for iter=1:numiter
            load(pageprefix(method,k,iter),'objV','objVtune','runtime');
            progress(:,iter)=objV(1:L(2));
            comptime(:,iter)=runtime(1:L(2));
        end
        
        avgprogress=sum(progress,2)./numiter;
        avgtime=sum(comptime,2)./numiter;
        save(reportfile,'seq','progress','avgprogress','avgtime');
        maxobj=max(max(avgprogress),maxobj);
        maxtime=max(max(avgtime),maxtime);
    end
    runtimetotal=cumsum(avgtime);
    minruntime = min( minruntime, runtimetotal( 1 ) );
    maxruntime = max( maxruntime, runtimetotal( end ) );
    
    fig11=figure(11);
    ignore = semilogx(seq,avgprogress,'Color',...
        col{imethod}/255,'LineWidth',LWIDTH);
    hold on;
    set( get( get( ignore, 'Annotation' ), 'LegendInformation' ),...
        'IconDisplayStyle', 'off' );
    subseq = round( (1:15) * length(seq) / 15 + imethod-4);
    semilogx(seq(subseq),avgprogress(subseq),marker{imethod},...
        'MarkerFaceColor',col{imethod}/255,'MarkerSize',MSIZE);
    hold on;
    
    fig22=figure(22);
    ignore = semilogx(runtimetotal,avgprogress,'Color',...
        col{imethod}/255,'LineWidth',LWIDTH);
    hold on;
    set( get( get( ignore, 'Annotation' ), 'LegendInformation' ),...
        'IconDisplayStyle', 'off' );
    subseq = round( (1:15) * length(seq) / 15 + imethod-4);
    semilogx(runtimetotal(subseq),avgprogress(subseq),marker{imethod},...
        'MarkerFaceColor',col{imethod}/255,'MarkerSize',MSIZE);
    hold on;
    
end

%% PLOT TRUTH (if reading from a report then nothing to do)
method='truth';
reportfile=sprintf(['../page/profile/SAKPCA/%s/%s/%s[',...
    'rank=%d,gamma=%f].mat'],method,dataname,method,k,gamma);

load(reportfile,'objV');
trueobjV=objV;

FSIZE1=45; %70;
FSIZE2=50; %70;
FSIZE3=50; %40;

figure(11); semilogx(seq,repmat(trueobjV,size(seq)),'k-','LineWidth',LWIDTH); hold on;
grid; xlabel('Iteration','FontSize',FSIZE2);%,'Interpreter','Latex');
maxobj=max(maxobj,trueobjV);
axis([5 seq(end) 0 maxobj]);
ylabel('Objective','FontSize',FSIZE2);%,'Interpreter','Latex');
set(gca,'FontSize',FSIZE1,'XTick',[10^0 10^1 10^2 10^3 10^4 10^5],...
    'XMinorGrid','off');
if(k==5)
mylegend=legend({'ERM','Nystrom','RF-Oja','RF-ERM','truth'},...
    'Location','Northeast');
set(mylegend,'FontSize',FSIZE3);
end

figure(22);
hold on; grid;
xlabel('Time','FontSize',FSIZE2);
axis([0 maxruntime 0 inf]);
ylabel('Objective','FontSize',FSIZE2);%,'Interpreter','Latex');
set(gca,'FontSize',FSIZE1,'XTick',[10^0 10^1 10^2 10^3 10^4 10^5],...
    'XMinorGrid','off', 'YScale', 'log');
% if k==25
% mylegend=legend({'FTL','Nystrom','Oja+RFF','FTL+RFF'},...
%     'Location','Southeast');
% set(mylegend,'FontSize',FSIZE3);
% end



%% Create PDFs
topdf(fig11,fnames{1});
topdf(fig22,fnames{2});

end
