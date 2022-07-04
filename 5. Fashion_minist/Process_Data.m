x_sam = xlsread('data.xlsx','Sample','E2:J301');
x_prd = xlsread('data.xlsx','Predict','E2:J51');


% 创造样本矩阵
X = zeros(10,30,6);
Y = zeros(10,5,6);

for i=1:10
    X(i,:,:) = x_sam((i-1)*30+1:(i-1)*30+30,:);
    Y(i,:,:) = x_sam((i-1)*5+1:(i-1)*5+5,:);
end

for i=1:10
    X(i,:,:) = zscore(X(i,:,:));
    Y(i,:,:) = zscore(Y(i,:,:));
end

for i=1:10
    [coeff,score,latent] = pca(squeeze(X(i,:,:)));
    eff(i,:,:)=coeff;
    con(:,i)=latent;
end

figure(1)
for i=1:6
    plot([1:6],con(:,i),'-*')
    hold on
end

fineff = squeeze(mean(eff,1));
fincon = mean(con,2);
for i = 1:length(fincon)
    perc(i) = fincon(i)/sum(fincon);
end
plot([1:6],fincon,'-*','linewidth',2)

y = zeros(10,30);
for i=1:10
    y(i,:)=(squeeze(X(i,:,:))*fineff(:,1))';
end
figure(2)
for i = 1:10
    plot([1:30],y(i,:),'LineWidth',1)
    hold on
end

figure(3)
for i=1:10
    X(i,:,:) = x_sam((i-1)*30+1:(i-1)*30+30,:);
    Y(i,:,:) = x_sam((i-1)*5+1:(i-1)*5+5,:);
end
t=linspace(1,30,30)
y=(squeeze(X(4,:,:))*fineff(:,1))'
plot(t,y,'LineWidth',1)

save('result.mat','t','y','fineff','fincon','perc')







