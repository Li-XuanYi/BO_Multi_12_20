load EEI_DATA.mat
y_ = EEI_DATA{5}.y(:,1:2);
x_ = EEI_DATA{5}.x;
[FrontValue,~] = NonDominateSort(y_,1);     
Next = find(FrontValue==1);
PF_ = y_(Next,:);
PS_ = y_(Next,:);
PF_2obj(:,1) = PF_(:,1);
PF_2obj(:,2) = PF_(:,2);
hold on
scatter(PF_(:,1),PF_(:,2),'rx')

%% %%%%%%%%%%%%%%%%%%%%%% Build Inverse Models %%%%%%%%%%%%%%%%%%%%%%
% Normalization
norm_PF_2obj = (PF_2obj - min(PF_2obj) + 1e-10)./(max(PF_2obj) - min(PF_2obj) + 1e-10);
c = sum(norm_PF_2obj,2)./norm_PF_2obj;
w = c./sum(c,2);
% Build Indpendent Inverse GP Models
[m,d] = size(PS_);
for i = 1:d
    inverse_model{i} = fitrgp(w,PS_(:,i));
end
