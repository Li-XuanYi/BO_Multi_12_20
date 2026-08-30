function Cal_DB_HV
    load ParEGO_DATA.mat
    for j = 1:5
        for i = 21:120
            y_sub = ParEGO_DATA{j}.y(1:i,:);
            y_cal_hv = (y_sub - [680,290,8e-10])./([5800,350,3e-09] - [700,290,2.5e-10]);
            HV(i-20) = Hypervolume(y_cal_hv,[1.1,1.1,1.1]);
        end
        HV_ParEGO(j,:) = HV;
    end
    errorbar(mean(HV_ParEGO),std(HV_ParEGO),'b')
%     plot(HV_ParEGO,'b')
    
    load EEI_DATA.mat
    for j = 1:5
        for i = 21:120
            y_sub = EEI_DATA{j}.y(1:i,:);
            y_cal_hv = (y_sub - [680,290,8e-10])./([5800,350,3e-09] - [700,290,2.5e-10]);
            HV(i-20) = Hypervolume(y_cal_hv,[1.1,1.1,1.1]);
        end
        HV_EEI(j,:) = HV;
    end
    hold on
    errorbar(mean(HV_EEI),std(HV_EEI),'r')
%     plot(HV_EEI,'r')
    legend('ParEGO','EEI-ParEGO')
    xlabel('Iterations')
    ylabel('Hypervolume')
    set(gca,'fontname','Cambria','fontsize',13);
end