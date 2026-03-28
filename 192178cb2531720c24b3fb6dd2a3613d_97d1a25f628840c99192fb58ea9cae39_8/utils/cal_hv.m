function HV = cal_hv(EEI_DATA,REFER_UP,REFER_LOW)
HV = zeros(1,200);
count = 1;
for i = 20:1:200
    y_sub = EEI_DATA.y(1:i,:);
    y_cal_hv = (y_sub - REFER_LOW)./(REFER_UP - REFER_LOW);
    HV(count) = Hypervolume(y_cal_hv,[1,1,1]);
    count = count + 1;
    fprintf("%d\n",i);
end
%HV_EEI(:) = HV;
%hold on
%errorbar(mean(HV_EEI),std(HV_EEI),'r')
%xlabel('Iterations')
%ylabel('Hypervolume')
%set(gca,'fontname','Cambria','fontsize',13);
end

