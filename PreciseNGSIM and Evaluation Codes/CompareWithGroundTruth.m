clear
close all
clc

% To compare with ground truth, part of these codes are from Coifman&Li(2017)

% I-80 data
load('CoifmanAndLi2017.mat'); % our newly re-extracted data
dataPartB=data;
dataPartB=sortrows(dataPartB,[1,2]);

rawNGSIMFilename = "trajectories-0400pm-0415pm.csv"; % the original NGSIM data
dataN = table2array(readtable(rawNGSIMFilename));

filename = "..\PostProcess\final_I80_trajectories-0400pm-0415pm.csv";
dataZ = table2array(readtable(filename));

idmax=max(dataZ(:,1));
dataPartB=dataPartB(dataPartB(:,1)<=idmax,:);  % these re-extracted vehicles should also exist in the original NGSIM data
% dataZ = dataZ(dataZ(:, 1) <= idmax, :);
traj_ids = unique(dataZ(:, 1));
% DataC6 = dataZ(dataZ(:, 1) == dataPartB(:, 1) & dataZ(:, 2) == dataPartB(:, 2), :);

TRAJ_PartB = [];
DataC6 = [];

for i = 1 : length(traj_ids)
    
    traj_id = traj_ids(i);
    traj_PartB = dataPartB(dataPartB(:, 1) == traj_id, :);
    traj_dataZ = dataZ(dataZ(:, 1) == traj_id, :);

    min_frame = max(min(traj_PartB(:, 2)), min(traj_dataZ(:, 2)));
    max_frame = min(max(traj_PartB(:, 2)), max(traj_dataZ(:, 2)));

    if max_frame - min_frame < 10

        continue

    end

    traj_PartB = traj_PartB(traj_PartB(:, 2) <= max_frame & traj_PartB(:, 2) >= min_frame, :); 
    traj_dataZ = traj_dataZ(traj_dataZ(:, 2) <= max_frame & traj_dataZ(:, 2) >= min_frame, :); 

    if isempty(TRAJ_PartB)

        TRAJ_PartB = traj_PartB;
        DataZ_C6 = traj_dataZ;

    else

        TRAJ_PartB = vertcat(TRAJ_PartB, traj_PartB);
        DataZ_C6 = vertcat(DataZ_C6, traj_dataZ);

    end

end

TRAJ_PartB = sortrows(TRAJ_PartB, [1 2]);
DataZ_C6 = sortrows(DataZ_C6, [1 2]);

%% figure 1 all trajectories
newvI=[0;find(diff(TRAJ_PartB(:,1)));length(TRAJ_PartB)];

dataO=TRAJ_PartB*0;   %  pre-initializing dataO
for ii=2:length(newvI)
    curVeh=dataN(dataN(:,1)==TRAJ_PartB(newvI(ii),1),:);
    curVeh(curVeh(:,2)<TRAJ_PartB(newvI(ii-1)+1,2),:)=[];
    curVeh(curVeh(:,2)>TRAJ_PartB(newvI(ii),2),:)=[];
    dataO((newvI(ii-1)+1):newvI(ii),:)=curVeh;
end

%%
vehEx2{1}=dataO(dataO(:,1)==1456,:);
vehEx2{2}=dataO(dataO(:,1)==1463,:);
vehEx2{3}=dataO(dataO(:,1)==1478,:);
vehEx2{4}=dataO(dataO(:,1)==1486,:);

vehEx2n{1}=TRAJ_PartB(TRAJ_PartB(:,1)==1456,:);
vehEx2n{2}=TRAJ_PartB(TRAJ_PartB(:,1)==1463,:);
vehEx2n{3}=TRAJ_PartB(TRAJ_PartB(:,1)==1478,:);
vehEx2n{4}=TRAJ_PartB(TRAJ_PartB(:,1)==1486,:);

vehEx2p{1}=DataZ_C6(DataZ_C6(:,1)==1456,:);
vehEx2p{2}=DataZ_C6(DataZ_C6(:,1)==1463,:);
vehEx2p{3}=DataZ_C6(DataZ_C6(:,1)==1478,:);
vehEx2p{4}=DataZ_C6(DataZ_C6(:,1)==1486,:);


figure(1)
for ii=1:4
    subplot(2,2,ii)
    plot(vehEx2{ii}(:,2)/10,vehEx2{ii}(:,12),'--',vehEx2n{ii}(:,2)/10,vehEx2n{ii}(:,12), '-.', vehEx2p{ii}(:,2)/10,vehEx2p{ii}(:,12))
    axis([470,510,0,25])
    xlabel('time (s)')
    ylabel('speed (ft/s)')
    title(['Trajectory ',num2str(vehEx2{ii}(1,1))])
end
subplot(2,2,1)
legend('raw NGSIM','GroundTruth', 'Precise')

%%

figure(2)
for ii=1:4
    vehSelf=vehEx2{ii};
    vehLeader=dataO(dataO(:,1)==vehSelf(1,15),:);

    vehSelfN=vehEx2n{ii};
    vehLeaderN=TRAJ_PartB(TRAJ_PartB(:,1)==vehSelfN(1,15),:);

    vehSelfP=vehEx2p{ii};
    vehSelfP = vehSelfP(vehSelfP(:, 2) >= min(vehSelfN(:, 2)) & vehSelfP(:, 2) <= max(vehSelfN(:, 2)), :);
    vehLeaderP=DataZ_C6(DataZ_C6(:,1)==vehSelfP(1,15),:);
    vehLeaderP = vehLeaderP(vehLeaderP(:, 2) >= min(vehSelfN(:, 2)) & vehLeaderP(:, 2) <= max(vehSelfN(:, 2)), :); 
    
    if vehLeaderN(1,2)<vehSelfN(1,2)
        vehLeader(vehLeader(:,2)<vehSelf(1,2),:)=[];   % getting rid of leader data before follower
        vehLeaderN(vehLeaderN(:,2)<vehSelfN(1,2),:)=[];   % getting rid of leader data before follower
        vehLeaderP(vehLeaderP(:,2)<vehSelfP(1,2),:)=[];   % getting rid of leader data before follower
    end

    vehSelf(vehSelf(:,2)>max(vehLeader(:,2)),:)=[];   % getting rid of follower data after leader
    vehSelfN(vehSelfN(:,2)>max(vehLeaderN(:,2)),:)=[];   % getting rid of follower data after leader
    vehSelfP(vehSelfP(:,2)>max(vehLeaderP(:,2)),:)=[];   % getting rid of follower data after leader
    
    subplot(2,2,ii)
    plot(vehLeader(:,6)-vehLeader(:,9)-vehSelf(:,6)+vehSelf(:,9),vehSelf(:,12),'.-',vehLeaderN(:,6)-vehSelfN(:,6),vehSelfN(:,12),'.-', vehLeaderP(:,6) - vehLeaderP(:,9) - vehSelfP(:,6) + vehSelfP(:,9), vehSelfP(:,12),'.-')
    ax=axis;
    axis([ax(1:2),0,25])
    xlabel('spacing (ft)')
    ylabel('speed (ft/s)')
    title(['Trajectory ',num2str(vehEx2{ii}(1,1))])
end
subplot(2,2,1)
legend('raw NGSIM','GroundTruth', 'Precise', 'Location','northwest')



%%

figure(86)  % histogram will open a plot, so do it in this temp figure
posErr=DataZ_C6(:,6)-DataZ_C6(:,9)-TRAJ_PartB(:,6);
spdErr=DataZ_C6(:,12)-TRAJ_PartB(:,12);

bins=-10.5:1:10.5;
binC=-10:1:10;

binV=0:5:50;
binVC=2.5:5:47.5;

binY=1240:10:1520;
binYC=1245:10:1515;

binL=5:10:75;
binLC=10:10:70;

h=histogram(posErr,bins);
pdfY=h.Values/length(posErr);

h=histogram(spdErr,bins);
pdfV=h.Values/length(spdErr);

for ii=1:6
    uu=DataZ_C6(:,14)==ii;
    
    posLerr=posErr(uu);
    spdLerr=spdErr(uu);
    
    h=histogram(posLerr,bins);
    pdfY(ii+1,1:21)=h.Values/length(posLerr);
    
    h=histogram(spdLerr,bins);
    pdfV(ii+1,1:21)=h.Values/length(spdLerr);
end
close(86)


for jj=1:length(binV)-1
    vv=TRAJ_PartB(:,12)>=binV(jj)&TRAJ_PartB(:,12)<binV(jj+1);
    if sum(vv)>50
        PvsV(jj)=mean(posErr(vv));
        SvsV(jj)=mean(spdErr(vv));
    else
        PvsV(jj)=nan;
        SvsV(jj)=nan;
    end
end

for ii=1:6
    uu=dataO(:,14)==ii;
    for jj=1:length(binV)-1
        vv=TRAJ_PartB(:,12)>=binV(jj)&TRAJ_PartB(:,12)<binV(jj+1)&uu;
        if sum(vv)>50
            PvsV(ii+1,jj)=mean(posErr(vv));
            SvsV(ii+1,jj)=mean(spdErr(vv));
        else
            PvsV(ii+1,jj)=nan;
            SvsV(ii+1,jj)=nan;
        end
    end
end




for jj=1:length(binY)-1
    vv=TRAJ_PartB(:,6)>=binY(jj)&TRAJ_PartB(:,6)<binY(jj+1);
    if sum(vv)>50
        PvsY(jj)=mean(posErr(vv));
        SvsY(jj)=mean(spdErr(vv));
    else
        PvsY(jj)=nan;
        SvsY(jj)=nan;
    end
end

for ii=1:6
    uu = DataZ_C6(:,14)==ii;
    for jj=1:length(binY)-1
        vv=TRAJ_PartB(:,6)>=binY(jj)&TRAJ_PartB(:,6)<binY(jj+1)&uu;
        if sum(vv)>50
            PvsY(ii+1,jj)=mean(posErr(vv));
            SvsY(ii+1,jj)=mean(spdErr(vv));
        else
            PvsY(ii+1,jj)=nan;
            SvsY(ii+1,jj)=nan;
        end
    end
end



clear PvsL SvsL

for jj=1:length(binL)-1
    vv=DataZ_C6(:,9)>=binL(jj)&DataZ_C6(:,9)<binL(jj+1);
    if sum(vv)>50
        PvsL(jj)=mean(posErr(vv));
        SvsL(jj)=mean(spdErr(vv));
    else
        PvsL(jj)=nan;
        SvsL(jj)=nan;
    end
end

for ii=1:6
    uu=DataZ_C6(:,14)==ii;
    for jj=1:length(binL)-1
        vv=DataZ_C6(:,9)>=binL(jj)&DataZ_C6(:,9)<binL(jj+1)&uu;
        if sum(vv)>50
            PvsL(ii+1,jj)=mean(posErr(vv));
            SvsL(ii+1,jj)=mean(spdErr(vv));
        else
            PvsL(ii+1,jj)=nan;
            SvsL(ii+1,jj)=nan;
        end
    end
end



figure(3)
subplot(211)
plot(binC,pdfY)
legend('all lanes','lane 1','lane 2','lane 3','lane 4','lane 5','lane 6')
grid on
axis([-10,10,0,0.15])
xlabel('position error (ft)')
ylabel('pdf')

subplot(212)
plot(binVC,PvsV,'.-')
axis([0,45,-4,8])
xlabel('speed (ft/s)')
ylabel('avg position error (ft)')
grid on


%%

figure(4)
subplot(211)
plot(binC,pdfV)
legend('all lanes','lane 1','lane 2','lane 3','lane 4','lane 5','lane 6')
grid on
axis([-6,6,0,0.4])
xlabel('speed error (ft/s)')
ylabel('pdf')

subplot(212)
plot(binVC,SvsV,'.-')
axis([0,45,-0.5,2.5])
xlabel('speed (ft/s)')
ylabel('avg speed error (ft/s)')
grid on


%% comparison on lane 4

Colmat = hsv();

figure(5)
subplot(212)
ln=4;
dataX=dataZ(dataZ(:,14)==ln,:);
bb=find(diff(dataX(:,2))~=1);
dataX(bb+1,:)=nan;

traj_ids = unique(dataX(:, 1));

for idx = 1 : length(traj_ids)

    id = traj_ids(idx);

    if isnan(id)

        continue

    end

    traj_temp = dataX(dataX(:, 1) == id, :);

    x = traj_temp(:,2)'/10; 

    y1 = traj_temp(:,6)';

    y2 = y1-traj_temp(:,9)';
    
    k = mod(id, 50);
    Colour = Colmat(mod(7*(k-1), 64)+1, :)*(0.9-0.3*(  mod(floor(7*(k-1)/64), 3))); % --->Brightness

    patch([x fliplr(x)], [y1  fliplr(y2)], Colour, 'FaceAlpha',.5)

    hold on

%     plot(x, y1, 'b-', 'LineWidth', 1.5);
%     plot(x, y2, 'm-', 'LineWidth', 1.5);
% 
%     hold on

end

hold off
axis([770, 820, 600,1050])
xlabel('time (s)')
ylabel('dist (ft)')
title(['Precise NGSIM lane ',num2str(ln)])
%, 'Front bumper','Rear bumper')


subplot(211)
ln=4;
dataX=dataN(dataN(:,14)==ln,:);
bb=find(diff(dataN(:,2))~=1);
dataX(bb+1,:)=nan;

traj_ids = unique(dataX(:, 1));

for idx = 1 : length(traj_ids)

    id = traj_ids(idx);

    if isnan(id)

        continue

    end

    traj_temp = dataX(dataX(:, 1) == id, :);

    x = traj_temp(:,2)'/10; 

    y1 = traj_temp(:,6)';

    y2 = y1-traj_temp(:,9)';

    k = mod(id, 50);
    Colour = Colmat( mod(7*(k-1), 64)+1, :)*(0.9-0.3*(mod(floor(7*(k-1)/64), 3)));

    patch([x fliplr(x)], [y1  fliplr(y2)], Colour, 'FaceAlpha',.5)

    hold on

%     plot(x, y1, 'b-', 'LineWidth', 1.5);
%     plot(x, y2, 'm-', 'LineWidth', 1.5);
% 
%     hold on

end

hold off
axis([770, 820, 600,1050])
xlabel('time (s)')
ylabel('dist (ft)')
title(['Raw NGSIM lane ',num2str(ln)])
% legend('Vehicle Strand', 'Front bumper','Rear bumper')
legend('Vehicle Strand') 
%%
veh_ids = {2846, 2847, 2855, 2868, 2874, 2885, 2898, 2893};
vehs_val = {};
vehs_val_n = {};

for val_idx = 1 : length(veh_ids)

    veh_id = veh_ids{val_idx};

    vehs_val{val_idx} = dataZ(dataZ(:,1)==veh_id,:);

    vehs_val_n{val_idx} = dataN(dataN(:,1)==veh_id,:);

end

figure(6)
for ii=1:length(veh_ids)
    subplot(2,4,ii)
    plot(vehs_val{ii}(:,2)/10,vehs_val{ii}(:,12),'-',vehs_val_n{ii}(:,2)/10,vehs_val_n{ii}(:,12), '--')
    axis([780,820,0,35])
    xlabel('time (s)')
    ylabel('speed (ft/s)')
    title(['Trajectory ',num2str(vehs_val{ii}(1,1))])

end
subplot(2,4,1)

legend('Precise NGSIM', 'Raw NGSIM', 'Location','northwest')