%% platoon Consistency
clear
close all
clc

US101_periods = {'0750am-0805am', '0805am-0820am', '0820am-0835am'}; 
I80_periods = {'0400pm-0415pm', '0500pm-0515pm', '0515pm-0530pm'};

minTrajectoryLength = 100;

US101_periods_records = zeros(3, 4);
I80_periods_records = zeros(3, 4);

for period_num = 1 : 3

    US101_period = US101_periods{period_num};
    polyfit_filename = sprintf('final_US101_trajectories-%s.csv', US101_period);

    trajectoryData = table2array(readtable(polyfit_filename));
    
    % Sort trajectory data by vehicle ID and frame ID
    trajectoryData = sortrows(trajectoryData, [1, 2]);
    
    epsilon_PS = [];

    for lane = 1 : 5

        pairVehicleData = getTrajectoryPairs(trajectoryData, lane, minTrajectoryLength);  % |Pair_no|Leader_id Leader_position Leader_speed|Follower_id Follower_position Follower_speed| 
        
        % Compute platoon consistency for each vehicle pair
        vehiclePairIDs = unique(pairVehicleData(:,1));
        numVehicles = numel(vehiclePairIDs);
        pairConsistency = zeros(numVehicles, 1);
        
        for i = 1:numVehicles
        
            vehiclePairID = vehiclePairIDs(i);
            vehiclePairData = pairVehicleData(pairVehicleData(:,1) == vehiclePairID, :);
            leader_data = vehiclePairData(:, 3:4);
            follower_data = vehiclePairData(:, 6:7);
            sn0 = leader_data(1, 1);
            sp0 = follower_data(1, 1);
        
            t = 1 : size(vehiclePairData, 1);

            t = t/10;
    
            epsilon_npt_PS = zeros(size(t));
        
            for j = 1:numel(t)

                if j == 1

                    epsilon_npt_PS(j) = 0;

                else

                    % Compute integrals for subject vehicle (n) and following vehicle (p)
                    integral_vnt = trapz(t(1:j), leader_data(1:j, 2)); % assuming vnt is the speed of subject vehicle at each time step
                    integral_vpt = trapz(t(1:j), follower_data(1:j, 2)); % assuming vpt is the speed of following vehicle at each time step
                    
                    % Compute platoon consistency for the current time step
                    epsilon_npt_PS(j) = (leader_data(j, 1) - follower_data(j, 1)) - (sn0 - sp0 + integral_vnt - integral_vpt);

                end

            end

    
            epsilon_PS = [epsilon_PS epsilon_npt_PS];


        end

    end


    RMSE_bias = sqrt(mean(epsilon_PS.^2));
    max_bias = max(epsilon_PS);
    min_bias = min(epsilon_PS);
    mean_bias = mean(epsilon_PS);

    US101_periods_records(period_num, 1) = max_bias;
    US101_periods_records(period_num, 2) = min_bias;
    US101_periods_records(period_num, 3) = mean_bias;
    US101_periods_records(period_num, 4) = RMSE_bias;

end

US101_periods_records = US101_periods_records.';

US101_NEW_Platoon_Consistency = array2table(US101_periods_records);
% Default heading for the columns will be A1, A2 and so on. 
% You can assign the specific headings to your table in the following manner
US101_NEW_Platoon_Consistency.Properties.VariableNames(1:3) = US101_periods;
US101_NEW_Platoon_Consistency.Properties.RowNames = {'max_bias' 'min_bias', 'mean_bias', 'RMSE_bias'};

%%
I80_periods = {'0400pm-0415pm', '0500pm-0515pm', '0515pm-0530pm'};
minTrajectoryLength = 100;
I80_periods_records = zeros(3, 4);

for period_num = 1 : 3

    I80_period = I80_periods{period_num};
    polyfit_filename = sprintf('final_I80_trajectories-%s.csv', I80_period);
    trajectoryData = table2array(readtable(polyfit_filename));
    
    % Sort trajectory data by vehicle ID and frame ID
    trajectoryData = sortrows(trajectoryData, [1, 2]);

    epsilon_PS = [];
    minTrajectoryLength = 100;


    for lane = 1 : 5
    
        pairVehicleData = getTrajectoryPairs(trajectoryData, lane, minTrajectoryLength);  % |Pair_no|Leader_position Leader_speed|Follower_position Follower_speed| 
        
        % Compute platoon consistency for each vehicle pair
        vehiclePairIDs = unique(pairVehicleData(:,1));
        numVehicles = numel(vehiclePairIDs);
        pairConsistency = zeros(numVehicles, 1);
        epsilon_npt_PS = {};
        
        for i = 1:numVehicles
        
            vehiclePairID = vehiclePairIDs(i);
            vehiclePairData = pairVehicleData(pairVehicleData(:,1) == vehiclePairID, :);
            leader_data = vehiclePairData(:, 3:4);
            follower_data = vehiclePairData(:, 6:7);
            sn0 = leader_data(1, 1);
            sp0 = follower_data(1, 1);
        
            t = 1 : size(vehiclePairData, 1);
            t = t/10;
        
            epsilon_npt_PS = zeros(size(t));
        
             for j = 1 : numel(t)

                if j == 1

                    epsilon_npt_PS(j) = 0;

                else

                    % Compute integrals for subject vehicle (n) and following vehicle (p)
                    integral_vnt = trapz(t(1:j), leader_data(1:j, 2)); % assuming vnt is the speed of subject vehicle at each time step
                    integral_vpt = trapz(t(1:j), follower_data(1:j, 2)); % assuming vpt is the speed of following vehicle at each time step
                    
                    % Compute platoon consistency for the current time step
                    epsilon_npt_PS(j) = (leader_data(j, 1) - follower_data(j, 1)) - (sn0 - sp0 + integral_vnt - integral_vpt);

                end

             end

            epsilon_PS = [epsilon_PS epsilon_npt_PS];
        
        end

    end

    RMSE_bias = sqrt(mean(epsilon_PS.^2));
    max_bias = max(epsilon_PS);
    min_bias = min(epsilon_PS);
    mean_bias = mean(epsilon_PS);


    I80_periods_records(period_num, 1) = max_bias;
    I80_periods_records(period_num, 2) = min_bias;
    I80_periods_records(period_num, 3) = mean_bias;
    I80_periods_records(period_num, 4) = RMSE_bias;

end

I80_periods_records = I80_periods_records';


I80_NEW_Platoon_Consistency = array2table(I80_periods_records);
% Default heading for the columns will be A1, A2 and so on. 
% You can assign the specific headings to your table in the following manner
I80_NEW_Platoon_Consistency.Properties.VariableNames(1:3) = I80_periods;
I80_NEW_Platoon_Consistency.Properties.RowNames = {'max_bias' 'min_bias', 'mean_bias', 'RMSE_bias'};

%%
I80_NEW_Platoon_Consistency
US101_NEW_Platoon_Consistency
