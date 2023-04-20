clear
close all
clc

US101_periods = {'0750am-0805am', '0805am-0820am', '0820am-0835am'}; 
I80_periods = {'0400pm-0415pm', '0500pm-0515pm', '0515pm-0530pm'};


US101_periods_records = zeros(3, 4);
I80_periods_records = zeros(3, 4);

for period_num = 1 : 3

    max_bias = -10;
    min_bias = 1000;
    mean_bias = 0;
    RMSE_bias = 0;

    US101_period = US101_periods{period_num};
    polyfit_filename = sprintf('final_US101_trajectories-%s.csv', US101_period);

    ngsim_data = table2array(readtable(polyfit_filename));

%     ngsim_data = ngsim_data(ngsim_data(:, 14) < 6, :);

    % Sort trajectory data by vehicle ID and frame ID
    ngsim_data = sortrows(ngsim_data, [1, 2]);
    
    % Compute speed consistency for each vehicle
    vehicleIDs = unique(ngsim_data(:,1));
    numVehicles = numel(vehicleIDs);

    % Extract relevant data from the dataset
    veh_ids = ngsim_data(:, 1); % array of vehicle IDs
    frames = ngsim_data(:, 2); % array of frames (timestamps)
    distances = ngsim_data(:, 6); % array of vehicle distances (in meters)
    speeds = ngsim_data(:, 12); % array of vehicle speeds (in m/s)

    % Compute timestamps from frames (assuming each frame is 0.1 seconds)
    timestamps = frames * 0.1;
    
    % Initialize arrays to store internal consistency results
    epsilon_s = {}; % array to store internal consistency values for each vehicle

    for i = 1:numVehicles
        vehicleID = vehicleIDs(i);
        veh_indices = find(veh_ids == vehicleID); % indices of rows corresponding to the current vehicle

        if numel(veh_indices) < 100
            continue
        end

        position = distances(veh_indices);
        speed = speeds(veh_indices);
        
        % Compute integral of speed with respect to time
        integral_speed = cumtrapz(timestamps(veh_indices), speed);
        
        % Compute estimated position based on the integral of speed
        estimated_position = position(1) + integral_speed;
        
        % Compute internal consistency value for the current vehicle at each time step
        epsilon_s{i} = position - estimated_position;

    end

    epsilon_s = cat(1, epsilon_s{:});

    RMSE_bias = sqrt(mean(epsilon_s.^2));
    max_bias = max(epsilon_s);
    min_bias = min(epsilon_s);
    mean_bias = mean(epsilon_s);

    US101_periods_records(period_num, 1) = max_bias;
    US101_periods_records(period_num, 2) = min_bias;
    US101_periods_records(period_num, 3) = mean_bias;
    US101_periods_records(period_num, 4) = RMSE_bias;

end


for period_num = 1 : 3

    max_bias = -10;
    min_bias = 1000;
    mean_bias = 0;
    RMSE_bias = 0;

    I80_period = I80_periods{period_num};
    polyfit_filename = sprintf('final_I80_trajectories-%s.csv', I80_period);

    ngsim_data = table2array(readtable(polyfit_filename));

%     ngsim_data = ngsim_data(ngsim_data(:, 14) < 6, :);

    % Sort trajectory data by vehicle ID and frame ID
    ngsim_data = sortrows(ngsim_data, [1, 2]);
    
    % Compute speed consistency for each vehicle
    vehicleIDs = unique(ngsim_data(:,1));
    numVehicles = numel(vehicleIDs);

    % Extract relevant data from the dataset
    veh_ids = ngsim_data(:, 1); % array of vehicle IDs
    frames = ngsim_data(:, 2); % array of frames (timestamps)
    distances = ngsim_data(:, 6); % array of vehicle distances (in meters)
    speeds = ngsim_data(:, 12); % array of vehicle speeds (in m/s)

    % Compute timestamps from frames (assuming each frame is 0.1 seconds)
    timestamps = frames * 0.1;
    
    % Initialize arrays to store internal consistency results
    epsilon_s = {}; % array to store internal consistency values for each vehicle

    for i = 1:numVehicles
        vehicleID = vehicleIDs(i);
        veh_indices = find(veh_ids == vehicleID); % indices of rows corresponding to the current vehicle

        if numel(veh_indices) < 100
            continue
        end

        position = distances(veh_indices);
        speed = speeds(veh_indices);
        
        % Compute integral of speed with respect to time
        integral_speed = cumtrapz(timestamps(veh_indices), speed);
        
        % Compute estimated position based on the integral of speed
        estimated_position = position(1) + integral_speed;
        
        % Compute internal consistency value for the current vehicle at each time step
        
        epsilon_s{i} = position - estimated_position;


    end

    epsilon_s = cat(1, epsilon_s{:});

    RMSE_bias = sqrt(mean(epsilon_s.^2));
    max_bias = max(epsilon_s);
    min_bias = min(epsilon_s);
    mean_bias = mean(epsilon_s);

    I80_periods_records(period_num, 1) = max_bias;
    I80_periods_records(period_num, 2) = min_bias;
    I80_periods_records(period_num, 3) = mean_bias;
    I80_periods_records(period_num, 4) = RMSE_bias;

end
I80_periods_records = I80_periods_records';
US101_periods_records = US101_periods_records';

%%
US101_NEW_Internal_Consistency = array2table(US101_periods_records);
% Default heading for the columns will be A1, A2 and so on. 
% You can assign the specific headings to your table in the following manner
US101_NEW_Internal_Consistency.Properties.VariableNames(1:3) = US101_periods;
US101_NEW_Internal_Consistency.Properties.RowNames = {'max_bias' 'min_bias', 'mean_bias', 'RMSE_bias'};


I80_NEW_Internal_Consistency = array2table(I80_periods_records);
% Default heading for the columns will be A1, A2 and so on. 
% You can assign the specific headings to your table in the following manner
I80_NEW_Internal_Consistency.Properties.VariableNames(1:3) = I80_periods;
I80_NEW_Internal_Consistency.Properties.RowNames = {'max_bias' 'min_bias', 'mean_bias', 'RMSE_bias'};

I80_NEW_Internal_Consistency
US101_NEW_Internal_Consistency
