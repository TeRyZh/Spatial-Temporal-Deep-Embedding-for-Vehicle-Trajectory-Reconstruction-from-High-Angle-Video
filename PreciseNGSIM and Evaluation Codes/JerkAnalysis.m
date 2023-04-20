clear
close all
clc

US101_periods = {'0750am-0805am', '0805am-0820am', '0820am-0835am'}; 
I80_periods = {'0400pm-0415pm', '0500pm-0515pm', '0515pm-0530pm'};


US101_periods_records = zeros(3, 4);
I80_periods_records = zeros(3, 4);

num_delta_t = 1;
jerk_threshold = 49.2126; % ft/s3
window_size = 10;

for period_num = 1 : 3

    anormaly_percent = 0;
    min_jerk = 0;
    max_jerk = 0;
    sign_jerk = 0;

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
    accels = ngsim_data(:, 13); % array of vehicle speeds (in m/s)

    % Compute timestamps from frames (assuming each frame is 0.1 seconds)
    timestamps = frames * 0.1;
    
    % Initialize arrays to store internal consistency results
    jerks_ngsim = {}; % array to store internal consistency values for each vehicle

    anomaly_signs = 0;

    window_cnt = 0;

    for i = 1:numVehicles
        vehicleID = vehicleIDs(i);
        veh_indices = find(veh_ids == vehicleID); % indices of rows corresponding to the current vehicle

        if numel(veh_indices) < 100
            continue
        end

        accels_veh = accels(veh_indices);

        jerks = diff(accels_veh)/(num_delta_t * 0.1);
        
        for j = 1 : (length(jerks) - window_size)

            window_cnt = window_cnt + 1;

            % Find data points within the window size
            jerk_window = jerks(j : j + window_size);
            
            % Compute acceleration and jerk values within the window
            num_jerk_changes = sum(diff(sign(jerk_window)) ~= 0);
        
            if num_jerk_changes > 1
    
                anomaly_signs = anomaly_signs + 1; % set anomaly flag to 1 if there are more than one sign changes in jerk
        
            end

        end
        
        % Compute internal consistency value for the current vehicle at each time step
        jerks_ngsim{i} = jerks;

    end

    anomaly_signs_pecent = anomaly_signs/window_cnt;

    jerks_ngsim = cat(1, jerks_ngsim{:});

    max_jerk = max(jerks_ngsim);
    min_jerk = min(jerks_ngsim);
    anormaly_percent = sum(abs(jerks_ngsim) > jerk_threshold)/length(jerks_ngsim);

    US101_periods_records(period_num, 1) = max_jerk;
    US101_periods_records(period_num, 2) = min_jerk;
    US101_periods_records(period_num, 3) = anormaly_percent * 100;
    US101_periods_records(period_num, 4) = anomaly_signs_pecent * 100;

end
%%

for period_num = 1 : 3

    max_jerk = -10;
    min_jerk = 1000;

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
    accels = ngsim_data(:, 13); % array of vehicle speeds (in m/s)

    % Compute timestamps from frames (assuming each frame is 0.1 seconds)
    timestamps = frames * 0.1;
    
    % Initialize arrays to store internal consistency results
    jerks_ngsim = {}; % array to store internal consistency values for each vehicle

    anomaly_signs = 0;

    window_cnt = 0;

    for i = 1:numVehicles
        vehicleID = vehicleIDs(i);
        veh_indices = find(veh_ids == vehicleID); % indices of rows corresponding to the current vehicle

        if numel(veh_indices) < 100
            continue
        end

        accels_veh = accels(veh_indices);

        jerks = diff(accels_veh)/(num_delta_t * 0.1);

%         if max(jerks) > 80 || min(jerks) < -80
% 
%             I80_period
% 
%             vehicleID
% 
%             continue
% 
%         end

        
        for j = 1 : (length(jerks) - window_size)

            window_cnt = window_cnt + 1;

            % Find data points within the window size
            jerk_window = jerks(j : j + window_size);
            
            % Compute acceleration and jerk values within the window
            num_jerk_changes = sum(diff(sign(jerk_window)) ~= 0);
        
            if num_jerk_changes > 1
    
                anomaly_signs = anomaly_signs + 1; % set anomaly flag to 1 if there are more than one sign changes in jerk
        
            end

        end
        
        % Compute internal consistency value for the current vehicle at each time step
        jerks_ngsim{i} = jerks;

    end

    anomaly_signs_pecent = anomaly_signs/window_cnt;

    jerks_ngsim = cat(1, jerks_ngsim{:});

    max_jerk = max(jerks_ngsim);
    min_jerk = min(jerks_ngsim);
    anormaly_percent = sum(abs(jerks_ngsim) > jerk_threshold)/length(jerks_ngsim);

    I80_periods_records(period_num, 1) = max_jerk;
    I80_periods_records(period_num, 2) = min_jerk;
    I80_periods_records(period_num, 3) = anormaly_percent * 100;
    I80_periods_records(period_num, 4) = anomaly_signs_pecent * 100;

end

I80_periods_records = I80_periods_records';
US101_periods_records = US101_periods_records';

%%
US101_NEW_Jerk = array2table(US101_periods_records);
% Default heading for the columns will be A1, A2 and so on. 
% You can assign the specific headings to your table in the following manner
US101_NEW_Jerk.Properties.VariableNames(1:3) = US101_periods;
US101_NEW_Jerk.Properties.RowNames = {'max_jerk' 'min_jerk', 'anormaly_percentage', 'anomaly_signs_pecent'};

I80_NEW_Jerk = array2table(I80_periods_records);
% Default heading for the columns will be A1, A2 and so on. 
% You can assign the specific headings to your table in the following manner
I80_NEW_Jerk.Properties.VariableNames(1:3) = I80_periods;
I80_NEW_Jerk.Properties.RowNames = {'max_jerk' 'min_jerk', 'anormaly_percentage', 'anomaly_signs_pecent'};

US101_NEW_Jerk
I80_NEW_Jerk