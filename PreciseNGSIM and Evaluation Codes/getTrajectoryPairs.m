function table = getTrajectoryPairs(data, lane, minTrajectoryLenght)

    % Filter the data based on the lane and make sure follower - leader
    % fields are not -1
    lane_data = data((data(:,14)==lane & data(:,15) > 0 & data(:,16)> 0 & data(:,11) ~= 1),:);
    unique_vehicles = unique(lane_data(:,1));

    pairArr = [];
    leaderArr = [];
    followerArr = [];
    timeArr = [];

    count = 0;

    for i=1:(length(unique_vehicles)-1)
        % Filter data based on follower and leader fields
        leader_data = lane_data(lane_data(:,1)==unique_vehicles(i),:);

        unique_followers = unique(leader_data(:,16));

        for j = 1 : length(unique_followers)
        
            follower_data = lane_data((lane_data(:,1)==unique_followers(j) & ...
                lane_data(:,15)==unique_vehicles(i)),:);
    
            % Find the common start - end times of the trajectory
            maxStartTime = max(min(leader_data(:,2)), min(follower_data(:,2)));
            minEndTime = min(max(leader_data(:,2)), max(follower_data(:,2)));
    
            if (maxStartTime < minEndTime)
                % Filter leader and follower data based on the start-end time 
                leader_data_seg = leader_data(leader_data(:,2)>=maxStartTime & ...
                leader_data(:,2)<=minEndTime,:);
                follower_data_seg = follower_data(follower_data(:,2)>=maxStartTime & ...
                    follower_data(:,2)<=minEndTime,:);
    
%                 time = 1:length(leader_data_seg);

                frame_diffs = diff(leader_data_seg(:, 2)); % it should consecutive following

                % Check minimum trajectory length and data lenght 
                if(length(leader_data_seg)>minTrajectoryLenght && ...
                        length(leader_data_seg)==length(follower_data_seg) &&...
                        all(frame_diffs == 1))
    
                    count = count + 1;
                    % Accumulate results in respective arrays
                    leaderArr = [leaderArr; leader_data_seg(:, [1 6 12])];
                    followerArr = [followerArr; follower_data_seg(:, [1 6 12])];
%                     timeArr = [timeArr; time'];
                    pairArr = [pairArr; repmat(count, length(leader_data_seg), 1)];
                end   
            end
        end
    end
    % Save the data as a matrix. 
    % Matrix format: |Pair_no|Leader_position|Follower_position|time| 
    table = [pairArr, leaderArr, followerArr] ;