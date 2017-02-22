function result_time= Lpq_weight(start_frame, stop_frame, ...
    power_syllable, power_time, frame_disturbance, time_weight)

global NUMBER_OF_PSQM_FRAMES_PER_SYLLABE

% fid= fopen( 'tmp_mat1.txt', 'at');
% fprintf( 'result_time:\n');

result_time= 0;
total_time_weight_time = 0;
 %fprintf( 'start/end frame: %d/%d\n', start_frame, stop_frame);
for start_frame_of_syllable = start_frame: ...
        NUMBER_OF_PSQM_FRAMES_PER_SYLLABE/2: stop_frame
    result_syllable = 0;
    count_syllable = 0;
    
    for frame = start_frame_of_syllable: ...
            start_frame_of_syllable + NUMBER_OF_PSQM_FRAMES_PER_SYLLABE- 1
        if (frame <= stop_frame)
            h = frame_disturbance(1+ frame);
            %             if (start_frame_of_syllable== 101)
            %                 fprintf( fid, '%f\n', h);
            %             end
            result_syllable = result_syllable+ (h^ power_syllable);
        end
        count_syllable = count_syllable+ 1;
    end
    
    result_syllable = result_syllable/ count_syllable;
    result_syllable = result_syllable^ (1/power_syllable);
    
    result_time= result_time+ (time_weight (...
        1+ start_frame_of_syllable - start_frame) * ...
        result_syllable)^ power_time;
    total_time_weight_time = total_time_weight_time+ ...
        time_weight (1+ start_frame_of_syllable - start_frame)^ power_time;
    
    %     fprintf( fid, '%f\n', result_time);
end
% fclose (fid);

% fprintf( 'total_time_weight_time is %f\n', total_time_weight_time);
result_time = result_time/ total_time_weight_time;
result_time= result_time^ (1/ power_time);
% fprintf( 'result_time is %f\n\n', result_time);