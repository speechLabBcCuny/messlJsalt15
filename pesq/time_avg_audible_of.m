function avg_pitch_pow_dens= time_avg_audible_of(number_of_frames, ...
    silent, pitch_pow_dens, total_number_of_frames)

global Nb abs_thresh_power

for band = 1: Nb
    result = 0;
    for frame = 1: number_of_frames
        if (~silent (frame))
            h = pitch_pow_dens (frame, band);
            if (h > 100 * abs_thresh_power (band))
                result = result + h;
            end
        end
        
        avg_pitch_pow_dens (band) = result/ total_number_of_frames;
    end
end