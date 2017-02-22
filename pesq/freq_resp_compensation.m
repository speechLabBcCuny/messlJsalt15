function mod_pitch_pow_dens_ref= freq_resp_compensation (number_of_frames, ...
    pitch_pow_dens_ref, avg_pitch_pow_dens_ref, ...
    avg_pitch_pow_dens_deg, constant)

global Nb

for band = 1: Nb
    x = (avg_pitch_pow_dens_deg (band) + constant) / ...
        (avg_pitch_pow_dens_ref (band) + constant);
    if (x > 100.0)
        x = 100.0;
    elseif (x < 0.01)
        x = 0.01;
    end
    
    for frame = 1: number_of_frames
        mod_pitch_pow_dens_ref(frame, band) = ...
            pitch_pow_dens_ref(frame, band) * x;
    end
end