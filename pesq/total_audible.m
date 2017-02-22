function total_audible_pow = total_audible (frame, ...
    pitch_pow_dens, factor)

global Nb abs_thresh_power

total_audible_pow = 0;
for band= 2: Nb
    h = pitch_pow_dens (frame+ 1,band);
    threshold = factor * abs_thresh_power (band);
    if (h > threshold)
        total_audible_pow = total_audible_pow+ h;
    end
end