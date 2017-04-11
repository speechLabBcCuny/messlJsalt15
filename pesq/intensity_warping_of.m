function loudness_dens = intensity_warping_of (...
    frame, pitch_pow_dens)

global abs_thresh_power Sl Nb centre_of_band_bark
ZWICKER_POWER= 0.23;
for band = 1: Nb
    threshold = abs_thresh_power (band);
    input = pitch_pow_dens (1+ frame, band);
    
    if (centre_of_band_bark (band) < 4)
        h =  6 / (centre_of_band_bark (band) + 2);
    else
        h = 1;
    end
    
    if (h > 2)
        h = 2;
    end
    h = h^ 0.15;
    modified_zwicker_power = ZWICKER_POWER * h;
    if (input > threshold)
        loudness_dens (band) = ((threshold / 0.5)^ modified_zwicker_power)...
            * ((0.5 + 0.5 * input / threshold)^ modified_zwicker_power- 1);
    else
        loudness_dens (band) = 0;
    end
    
    loudness_dens (band) = loudness_dens (band)* Sl;
end