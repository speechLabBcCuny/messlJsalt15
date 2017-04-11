function mod_disturbance_dens= multiply_with_asymmetry_factor (...
    disturbance_dens, frame, pitch_pow_dens_ref, pitch_pow_dens_deg)

global Nb
for i = 1: Nb
    ratio = (pitch_pow_dens_deg(1+ frame, i) + 50)...
        / (pitch_pow_dens_ref (1+ frame, i) + 50);
    h = ratio^ 1.2;
    if (h > 12)
        h = 12;
    elseif (h < 3)
        h = 0.0;
    end
    mod_disturbance_dens (i) = disturbance_dens (i) * h;
end