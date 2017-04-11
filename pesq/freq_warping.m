function pitch_pow_dens= freq_warping( hz_spectrum, Nb, frame)

global nr_of_hz_bands_per_bark_band pow_dens_correction_factor
global Sp

hz_band = 1;
for bark_band = 1: Nb
    n = nr_of_hz_bands_per_bark_band (bark_band);
    sum = 0;
    for i = 1: n
        sum = sum+ hz_spectrum( hz_band);
        hz_band= hz_band+ 1;
    end
    sum = sum* pow_dens_correction_factor (bark_band);
    sum = sum* Sp;
    pitch_pow_dens (bark_band) = sum;
    
end