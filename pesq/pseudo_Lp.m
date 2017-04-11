function result= pseudo_Lp (x, p)

global Nb width_of_band_bark
totalWeight = 0;
result = 0;
for band = 2: Nb
    h = abs (x (band));
    w = width_of_band_bark (band);
    prod = h * w;
    
    result = result+ prod^ p;
    totalWeight = totalWeight+ w;
end
result = (result/ totalWeight)^ (1/p);
result = result* totalWeight;