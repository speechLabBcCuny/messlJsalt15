function hz_spectrum= short_term_fft (Nf, data, Whanning, start_sample)

x1= data( start_sample: start_sample+ Nf-1).* Whanning;
x1_fft= fft( x1);
hz_spectrum= abs( x1_fft( 1: Nf/ 2)).^ 2;
hz_spectrum( 1)= 0;