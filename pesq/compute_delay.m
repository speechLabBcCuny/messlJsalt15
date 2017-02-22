function [best_delay, max_correlation] = compute_delay (...
    start_sample, stop_sample, search_range, ...
    time_series1, time_series2)

n = stop_sample - start_sample+ 1;
power_of_2 = 2^ (ceil( log2( 2 * n)));

power1 = pow_of (time_series1, start_sample, stop_sample, n)* ...
    n/ power_of_2;
power2 = pow_of (time_series2, start_sample, stop_sample, n)* ...
    n/ power_of_2;
normalization = sqrt (power1 * power2);
% fprintf( 'normalization is %f\n', normalization);

if ((power1 <= 1e-6) || (power2 <= 1e-6))
    max_correlation = 0;
    best_delay= 0;
end

x1( 1: power_of_2)= 0;
x2( 1: power_of_2)= 0;
y( 1: power_of_2)= 0;

x1( 1: n)= abs( time_series1( start_sample: ...
    stop_sample));
x2( 1: n)= abs( time_series2( start_sample: ...
    stop_sample));

x1_fft= fft( x1, power_of_2)/ power_of_2;
x2_fft= fft( x2, power_of_2);
x1_fft_conj= conj( x1_fft);
y= ifft( x1_fft_conj.* x2_fft, power_of_2);

best_delay = 0;
max_correlation = 0;

% these loop can be rewritten
for i = -search_range: -1
    h = abs (y (1+ i + power_of_2)) / normalization;
    if (h > max_correlation)
        max_correlation = h;
        best_delay= i;
    end
end
for i = 0: search_range- 1
    h = abs (y (1+i)) / normalization;
    if (h > max_correlation)
        max_correlation = h;
        best_delay= i;
    end
end
best_delay= best_delay- 1;