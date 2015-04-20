function R = add_lns(a_ln, b_ln)
    % function for computing the sum of logarithm
    % ln(a + b) = ln{exp[ln(a) - ln(b)] + 1} + ln(b)
    
    % 2^52-1 = 4503599627370495. log of that is 36.043653389117155867651465390794
    if (abs(a_ln - b_ln) >= 36.043653389117155) 
        R = max(a_ln, b_ln); 
    else
        % this branch is necessary, to avoid shifted_a_ln = a_ln - b_ln having too big value
        shifted_a_ln = a_ln - b_ln;
        shifted_sum = exp(shifted_a_ln) + 1; 
        shifted_sum_ln = log(shifted_sum); 
        unshifted_sum_ln = shifted_sum_ln + b_ln; 
        R = unshifted_sum_ln;
    end
end