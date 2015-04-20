function res=dispAudioLengthMean( title, rate, data1, data2, data3 )
    m = zeros([1,3]);
    
    [m(1),~] = size( data1 );
    [m(2),~] = size( data2 );
    [m(3),~] = size( data3 );
    
    m = 1000 * m / rate;
    
    res = mean( m );
    
    disp( strcat( title, ' : ', num2str( res ), 'ms' ) );
end