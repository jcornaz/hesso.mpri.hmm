function res = dispColumnMean( title, m1, m2, m3 )

    m = zeros([1,3]);
    
    [~,m(1)] = size( m1 );
    [~,m(2)] = size( m2 );
    [~,m(3)] = size( m3 );
    
    res = mean( m );
    
    disp( strcat( title, ' : ', num2str( res ) ) );
end