function phiy = activecontourCV( u0, center, radius, isinside, d_it, m_it, m_name )
% ??????CV???????u0????????
% ?????double?????1?256?????????????
% center?????????radius?????????isinside ?????????????????=1????????????=0????


% ?????
ITERATIONS = 1000;%????
delta_t = 0.1;%????
%????????
lambda1 = 20;
lambda2 = 20;
nu = 0;
%?????
h = 1; h_sq = h^2;
epsilon = 1;
mu = 0.01 * 255^2;


% ?????????
phi = initsdf( size( u0 ), center, radius, isinside );

figure,
for ii = 1 : ITERATIONS;


  % ????????
  fprintf( 1, '%d\n', ii );


  % ?d_it??????
%   if( mod( ii - 1, d_it ) == 0 )
%     disp( 'Displaying Segmented Image' );
%     segim = createim( u0, phi );
%     clf; imshow( segim );
%     drawnow;
%   end;
%   
%   % ?m_it???????
%    if( mod( ii - 1, m_it ) == 0 )
%     segim = createim( u0, phi );
%     filename = strcat( m_name, sprintf( '%06d', ( ( ii - 1 )/ m_it ) + 1 ), '.png' );
%     imwrite( segim, filename );
%   end;


  %delta_t*?????
  dirac_delta_t = delta_t * diracfunction( phi, epsilon );


  % ?????????
  [ inside, outside ] = calcenergyf( u0, phi, epsilon );
  energy_term = -nu - lambda1 .* inside + lambda2 .* outside;%???


  % ????
  dx_central = ( circshift( phi, [ 0, -1 ] ) - circshift( phi, [ 0, 1 ] ) ) / 2;
  dy_central = ( circshift( phi, [ -1, 0 ] ) - circshift( phi, [ 1, 0 ] ) ) / 2;


  % div(delt_phi/|delta_phi|)
  abs_grad_phi = ( sqrt( dx_central.^2 + dy_central.^2 ) + 0.00001 );
  x = dx_central ./ abs_grad_phi;
  y = dy_central ./ abs_grad_phi;
  grad_term = ( mu / h_sq ) .* divergence( x, y );%????


  % phi(n+1)
  imagesc(energy_term), axis image off, drawnow;
  phi = phi + dirac_delta_t .* ( grad_term + energy_term );
  phiy(:,:,ii)=phi;%????????
end;




%???
%???????????????
function phi = initsdf( imsize, center, radius, isinside )
m = imsize( 1 ); n = imsize( 2 );
phi = zeros( imsize );
for i = 1 : m;
  for j = 1 : n;
     distance = sqrt( sum( ( center - [ i, j ] ).^2 ) );
     phi( i, j ) = distance - radius;
    if( isinside == 0 )
      phi( i, j ) = -phi( i, j );
    end
  end
end


%??????????????????
function y = diracfunction( x, epsilon )
y = ( 1 ./ pi ) .* ( epsilon ./ ( epsilon.^2 + x.^2 ) );


%?????????
function value = heavisidef( z, epsilon )
value = 0.5 .* ( 1 + ( 2 ./ pi ) .* atan( z ./ epsilon ) );


%???????
function [ inside, outside ] = calcenergyf( u0, phi, epsilon )
H_phi = heavisidef( phi, epsilon );
H_phi_minus = 1 - heavisidef( phi, epsilon );


c1 = sum( sum( u0 .* H_phi ) ) /  sum( sum( H_phi ) );
c2 = sum( sum( u0 .* H_phi_minus ) ) / sum( sum( H_phi_minus ) );


inside = ( u0 - c1 ).^2;
outside = ( u0 - c2 ).^2;


%??????????????????ifro?????????????rgb??????????
function newim = createim( im, phi )
newim( :, :, 1 ) = im;
newim( :, :, 3 ) = im;


tempim = im;
tempim( find( isfro( phi ) ) ) = 255;


newim( :, :, 2 ) = tempim;


newim = uint8( newim );


%?????????
function front = isfro( phi )
[ n, m ] = size( phi );
front = zeros( size( phi ) );
for i = 2 : n - 1;
  for j = 2 : m - 1;


    maxVal = max( max( phi( i:i+1, j:j+1 ) ) );
    minVal = min( min( phi( i:i+1, j:j+1 ) ) );
    front( i, j ) = ( ( maxVal > 0 ) & ( minVal < 0 ) ) | phi( i, j ) == 0;


  end
end