%% Cimmino_Diagonal_Algorithm

%% Setup
%load data
data_file = "image_deblurring_data.mat";
blurred_image = load(data_file).blurred_image_full;
blurred_noisy_image = load(data_file).blurred_noisy_image;
PSF = load(data_file).PSF;
blur_type = load(data_file).blur_type;
original_image = load(data_file).original_image;

[M,N] = size(original_image);

[m_full, n_full] = size(blurred_image);

% Reshape blurred image to vector
% g the datas
g = blurred_noisy_image(:);


%% Implementation
x_update = zeros(M,N);

row_norms = conv2(ones(M,N), PSF.^2, 'full');

% step size tau given by 2 / number of pixels of an image
tau = 2 / (m_full * n_full);

for p = 1:1000
    
    % forward operation: K*x0
    Kx0 = conv2(x_update, PSF, 'full');
    
    % residual
    r = g - Kx0;
    
    % diagonal weighting: v = M_inv * r
    v = r ./ row_norms;
    
    % update direction: d = K.' * v
    d = conv2(v, rot90(PSF,2), 'valid');
    
    % update iteratively
    x_update = x_update + tau * d;
end

%% Results
recovered_image = reshape(x_update,M,N);

% Display results
figure;
subplot(1,3,1);
imshow(blurred_noisy_image, []);
title("Blurred Noisy Image");
subplot(1,3,2);
imshow(recovered_image, []);
title("Recovered Image");
subplot(1,3,3);
imshow(original_image, []);
title("Original Image");