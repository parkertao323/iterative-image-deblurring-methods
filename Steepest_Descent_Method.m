%% Steepest_Descent_Method

%% Setup
%load data
data_file = "image_deblurring_data.mat";
blurred_image = load(data_file).blurred_image_full;
blurred_noisy_image = load(data_file).blurred_noisy_image;
PSF = load(data_file).PSF;
blur_type = load(data_file).blur_type;
original_image = load(data_file).original_image;

[m,n] = size(original_image);
[m_full, n_full] = size(blurred_image);

g = blurred_noisy_image(:);

K = convmtx2(PSF, [m,n]);

%% Preconditioner
x0 = zeros (m * n, 1);
% Again, x0 is the function f
x_update = x0;

max_iters = 1000;
r = g - K * x0;
r_update = r;

%% Iteration
for i = 1:max_iters
    d = K' * r;
    w = K * d;
    tau = norm(d)^2 / norm(w)^2;
    x_update = x_update + tau * d;
    r_update = r_update - tau * w;
end

%% Results
recovered_image = reshape(x_update, m, n);
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