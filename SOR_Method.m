%% Successive Over-Relaxation Method

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
% Reshape blurred image to vector
b = blurred_noisy_image(:);


% Get A - use 'valid' to match 'same' convolution in forward problem
A = convmtx2(PSF, [m,n]);

%% Preconditioner
% Create ATA = A^T * A
ATA = A' * A;

% Decompose ATA = T + D + T^T
T = tril(ATA, -1);        % Lower triangular part (strictly lower)
D = diag(diag(ATA));      % Diagonal part as matrix
tau = 0.1;
M = D + tau * T;

%% Richardon Iteration
x0 = zeros(m*n, 1); % Initial guess
x = x0;
max_iters = 1000;
r = b - A * x0; % Initial residual
v = A' * r; % Initial gradient
d = M\v; % Initial search direction (perform d=M\v, which does Mw=v)

% refer to M in the text:
% Strategy 1.form M inverse costs a lot, which is relatively expensive
% Strategy 2.track the equation Mw=v for some w, then we have 
% w=M_inverse * v

%DO NOT FORM MATRIX INVERSES because it is expensive.

for k = 1:max_iters
    x = x + tau * d; % Update solution
    r = b - A * x; % Update residual
    v = A' * r; % Update gradient
    d = M\v; % Update search direction
    if norm(r) <= 0.01
        disp("converged") %no display, so 1000 iterations have been exhausted
        break;
        %stopping condition
    end
end

%% Results
% Reshape x to image
recovered_image = reshape(x, m, n);
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
