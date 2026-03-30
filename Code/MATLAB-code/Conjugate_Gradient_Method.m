%% Conjugate_Gradient_Method

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

%% Preallocation
max_iters = min([m,n,100]);

beta = zeros(max_iters+1,1);
gamma = zeros(max_iters,1);

w = zeros(length(g),max_iters+1);
w_hat = zeros(length(g),max_iters+1);

y = zeros(m*n,max_iters);
y_hat = zeros(m*n,max_iters);

%% Initialization
beta(1) = norm(g);
w(:,1) = g / beta(1);
y_hat(:,1) = K' * w(:,1);
gamma(1) = norm(y_hat(:,1));
y(:,1) = y_hat(:,1) / gamma(1);

k_final = 0;

%% GKB(Golub-Kahan Bidiagonalization) Iteration
for k = 2 : max_iters
    w_hat(:,k) = K * y(:,k-1) - gamma(k-1) * w(:,k-1);
    beta(k) = norm(w_hat(:,k));
    w(:,k) = w_hat(:,k) / beta(k);

    y_hat(:,k) = K' * w(:,k) - beta(k) * y(:,k-1);
    gamma(k) = norm(y_hat(:,k));
    y(:,k) = y_hat(:,k) / gamma(k);

    k_final = k;
end

%% Establish B_k
f = zeros(m*n,k_final);
image_error = zeros(max_iters,1);
residue = zeros(max_iters,1);

err = inf;
best_p = 1;

for p = 1 : k_final
    B_k = zeros(p+1,p);

    for i = 1 : p
        B_k(i,i) = gamma(i);
    end

    for j = 2 : p+1
        B_k(j,j-1) = beta(j);
    end

    beta_e = zeros(p+1,1);
    beta_e(1) = norm(g);

    f_hat = B_k \ beta_e;
    f(:,p) = y(:,1:p) * f_hat;

    err_update = norm(f(:,p) - original_image(:)) / norm(original_image(:));

    if  err_update < err
        err = err_update; 
        best_p = p;
    end

    residue(p) = norm(g - K * f(:,p));

    image_error(p) = norm(f(:,p) - original_image(:));
end

%% Plot the error between f and f_hat
figure;
plot(1:k_final,image_error)

%% Plot the true residue
figure;
plot(1:k_final,residue)

%% Results
recovered_image = reshape(f(:,best_p),m,n);

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