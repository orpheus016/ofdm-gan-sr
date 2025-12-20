%% =========================================================
%  3x3 O+ GAN Training  (w/o Toolbox)
% =========================================================
clear; clc;
close all;

% Set rand seed
 rng(0);

% Parameters パラメータ設定
img_size = 3;           % Image size
latent_dim = 2;        % Latent variables

D_hidden_L = 3;      % Number of hidden layers (Discriminator)
G_hidden_L = 3;      % Number of hidden layers (Generator)

num_epochs = 30000;     % Epochs 
%num_epochs = 2;     % Epochs 
eta_D = 0.001;
eta_G = 0.001;
save_path = "trained_simple_gan.mat";
DGL = 2;                 % DとGの学習比D/G = 2 -> D:G = 2:1

% ======== Train data, 学習データ (3x3 O+)  ========
circle = [1 1 1; ...
              1 -1 1; ...
              1 1 1]';

cross = [-1 1 -1; ...
               1 1 1; ...
               -1 1 -1]';

%data = [circle(:)'; cross(:)'; triangle(:)'];  % 3x25
data = [circle(:)'; cross(:)'];  % 3x25
num_data = size(data,1);

% ===  Loss data 損失記録変数 ===
loss_D = zeros(num_epochs,1);
loss_G = zeros(num_epochs,1);

% ======== Initialization (weights) ネットワーク初期化 ========
Wg2 = 0.1*randn(G_hidden_L, latent_dim);
bg2 = zeros(G_hidden_L,1);
Wg3 = 0.1*randn(img_size*img_size,G_hidden_L);
bg3 = zeros(img_size*img_size,1);

Wd2 = 0.1*randn(D_hidden_L,img_size*img_size);
bd2 = zeros(D_hidden_L,1);
Wd3 = 0.1*randn(1,D_hidden_L);
bd3 = 0;

fprintf("Initial weight (Discriminator)\n");
fprintf('W^D2\n'); disp(Wd2);
fprintf('b^D2\n'); disp(bd2);
fprintf('W^D3\n'); disp(Wd3);
fprintf('b^D3\n'); disp(bd3);

fprintf("Initial weight (Generator)\n");
fprintf('W^G2\n'); disp(Wg2);
fprintf('b^G2\n'); disp(bg2);
fprintf('W^G3\n'); disp(Wg3);
fprintf('b^G3\n'); disp(bg3);

% ======== Activation function 活性化関数 ========
sigmoid = @(x) 1./(1+exp(-x));
tanh_f = @(x) tanh(x);

% ======== Loop of learning 学習ループ ========
for epoch = 1:num_epochs
    % ==== Update descriminator (D) 識別器更新 ====
    % ==== D更新 ====
    idx = randi(num_data);
    x_real = data(idx,:)';

    % Generate fake Image フェイク画像の生成
    ng = randn(latent_dim,1);
    ag2 = tanh_f(Wg2*ng + bg2);
    x_fake = tanh_f(Wg3*ag2 + bg3);

    % Discriminate real image 正解画像の判別
    ad2_real = tanh_f(Wd2*x_real + bd2);
    y_real = sigmoid(Wd3*ad2_real + bd3);
    ad3_real = y_real;

    % Discriminate fake image フェイク画像の判別
    ad2_fake = tanh_f(Wd2*x_fake + bd2);
    y_fake = sigmoid(Wd3*ad2_fake + bd3);
    ad3_fake = y_fake;

    % === Calculate loss function (Discriminator) 損失計算 ===
    loss_D(epoch) = - (log(y_real + 1e-8) + log(1 - y_fake + 1e-8));
                                % Loss of real image + Loss of fake image 

    fignum=10;
    if epoch == 1 || epoch == num_epochs
        fprintf('Discriminator (Real Image) %d epoch\n',epoch);
        fprintf('x_real\n'); disp(x_real);        
        fprintf('ad2_real\n'); disp(ad2_real);
        fprintf('ad3_real\n'); disp(ad3_real);
        fprintf('BCE\n'); disp(- (log(y_real + 1e-8)));

        fprintf('Generator (Fake Image) %d epoch\n',epoch);
        fprintf('ng\n'); disp(ng);
        fprintf('ag2\n'); disp(ag2);
        fprintf('x_fake\n');disp(x_fake);

        fprintf('Discriminator (Fake Image) %d epoch\n',epoch);
        fprintf('ad2_real\n'); disp(ad2_real);
        fprintf('ad3_real\n'); disp(ad3_real);
        fprintf('BCE\n'); disp(- log(1 - y_fake + 1e-8));

         img = reshape(x_real/2+0.5, img_size, img_size)';
        figure(fignum); imagesc(img);
        colormap gray; axis image off;
        title(sprintf("X Real Image (epoch %d)", epoch));
        drawnow;
         fignum=fignum+1;
         img = reshape(x_fake/2*0.5, img_size, img_size)';
        figure(fignum); imagesc(img);
        colormap gray; axis image off;
        title(sprintf("X Fake image (epoch %d)", epoch));
        drawnow;
         fignum=fignum+1;
    end


    % 勾配（D）
    % Loss -> a^D3 units
    dLdy_real = -(1 - y_real);
    dLdad3_real = dLdy_real ;
    dLdy_fake = y_fake;
    dLdad3_fake = dLdy_fake;

    % Loss -> a^D3 -> z^D3 (delta^D3)
    dad3dzd3_real = ad3_real .* (1-ad3_real);             % Activation function : sigmoid, f'(x) = f(x)(1-f(x))
    deltad3_real = dLdad3_real .* dad3dzd3_real;

    % Loss -> delta^D3 -> w^D3 or -> b^D3
    dWd3_real = deltad3_real * ad2_real';
    dBd3_real = deltad3_real;

    % Loss -> delta^D3 -> a^D2 -> z^D2 (delta^D2)
    dLdad2_real = Wd3' * deltad3_real;
    dad3dzd3_real = (1 - ad2_real.^2);                      % Activation function : tanh
    deltad2_real = dLdad2_real .* dad3dzd3_real ;

    % Loss -> delta^D2 -> w^D2 or -> b^D2
    dWd2_real = deltad2_real * x_real';
    dBd2_real = deltad2_real;

    % Fake image
    deltad3_fake = dLdy_fake .* y_fake .* (1-y_fake);
    dWd3_fake = deltad3_fake * ad2_fake';
    dBd3_fake = deltad3_fake;

    deltad2_fake = (Wd3' * deltad3_fake) .* (1 - ad2_fake.^2);
    dWd2_fake = deltad2_fake * x_fake';
    dBd2_fake = deltad2_fake;

    Wd3 = Wd3 - eta_D * (dWd3_real + dWd3_fake);
    bd3 = bd3 - eta_D * sum(dBd3_real + dBd3_fake,2);
    Wd2 = Wd2 - eta_D * (dWd2_real + dWd2_fake);
    bd2 = bd2 - eta_D * sum(dBd2_real + dBd2_fake,2);

    % ==== 生成器更新 ====
    % ==== G更新 ====
    ng = randn(latent_dim,1);
    ag2 = tanh_f(Wg2*ng + bg2);
    x_fake = tanh_f(Wg3*ag2 + bg3);

    ad2_fake = tanh_f(Wd2*x_fake + bd2);
    y_fake = sigmoid(Wd3*ad2_fake + bd3);

    % === 生成器損失 ===
    loss_G(epoch) = - log(y_fake + 1e-8);

    dLdy_fake = -(1 - y_fake);
    deltad3_fake = dLdy_fake .* y_fake .* (1-y_fake);                      % Activation : sigmoid
    deltad2_fake = (Wd3' * deltad3_fake) .* (1 - ad2_fake.^2);      % Activation : tanh
    deltag3 = (Wd2' * deltad2_fake) .* (1 - x_fake.^2);                   % Activation : tanh

    dWg3 = deltag3 * ag2';
    dBg3 = deltag3;

    deltag2 = (Wg3' * deltag3) .* (1 - ag2.^2);                               % Activation : tanh

    dWg2 = deltag2 * ng';
    dBg2 = deltag2;

    if rem(epoch, DGL)== 0
        Wg3 = Wg3 - eta_G * dWg3;
        bg3  = bg3  - eta_G * sum(dBg3,2);
        Wg2 = Wg2 - eta_G * dWg2;
        bg2  = bg2  - eta_G * sum(dBg2,2);
    end

    % === 進捗表示 ===
    if mod(epoch, 1000) == 0
        fprintf("Epoch %d / %d  |  L_D=%.3f  L_G=%.3f\n", ...
                epoch, num_epochs, loss_D(epoch), loss_G(epoch));
        img = reshape(x_fake/2+0.5, img_size, img_size);
        imagesc(img);
        colormap gray; axis image off;
        title(sprintf("Generated sample (epoch %d)", epoch));
        drawnow;
    end
end

% ======== Store parameters パラメータ保存 ========
save(save_path, 'Wg2','bg2','Wg3','bg3','Wd2','bd2','Wd3','bd3','loss_D','loss_G');
fprintf("Finish train. Store parameters '%s' \n", save_path);

% ======== Generate images サンプル生成 ========
figure;
for i = 1:9
    ng = randn(latent_dim,1);
    ag2 = tanh_f(Wg2*ng + bg2);
    x_fake = tanh_f(Wg3*ag2 + bg3);
    subplot(3,3,i);
    imagesc(reshape(x_fake/2+0.5,img_size,img_size));
    colormap gray; axis image off;
end
sgtitle("Generated Samples (Trained GAN)");

% Store images 画像保存
exportgraphics(gcf,"generated_samples.png");
fprintf("Store images : 'generated_samples.png' \n");

% === Display loss curve 損失表示 ===
figure;
plot(loss_D,'r','DisplayName','Discriminator Loss'); hold on;
plot(loss_G,'b','DisplayName','Generator Loss');
legend; xlabel('Epoch'); ylabel('Loss');
title('GAN Training Loss');
grid on;
exportgraphics(gcf,"loss_curve.png");
fprintf("Sore loss curve :  'loss_curve.png' \n");

%% =========================================================
%  WEIGHT EXTRACTION FOR VERILOG (Signed Decimal)
% =========================================================
fprintf('\n--- Starting Weight Extraction (Signed Decimal) ---\n');

% 1. Define Conversion Functions (Anonymous functions)
% ---------------------------------------------------------
% Mengembalikan nilai integer murni (int8/int16), bukan string Hex.
% MATLAB otomatis melakukan saturasi jika nilai melebihi range int.

% Convert Float to Q1.7 Integer (8-bit signed)
% Range: -128 to +127
float2Q1_7 = @(x) int8(round(x * 128));

% Convert Float to Q8.8 Integer (16-bit signed)
% Range: -32768 to +32767
float2Q8_8 = @(x) int16(round(x * 256));

% Helper handle
write_weights = @(filename, data, converter) write_to_file(filename, data, converter);

% 2. EXECUTE Extraction
% ---------------------------------------------------------

% --- Extract GENERATOR Parameters ---
% Layer 1 (2 -> 3)
write_weights('gen_w1.txt', Wg2, float2Q1_7); 
write_weights('gen_b1.txt', bg2, float2Q8_8); 

% Layer 2 (3 -> 9)
write_weights('gen_w2.txt', Wg3, float2Q1_7);
write_weights('gen_b2.txt', bg3, float2Q8_8);

% --- Extract DISCRIMINATOR Parameters ---
% Layer 1 (9 -> 3)
write_weights('disc_w1.txt', Wd2, float2Q1_7);
write_weights('disc_b1.txt', bd2, float2Q8_8);

% Layer 2 (3 -> 1)
write_weights('disc_w2.txt', Wd3, float2Q1_7);
write_weights('disc_b2.txt', bd3, float2Q8_8);

fprintf('--- Extraction Complete. Files contain Signed Decimal values. ---\n');

% 3. Local Function Definition (MUST BE AT THE END)
% ---------------------------------------------------------
function write_to_file(filename, matrix, func_handle)
    fid = fopen(filename, 'w');
    % Transpose matrix agar urutan data sesuai pembacaan sequential (Row-Major)
    flat_data = matrix'; 
    flat_data = flat_data(:);
    
    for k = 1:length(flat_data)
        val = func_handle(flat_data(k));
        % Gunakan %d untuk mencetak Signed Decimal (contoh: -45, 120)
        fprintf(fid, '%d\n', val);
    end
    fclose(fid);
    fprintf('Saved: %s (%d elements)\n', filename, length(flat_data));
end