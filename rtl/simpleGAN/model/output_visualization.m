%% =========================================================
%  GAN Output Visualizer (High Contrast Version)
% =========================================================
clear; clc; close all;

% ==========================================================
%  USER INPUTS
% ==========================================================
% Input 1: RTL result
data_1 = [ ...
    -0.1211  0.5352  0.1523 ...
     0.5625 -0.1523  0.5859 ...
     0.0469  0.5703 -0.0156 ...
];

% Input 2: MATLAB result (expected)
data_2 = [ ...
    -0.1263  0.5758  0.1731 ... 
     0.6075 -0.1566  0.6390 ...
     0.0563  0.6114 -0.0221 ...
];

plot_list = {data_1, data_2};
titles    = {'Test 1: RTL Full GAN Mode 2', 'Test 2: MATLAB Full GAN Mode 2'};

% ==========================================================
%  VISUALIZATION LOGIC
% ==========================================================
img_size = 3;

figure('Name', 'GAN Result Visualization', 'Color', 'w', 'Position', [100, 100, 1000, 400]);
t = tiledlayout(1, length(plot_list), 'TileSpacing', 'loose', 'Padding', 'compact');

for i = 1:length(plot_list)
    raw_data = plot_list{i};
    if isempty(raw_data) continue; end
    
    % 1. Handle Input Format
    if isequal(size(raw_data), [3,3])
        x_fake = raw_data'; 
        x_fake = x_fake(:); 
    else
        x_fake = raw_data(:); 
    end
    
    % 2. Visualization Core
    % We use your training code's exact math here to match the look.
    % (Note: Auto-scaling makes the exact formula less critical, but consistency is good)
    img = reshape(x_fake/2 + 0.5, img_size, img_size)';
    
    % 3. Plotting
    nexttile; 
    imagesc(img);
    colormap gray; 
    axis image off;
    title(titles{i}, 'FontSize', 12, 'Interpreter', 'none');
    
end