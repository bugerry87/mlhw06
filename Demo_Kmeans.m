



%% Cleanup
clear;

%% Parameters
filename = myinput("Path to dataset.\n    X ('circle.txt'): ", 'circle.txt');
K = myinput("Number of clusters.\n    K (2): ", 2);

%% Load Data
X = csvread(filename);

%% Plot Data
figure(1);
scatter(X(:,1), X(:,2), 2);
title("Data");
pause(0.1);

%% Demo
d = size(X,2);
M = randn(K,d);

[M, Y] = kmeans(X, M, @plot_yield);

function plot_yield(X, Y, M)
    figure(1);
    scatter(X(:,1), X(:,2), 2, Y);
    hold on;
    scatter(M(:,1), M(:,2), 300, 'k', 'x');
    title("KMeans");
    hold off;
    pause(0.1);
end