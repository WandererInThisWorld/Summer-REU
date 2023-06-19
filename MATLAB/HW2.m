clear; clc; close all;

[X,Y] = meshgrid(-5:0.5:5,-5:0.5:5);
U = -1*X + 1*Y;
V = 0*X + -1*Y;

figure(1)
quiver(X,Y,U./sqrt(U.^2+V.^2),V./sqrt(U.^2+V.^2))
hold on
