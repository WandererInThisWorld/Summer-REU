clear; clc; close all;
% Lotka-Volterra equations (Predator-Prey Model)

% xdot = a x - b xy  rabbits
% ydot = c xy - d y  foxes


a = 2; b= 3;
c = 1; d = 1;

% Finding Equilibria
fun = @(V)LotkaVolterrafun(V,a,b,c,d); 
V0 =[1,1];

Veq = fsolve(fun,V0);


% Quiver Plot / Vector Plot

[X,Y] = meshgrid(0:0.1:3,0:0.1:3);
U= a*X - b*X.*Y;
V = c*X.*Y - d*Y;

figure(1)
quiver(X,Y,U./sqrt(U.^2+V.^2),V./sqrt(U.^2+V.^2))
hold on
plot(Veq(1),Veq(2),'ro','LineWidth',3)
plot(0,0,'ro','LineWidth',3)

hold off
axis([0 2 0 1])

% Nulclines
% x = 0;
% x = d/c;
% y =0
% y = a/b;

% Solving equation

[t,y] = ode45(@(t,V) LotkaVolterrafun2(t,V,a,b,c,d),...
    [0 20],[1; 0.6]);

figure(3)
plot(t,y(:,1),'LineWidth',1)
hold on
plot(t,y(:,2),'LineWidth',1)
hold off
legend('rabbit','foxes')

figure(4)
plot(y(:,1),y(:,2),'LineWidth',2)
xlabel('rabbits')
ylabel('foxes')





