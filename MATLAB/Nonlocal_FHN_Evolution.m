%FitzHugh Nagumo
% Code and parameteres taken from: FOurier spectral methods fro fractional
% in space reaction diffusion equations
% Not fractional laplacian
% But yes nonlocal diffusion.

% Equations taken from Shima's paper:

% u_t = K*Delta*(1-D* Delta)^-1 u + sigma*(u-u^3 -v)
% v_t = (a u + b)
clear; clc; close all


%----FHN Parameters
%These parameters should lead to FHN having a limit cycle.
a = 2;
b = 0.2;
sigma = (4/3)*(1/sqrt(a));
%sigma = 0.5;

d2=2.75; b1 = -0.5; b2=0.5;
c1 =0; c2=1.5;


aux = (4/3)*(1/sqrt(a*sigma));


%--- Code Parameters
N = 1024;
L = 150;
tend = 100;
dt=0.01;
M=1;


%We want to use Neumann BC so we use the cosine transform
[kx,ky] = meshgrid((0:N-1)*pi/L); 
LL = -(kx.^2+ky.^2);
BB = -1 + exp(-(kx.^2+ky.^2)*500);

dx = L/(N+1); 
[X,Y] = meshgrid((0:N-1)*dx);

g= exp(-((X-L/2).^2+(Y-L/2).^2)/50);

%---Main Loop: Implicit method

%Initial conditions to create spiral wave
% u0 = -1*ones(N,N);
% u0(1:N/2, 1:N/2)=1;
% u0(1:N/2,N/2+1:end) = -b/a;
% 
% v0= 0.2*ones(N,N);
% v0(1:N/2, 1:N/2)=0;
% v0(1:N/2,N/2+1:end) = -b/a + (b/a)^3;

u0 = ones(N,N)*(-b/a);
v0 = ones(N,N)*( -b/a + (b/a)^2);

u = u0;
v= v0;
for n = 1:round(tend/dt)
    u0dct = dct2(u); v0dct = dct2(v);
    t(n) = n*dt; 
    for m = 1:M
        % u_t = K*Delta*(1-Delta)^-1 u + sigma*(u-u^3) -v
        % v_t = (beta u + delta)

        f1 =(1/sigma)* (u -u.^3 -v) + c1*g.*v ;
        f2 = (a*u + b) + c2*g.*u;
        
        u = idct2((u0dct+ dt*dct2(f1) +dt*b1*BB.*v0dct  )./(1- dt*LL));
        v = idct2((v0dct + dt*dct2(f2) + dt*d2*LL.*u0dct + dt*b2*BB.*u0dct)./(1-dt*LL));
    end

    U(:,n) = u(:,N/2);
end

figure(1)
surf(X,Y,u), shading interp, lighting phong, axis tight
view([0 90]) ; %set(gca,'zlim',[-1 100])
%light('color',[1 1 0],'position',[-1,2,2])
material([0.30 0.60 0.60 40.00 1.00]);
daspect([1 1 1])


[T,XX]= meshgrid(t,(0:N-1)*dx);
figure(2)
surf(XX,T,U), shading interp, lighting phong, axis tight
view([0 90]) ; %set(gca,'zlim',[-1 100])
%light('color',[1 1 0],'position',[-1,2,2])
material([0.30 0.60 0.60 40.00 1.00]);

