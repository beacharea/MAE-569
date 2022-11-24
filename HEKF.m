% Extended Kalman Filter on 2D vortex interaction
clear; close all; clc;

%% generate data
% vortex parameter
gam1 = 2; gam2 = 1;

% IC
x0 = [-1 0 2 0]';   %[x1 y1 x2 y2]'
tspan = 0:118;

% ode45
ODEopt = odeset('RelTol',1e-6,'AbsTol',1e-6);
[t_true,x_true] = ode45(@(t,x) vortdyn(t,x,gam1,gam2),tspan,x0,ODEopt);
t_true = t_true'; x_true = x_true';

% select measurement data for t=10:10:110
measure_ind = (10:10:110)+1;
t_measure = t_true(measure_ind);
y = x_true(:,measure_ind);

% add zero mean white Gaussian noise
mu = 0; variance = 0.01;
sigma = sqrt(variance);
for ii = 1:length(measure_ind)
    v_noise = normrnd(mu,sigma,[4,1]);
    y(:,ii) = y(:,ii) + v_noise;
end

% plot result
MS = 'MarkerSize'; MFC = 'MarkerFaceColor'; MEC = 'MarkerEdgeColor';
figure
plot(t_true,x_true(1,:),'b-',t_true,x_true(2,:),'r-', ...
    t_true,x_true(3,:),'k-',t_true,x_true(4,:),'g-'), hold on
plot(t_measure,y(1,:),'s',MS,6,MFC,'b',MEC,'k')
plot(t_measure,y(2,:),'rs',MS,6,MFC,'r',MEC,'k')
plot(t_measure,y(3,:),'ks',MS,6,MFC,'k',MEC,'k')
plot(t_measure,y(4,:),'gs',MS,6,MFC,'g',MEC,'k')
title('(a): numerical simulation of true state components and noisy measurement')
xlabel('t'), ylabel('state component')
ylim([-2.5,2.5]), xlim([0,120])
legend('true x_1','true y_1','true x_2','true y_2','measured x_1','measured y_1','measured x_2','measured y_2','Location','bestoutside')

ani_flag = false;
if ani_flag
    figure
    for ii = 1:length(t_true)
          plot(x_true(1,1:ii),x_true(2,1:ii),'ro',x_true(3,1:ii),x_true(4,1:ii),'bo',MS,1)
          axis([-3 3 -3 3])
          title('(a): dynamic vortex interaction')
          xlabel('x'), ylabel('y')
          legend('true vortex_1','true vortex_2')
          drawnow
          pause(0.1)
    end
end

%% implement hybrid extended Kalman filter
syms x_1 y_1 x_2 y_2 gamma_1 gamma_2
fA = [
    -gamma_2*(y_1-y_2)/(2*pi*((x_1-x_2)^2 + (y_1-y_2)^2));
    gamma_2*(x_1-x_2)/(2*pi*((x_1-x_2)^2 + (y_1-y_2)^2));
    -gamma_1*(y_2-y_1)/(2*pi*((x_1-x_2)^2 + (y_1-y_2)^2));
    gamma_1*(x_2-x_1)/(2*pi*((x_1-x_2)^2 + (y_1-y_2)^2));
    0;
    0;
    ];

J = jacobian(fA,[x_1, y_1, x_2, y_2, gamma_1, gamma_2]);
pretty(J)
% equivalently
%J = [diff(fA,x_1), diff(fA,y_1), diff(fA,x_2), diff(fA,y_2),...diff(fA,gamma_1), diff(fA,gamma_2)];
Jlatex = latex(J);


%% (c)
% initialize the filter
xA_a_0 = randn(6,1);%[-1.25, 0.1, 1.75, -0.2, 3, 0.5]';
PA_a_0 = eye(6);

% model
%　w ~ N(0,Q)
%  v_k ~ N(0,R)

% state space representation
H = [eye(4),zeros(4,2)];
R = 0.01*eye(4);
Q = 2.5e-4*eye(6);
L = eye(6);
M = eye(4);

%% Kalman Filter
dt = t_measure(2) - t_measure(1);
% initialization
xA_f_vec = zeros(6,length(t_measure));
xA_a_vec = zeros(6,length(t_measure));

PA_a_km1 = PA_a_0;
xA_a_km1 = xA_a_0;
Z = [];

for k = 1:length(t_measure)
    % (a)
    p_entries = P2entries(PA_a_km1);
    zIC = [xA_a_km1; p_entries];
    tspanode = [t_measure(k)-dt, t_measure(k)-dt/2, t_measure(k)];
    [~,z_f_k] = ode45(@(t,z) odefun(t,z,L,Q,J),tspanode,zIC,ODEopt);
    z_f_k = z_f_k(end,:)';
    xA_f_k = z_f_k(1:6,1);
    xA_f_vec(:,k) = xA_f_k;
    PA_f_k = entries2P(z_f_k(7:27,1));
    
    % (b)
    K_k = PA_f_k*H.'/(H*PA_f_k*H.' + M*R*M.'); % Kalman gain
    xA_a_k = xA_f_k + K_k*(y(:,k) - H*xA_f_k); % x(t|t) <- x(t|t-1)
    xA_a_vec(:,k) = xA_a_k;
    PA_a_k = (eye(6) - K_k*H)*PA_f_k*(eye(6) - K_k*H).' + K_k*M*R*M.'*K_k.';
    
    % time update
    xA_a_km1 = xA_a_k;
    PA_a_km1 = PA_a_k;
end
    
% plot result
figure
plot(t_true,x_true(1,:),'b-',t_true,x_true(2,:),'r-', ...
    t_true,x_true(3,:),'k-',t_true,x_true(4,:),'g-'), hold on
plot(t_measure,y(1,:),'bs',MS,4,MFC,'b',MEC,'k')
plot(t_measure,y(2,:),'rs',MS,4,MFC,'r',MEC,'k')
plot(t_measure,y(3,:),'ks',MS,4,MFC,'k',MEC,'k')
plot(t_measure,y(4,:),'gs',MS,4,MFC,'g',MEC,'k')
plot(t_measure,xA_a_vec(1,:),'bd',MS,4,MFC,'b',MEC,'k')
plot(t_measure,xA_a_vec(2,:),'rd',MS,4,MFC,'r',MEC,'k')
plot(t_measure,xA_a_vec(3,:),'kd',MS,4,MFC,'k',MEC,'k')
plot(t_measure,xA_a_vec(4,:),'gd',MS,4,MFC,'g',MEC,'k')
plot(t_measure,xA_f_vec(1,:),'bo',MS,4,MFC,'b',MEC,'k')
plot(t_measure,xA_f_vec(2,:),'ro',MS,4,MFC,'r',MEC,'k')
plot(t_measure,xA_f_vec(3,:),'ko',MS,4,MFC,'k',MEC,'k')
plot(t_measure,xA_f_vec(4,:),'go',MS,4,MFC,'g',MEC,'k'), hold off
title('(c): Result of Hybrid Extended Kalman Filter')
xlabel('t'), ylabel('state component')
ylim([-2.5,2.5]), xlim([0,120])
legend('true x_1','true y_1','true x_2','true y_2',...
    'measured x_1','measured y_1','measured x_2','measured y_2',...
    'analysis x_1','analysis y_1','analysis x_2','analysis y_2',...
    'forcast x_1','forcast y_1','forcast x_2','forcast y_2',...
    'Location','bestoutside')

%% (d)
figure
plot(t_measure,xA_a_vec(5,:),'ro-',t_measure,xA_a_vec(6,:),'bo-'),hold on
yline(2,'r:'), yline(1,'b:'), hold off
xlabel('t'), ylabel('vorticity'),title('(d) Vorticity Estimation')
legend('estimated \gamma_1','estimated \gamma_2','true \gamma_1','true \gamma_2')

%% functions
function dxdt = vortdyn(t,x,gam1,gam2)
d = (x(1)-x(3))^2 + (x(2)-x(4))^2;
dxdt = [
    -gam2*(x(2)-x(4))/(2*pi*d);
    gam2*(x(1)-x(3))/(2*pi*d);
    -gam1*(x(4)-x(2))/(2*pi*d);
    gam1*(x(3)-x(1))/(2*pi*d)
    ];
end

function p_entries = P2entries(P)
lower_ind = tril(ones(6));
p_entries = P(lower_ind == 1);
end

function P = entries2P(p_entries)
L = tril(ones(6));
L(L==1) = p_entries;
tmp = L + L.';
P = tmp - diag(diag(tmp)/2);
end

function A = evalJ(J,xA)
x_1 = xA(1,1); y_1 = xA(2,1);
x_2 = xA(3,1); y_2 = xA(4,1);
gamma_1 = xA(5,1); gamma_2 = xA(6,1);
A = double(subs(J));
end

function dxAdt = compute_dxAdt(t,xA)
x = xA(1:4,1); gam1 = xA(5,1); gam2 = xA(6,1);
dxdt = vortdyn(t,x,gam1,gam2);
dxAdt = [dxdt; 0; 0];
end

function dzdt = odefun(t,z,L,Q,J)
% z = 27x1 vector
%   z(1:6,1) = [x_1,y_1,x_2,y_2,gamma_1,gamma_2]'
%   z(7:27,1) = p_entries

% extract xA_a_km1 and PA_a_km1
xA_hat = z(1:6,1);
PA = entries2P(z(7:27,1));

% compute dxA/dt
dxAdt = compute_dxAdt(t,xA_hat);

% compute dPA/dt
A = evalJ(J,xA_hat);
dPAdt = A*PA + PA*A.' + L*Q*L.';

% construct dzdt
p_entries = P2entries(dPAdt);
dzdt = [dxAdt; p_entries];
end

