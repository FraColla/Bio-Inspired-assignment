function [r, r_vec, tCalm, intH] = f_rewards(x, u, du, tCalmPrev, intHPrev, settings)
% x = [Δu; Δw; q; Δθ; Δh]
% u = δe  [rad]

h = x(5);   theta  = x(4);   q = x(3);

% ---- base reward: tracking ----------------------------------------------
r_h   = - settings.Qh  * h^2;
r_theta   = - settings.Qtheta  * theta^2;
r_q   = - settings.Qq  * q^2;

r     = r_h + r_theta + r_q;

% ---- control effort ----------------------------------------------------
r_u   = - settings.R    * u^2;  % control input value
r_du  = - settings.Rdot * du^2; % diff between two consecutive inputs

r = r + r_u + r_du;

% ---- calm bonus --------------------------------------------------------
isCalm = abs(h) < settings.calmTol_h  && ...
         abs(theta)  < settings.calmTol_theta;
if isCalm
    tCalm = tCalmPrev + settings.dt;
else
    tCalm = 0;
end
if tCalm >= settings.calmTime
    r = r + settings.calmBonus;
end

% integral term on altitude
intH = intHPrev + abs(h)*settings.dt;          % accumula |Δh|·dt
r_int = - settings.Qint  * (intH)^2;             % quadratico

r = r + r_int;

% vec reward output
r_vec = [r_h; r_theta; r_q; r_u; r_du; r_int];
end
