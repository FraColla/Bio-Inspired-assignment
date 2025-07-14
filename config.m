% -----------------------------------------------------------------------
% This script builds the structure `settings` that is shared across the
% environment (funcReset / funcStep) and the main training script.
% -----------------------------------------------------------------------
%   • State  x = Delta[V; alpha; q; theta; h]^T     (deviations from trim)
%   • Input  u = de  [elevator] [rad]               (continuous, saturated)
% -----------------------------------------------------------------------


%% SIMULATION OPTIONS

settings.trainAgent          = false;  % true = train from scratch, false = load agent
settings.runParallel         = false;  % parallel pool for RL (works with SAC/DDPG)
settings.saveResults         = true;   % save figures & MAT‑files
settings.mainDevice          = "cpu";  % "cpu" or "gpu" for DL networks
settings.ignoreTermination   = false;  % to ignore term conditions in final simulation
settings.RandomStart         = true;   % random h0 values for each episode
settings.PostProcess         = true;   % plot at the end of the training

% Timing
settings.dt                  = 0.05;   % [s] integration step / agent sample time
settings.total_max_episodes  = 800;    % max episodes during training


%% AIRCRAFT MODEL (LINEAR)
% x_dot = A x + B u     typical linear system form

[A,B,~,~] = linearizeCessna('Cessna_Model');

settings.A = A;     % State matrix
settings.B = B;     % Input matrix

% Trim point (deviations are zero by definition, keep for completeness)
settings.trim_state = zeros(5,1);
settings.trim_input = 0;      % δe_trim [rad]


%% ACTUATOR LIMITS
settings.de_max =  deg2rad(25);   % [rad] +25° nose‑up
settings.de_min = -deg2rad(25);   % [rad] −25° nose‑down


%% REWARD WEIGHTS 
% LQR‑style quadratic cost:  r = -x^T Q x  - u^T R u

settings.Qh = 0.5;                          % altitude weight
settings.Qint = 0;                       % integral term over h
settings.Qtheta = 1.0;                      % theta weight
settings.Qq = 0.5;                          % q weight
settings.R  = 0.6;                          % input weight
settings.Rdot = 1.0;                        % input variation weight

% Calm Bonus
settings.calmTol_h     = 0.5;            % [m]    |Δh| ≤ 0.5
settings.calmTol_theta = deg2rad(1.0);   % [rad]  |Δθ| ≤ 1.0°
settings.calmTime      = 0.5;            % [s]    minimum calm time
settings.calmBonus     = 0.05;           % reward for each 'calm' step

%% TERMINATION CONDITIONS

% fail
settings.h_lim    = 30;            % [m] |Δh| beyond which episode ends
settings.theta_lim = deg2rad(15);   % [rad] |Δθ| beyond which episode ends
settings.endPenalty = 300;          % penalty extra for fail

% success
settings.succTol_h     =  1.0;           % [m]   |Δh| ≤ 1 m
settings.succTol_theta = deg2rad(1);     % [rad] |Δθ| ≤ 2°
settings.succTime      = 2;              % [s]   (es. 2 seconds)
settings.successReward = 750;            % success bonus

%% SIMULATION START

settings.h0 = 10;