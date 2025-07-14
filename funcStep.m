function [NextObs,Reward,IsDone,LoggedSignals] = ...
         funcStep(Action,LoggedSignals,settings)

x  = LoggedSignals.State;     % 5×1
dt = settings.dt;

% continuous action: ±25°
u = max(min(Action, settings.de_max), settings.de_min);   % scalare δe

LoggedSignals.cumulativeInput = [LoggedSignals.cumulativeInput, u];


% linear dynamics
xdot = settings.A*x + settings.B*u;
xnew = x + dt*xdot;
tnew = LoggedSignals.t + dt;

% Success conditions
in_h     = abs(xnew(5))  < settings.succTol_h;
in_theta = abs(xnew(4))  < settings.succTol_theta;
condOK   = in_h && in_theta;

% Time inside success condition updating
if condOK
    LoggedSignals.tInside = LoggedSignals.tInside + dt;
else
    LoggedSignals.tInside = 0;
end

% Storing previous input command
if ~isfield(LoggedSignals,'prevU'); LoggedSignals.prevU = 0; end
du = abs(u - LoggedSignals.prevU);

% reward (tracking + sforzo attuatore)
[Reward, r_vec, LoggedSignals.tCalm, LoggedSignals.intH] = ...
        f_rewards(xnew, u, du, LoggedSignals.tCalm, LoggedSignals.intH, settings);

LoggedSignals.cumulativeReward = [LoggedSignals.cumulativeReward, r_vec];

LoggedSignals.prevU = u; % field updating

% Episode termination -----------------------------------------------------
IsFail = abs(xnew(5)) > settings.h_lim || abs(xnew(4)) > settings.theta_lim;
IsSucc = LoggedSignals.tInside >= settings.succTime;

if settings.ignoreTermination
    IsDone = false;
else
    IsDone = IsFail || IsSucc;
end

% Fail penalty / Success bonus
if IsFail
    Reward = Reward - settings.endPenalty;
elseif IsSucc
    Reward = Reward + settings.successReward;
end

% Update log & output ----------------------------------------------------
LoggedSignals.State = xnew;
LoggedSignals.t     = tnew;
LoggedSignals.cumulativeState = [LoggedSignals.cumulativeState, xnew];

NextObs = xnew;
end
