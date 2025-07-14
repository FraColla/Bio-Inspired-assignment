function [InitialObservation, LoggedSignals] = funcReset(settings)

x0 = settings.trim_state;

% Random value of h0 within +- h_span from trim condition
h_span = 10;

if settings.RandomStart
    x0(5) = (rand*2-1) * h_span;
else
    x0(5) = settings.h0;
end

LoggedSignals.State            = x0;
LoggedSignals.h0               = x0(5);
LoggedSignals.cumulativeState  = x0;
LoggedSignals.cumulativeReward = [];
LoggedSignals.cumulativeInput  = [];
LoggedSignals.t                = 0;
LoggedSignals.tInside          = 0;
LoggedSignals.prevU            = 0;
LoggedSignals.tCalm            = 0;
LoggedSignals.intH             = 0;

InitialObservation = x0;
end
