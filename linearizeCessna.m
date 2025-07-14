function [A,B,C,D] = linearizeCessna(modelName)
%LINEARIZECESSNA  Linearize a Simulink aircraft model and return its state-space matrices.
%
%   [A,B,C,D] = LINEARIZECESSNA() linearizes the default model
%   'Cessna_Model' using the Analysis I/O markers already defined inside
%   the Simulink file and returns the state-space matrices A, B, C, D
%   evaluated at the model’s initial operating point.
%
%   [A,B,C,D] = LINEARIZECESSNA(modelName) performs the same operation on
%   the model specified by the string modelName.
%
%   Current assumptions (matching your setup):
%       • Input  : elevator (1 input)
%       • Outputs: V, Ze, q, theta (4 outputs)
%       • State  : V, alpha, q, theta, Ze (5 states)
%
%   Example
%   -------
%       [A,B,C,D] = linearizeCessna;   % use default model
%       sys = ss(A,B,C,D);
%       step(sys)                      % plot step response
%
%   See also: LINEARIZE, GETLINIO, OPERPOINT, SSDATA

    % Use default model if none specified
    if nargin < 1 || isempty(modelName)
        modelName = 'Cessna_Model';
    end

    % Ensure the model is loaded
    load_system(modelName);

    % Retrieve Analysis I/O points from the model
    io = getlinio(modelName);

    % Operating point: model initial conditions
    op = operpoint(modelName);

    % Perform exact linearization
    sys = linearize(modelName, io, op);

    % Extract state-space matrices
    [A,B,C,D] = ssdata(sys);
end



