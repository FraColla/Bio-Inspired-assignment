%% -- Main code for Cessna 620 control -- %%

%% Initialize code
clc
close all
clear all

% Fix seed for reproducibility
rng(42)

% Load configurations file
config;

%% Prepare computing setup
nCores = feature('numcores');
p = gcp('nocreate');
if isempty(p) && settings.runParallel == true
    % There is no parallel pool
    pool = parpool(nCores);
    disp(['Parallel pool running with cores: ', num2str(nCores)])
else
    if settings.runParallel ==  true
        % There is a parallel pool of <p.NumWorkers> workers
        disp(['Parallel started with cores: ', num2str(nCores)])
    else
        disp('Running single core')
    end
end

%% Create Observation and Action spaces

ObservationInfo = rlNumericSpec([5 1]); % Tells MATLAB my state is of 5 elements
ObservationInfo.Name = 'Cessna 620 state vector';
ObservationInfo.Description = 'V, alpha, q, theta, h';

ActionInfo = rlNumericSpec([1 1], ...               % Tells MATLAB which actions my vehicle can take at every step
                'LowerLimit', settings.de_min, ...
                'UpperLimit', settings.de_max);
ActionInfo.Name = 'Cessna 620 guidance';
ActionInfo.Description = 'Possible actions at each timestep';

%% Create environment using observation and action functions

% Create handles for reset and step functions
ResetHandle = @() funcReset(settings); 
StepHandle = @(Action,LoggedSignals) funcStep(Action,LoggedSignals,settings); 
env = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandle,ResetHandle); % Creates environment using the functions filenames

%% Create Neural Network
% CHECK INPUT AND OUTPUT LAYERS DIMENSIONS
% Actor Net creation
actorNet = [
    featureInputLayer(ObservationInfo.Dimension(1),'Name','state')
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(1,'Name','fc_out')
    tanhLayer('Name','tanh')
    scalingLayer('Scale',settings.de_max,'Name','de')
];

actorNetwork = dlnetwork(actorNet); % Network Initialization

% Critic Net creation
statePath  = featureInputLayer(ObservationInfo.Dimension(1), ...
                               'Name','state');
actionPath = featureInputLayer(ActionInfo.Dimension(1), ...
                               'Name','action');
commonPath = [
    concatenationLayer(1,2,'Name','concat')
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(1,'Name','Q')
];

lg = layerGraph(statePath);
lg = addLayers(lg,actionPath);
lg = addLayers(lg,commonPath);
lg = connectLayers(lg,'state','concat/in1');
lg = connectLayers(lg,'action','concat/in2');

criticNetwork = dlnetwork(lg); % Network Initialization

% Display network properties
disp('Actor Net:')
summary(actorNetwork);
disp('Critic Net:')
summary(criticNetwork);

% % Plot network structures
% actorLG  = layerGraph(actorNetwork);
% criticLG = lg;
% figure('Position',[100 100 1200 400])
% subplot(1,2,1); plot(actorLG);  title('Actor')
% subplot(1,2,2); plot(criticLG); title('Critic')

%% Create the agent based on the AC neural network

% Create the critic approximator with environment, available actions and NN
actor = rlContinuousDeterministicActor(actorNetwork,...
        ObservationInfo,ActionInfo,...
        'ObservationInputNames','state',...
        'UseDevice',settings.mainDevice);

% Create the critic approximator with environment, available actions and NN
critic = rlQValueFunction(criticNetwork, ...
        ObservationInfo, ActionInfo, ...
        'ObservationInputNames','state', ...
        'ActionInputNames','action', ...
        'UseDevice',settings.mainDevice);

% Agent options setting
ddpgOpts = rlDDPGAgentOptions( ...
               SampleTime            = settings.dt, ...
               TargetSmoothFactor    = 5e-3, ...            % τ
               DiscountFactor        = 0.99, ...            % gamma
               MiniBatchSize         = 256, ...
               ExperienceBufferLength= 1e5);

% Optimizators Tuning
ddpgOpts.ActorOptimizerOptions.LearnRate         = 1e-3;
ddpgOpts.CriticOptimizerOptions.LearnRate         = 1e-2;

ddpgOpts.NoiseOptions.Variance           = (settings.de_max*0.8)^2;
ddpgOpts.NoiseOptions.VarianceDecayRate  = 1e-4;
ddpgOpts.NoiseOptions.VarianceMin        = (settings.de_max*0)^2;

% Agent creation
agent = rlDDPGAgent(actor, critic, ddpgOpts);

%% Train the agent

settings.resultType = "training";

if settings.trainAgent == true
    % Define training settings for the agent
    trainOpts = rlTrainingOptions(...
        MaxEpisodes=settings.total_max_episodes, ...
        MaxStepsPerEpisode=200, ...
        Verbose=false, ...
        Plots="training-progress",...  
        StopTrainingCriteria="AverageReward", ...
        ScoreAveragingWindowLength = 50, ...
        StopTrainingValue = 550, ...
        UseParallel=settings.runParallel);
    % Train the agent
    trainingStats = train(agent,env,trainOpts);
    save(pwd + "/SimOut_Data/trainingStats.mat", 'trainingStats');
    save(pwd + "/SimOut_Agents/agentRef.mat", 'agent');

    if settings.runParallel == false
        Y_tot = env.LoggedSignals.cumulativeState;
        disp("Cumulative reward: " + ...
        num2str(sum(sum(env.LoggedSignals.cumulativeReward))))
    end
    disp("Training terminated")
    totEpisodes = trainingStats.EpisodeIndex(end);
    fprintf('Number of total episodes: %d \n', totEpisodes)
    disp("-----------------")
else
    load(pwd + "/SimOut_Agents/agentRef", 'agent');
    load(pwd + "/SimOut_Data/trainingStats.mat", 'trainingStats');
end

%% POST PROCESSING --------------------------------------------------------

if settings.PostProcess


% Extract and plot rewards timesries after training
f_rewards_plot(trainingStats, settings)

% Simulate the agent once training is over

fprintf('\n Simulation with trained policy... \n')

% No random values at the beginning: h0 = h_span
settings.RandomStart = false;

% Ignoring term conditions and new environment creation
settings.ignoreTermination = true;         
ResetHandleSim = @() funcReset(settings);
StepHandleSim  = @(Action,LoggedSignals) funcStep(Action,LoggedSignals,settings);
env = rlFunctionEnv(ObservationInfo,ActionInfo,...
                       StepHandleSim,ResetHandleSim);

% 1. opzioni di simulazione
simOpts = rlSimulationOptions( ...
            'MaxSteps',200, ...
            'NumSimulations',1);

% 2. run
experience = sim(env,agent,simOpts);

% 3. nomi campi auto-generati  (spazi → camelCase)
obsField = matlab.lang.makeValidName(ObservationInfo.Name);   % 'Cessna620StateVector'
actField = matlab.lang.makeValidName(ActionInfo.Name);        % 'Cessna620Guidance'

% 4. estrazione dati
t = experience.Observation.(obsField).Time;                 % column N
X = squeeze(experience.Observation.(obsField).Data);        % 5×N
U = squeeze(experience.Action.(actField).Data);             % 1×N-1
tU = t(1:numel(U));                                         % column N-1

% 5. plot essenziale
figure('Name','Simulation – RL controller','Position',[100 100 900 500])
lw = 1.5;

subplot(4,1,1)
plot(t, X(5,:),'LineWidth',lw); grid on
ylabel('\Delta h [m]'); title('Altitude');

subplot(4,1,2)
plot(t, rad2deg(X(4,:)),'LineWidth',lw); grid on
ylabel('\Delta \theta [deg]'); title('Pitch')

subplot(4,1,3)
plot(t, rad2deg(X(3,:)),'LineWidth',lw); grid on
ylabel('q [deg]'); title('Pitch Rate')

subplot(4,1,4)
plot(tU, rad2deg(U),'LineWidth',lw); grid on
ylabel('\delta_e [deg]'); xlabel('t [s]'); title('Elevator')

% Save figure
%saveas(gcf, fullfile('SimOut_Media', 'SimulationResults.jpg'));

% Robustness
settings.RandomStart = false;
h0_list = [10 5 0 -5 -10];
figure('Position', [100 100 1000 400]); 
hold on; grid on;

for i = 1:length(h0_list)
    settings.h0 = h0_list(i);
    % Simulation
    ResetHandleSim = @() funcReset(settings);
    StepHandleSim  = @(Action,LoggedSignals) funcStep(Action,LoggedSignals,settings);
    env = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandleSim,ResetHandleSim);
    experience = sim(env,agent,simOpts);
    % Data
    t = experience.Observation.(obsField).Time;
    X = squeeze(experience.Observation.(obsField).Data);
    % Plot
    plot(t, X(5,:), 'LineWidth',1.5, ...
        'DisplayName', sprintf('h_0 = %.0f m', h0_list(i)));
end

title('Different h_0 values');
xlabel('t [s]'); ylabel('\Delta h [m]'); legend('show');

% Save figure
%saveas(gcf, fullfile('SimOut_Media', 'Robustness_h0.jpg'));

end
