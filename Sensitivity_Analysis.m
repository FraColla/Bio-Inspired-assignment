% Perform sensitivity analysis by plotting the Avg Rewards from different
% datasets

%% LR

% Nominal case
load(pwd + "/SimOut_Data/trainingStats_Nominal.mat"); 
episodes_nominal = trainingStats.EpisodeIndex;
rewards_nominal = trainingStats.AverageReward;

% higher LR
load(pwd + "/SimOut_Data/trainingStats_LRplus.mat"); 
episodes_LRplus = trainingStats.EpisodeIndex;
rewards_LRplus = trainingStats.AverageReward;

% lower LR
load(pwd + "/SimOut_Data/trainingStats_LRminus.mat"); 
episodes_LRminus = trainingStats.EpisodeIndex;
rewards_LRminus = trainingStats.AverageReward;

% Plot
figure('Position',[100 100 1000 400]); hold on; grid on;
plot(episodes_nominal, rewards_nominal, 'b', 'LineWidth',2, 'DisplayName','Nominal');
plot(episodes_LRminus, rewards_LRminus, 'r', 'LineWidth',2, 'DisplayName','Low learning rate');
plot(episodes_LRplus, rewards_LRplus, 'g', 'LineWidth',2, 'DisplayName','High learning rate');
yline(550, 'k--', 'LineWidth',1, 'DisplayName', 'threshold')
xlabel('Episodes');
ylabel('Average reward');
title('Comparison of average reward for different LR values');
legend('show','Location','best');
saveas(gcf, fullfile('SimOut_Media', 'SensAnalysis_LR.jpg'));

%% MBS

% Nominal
load(pwd + "/SimOut_Data/trainingStats_Nominal.mat"); 
episodes_nominal = trainingStats.EpisodeIndex;
rewards_nominal = trainingStats.AverageReward;

% higher MBS
load(pwd + "/SimOut_Data/trainingStats_MBSplus.mat"); 
episodes_MBSplus = trainingStats.EpisodeIndex;
rewards_MBSplus = trainingStats.AverageReward;

% lower MBS
load(pwd + "/SimOut_Data/trainingStats_MBSminus.mat"); 
episodes_MBSminus = trainingStats.EpisodeIndex;
rewards_MBSminus = trainingStats.AverageReward;

% Plot
figure('Position',[100 100 1000 400]); hold on; grid on;
plot(episodes_nominal, rewards_nominal, 'b', 'LineWidth',2, 'DisplayName','Nominal');
plot(episodes_MBSminus, rewards_MBSminus, 'r', 'LineWidth',2, 'DisplayName','MBS = 128');
plot(episodes_MBSplus, rewards_MBSplus, 'g', 'LineWidth',2, 'DisplayName','MBS = 512');
yline(550, 'k--', 'LineWidth',1, 'DisplayName', 'threshold')
xlabel('Episodes');
ylabel('Average reward');
title('Comparison of average reward for different MBS values');
legend('show','Location','best');
saveas(gcf, fullfile('SimOut_Media', 'SensAnalysis_MBS.jpg'));

%% Nodes number

% Nominal
load(pwd + "/SimOut_Data/trainingStats_Nominal.mat"); 
episodes_nominal = trainingStats.EpisodeIndex;
rewards_nominal = trainingStats.AverageReward;

% higher nodes number
load(pwd + "/SimOut_Data/trainingStats_Nodesplus.mat"); 
episodes_Nodesplus = trainingStats.EpisodeIndex;
rewards_Nodesplus = trainingStats.AverageReward;

% lower nodes number
load(pwd + "/SimOut_Data/trainingStats_Nodeminus.mat"); 
episodes_Nodesminus = trainingStats.EpisodeIndex;
rewards_Nodesminus = trainingStats.AverageReward;

% Plot
figure('Position',[100 100 1000 400]); hold on; grid on;
plot(episodes_nominal, rewards_nominal, 'b', 'LineWidth',2, 'DisplayName','Nominal');
plot(episodes_Nodesminus, rewards_Nodesminus, 'r', 'LineWidth',2, 'DisplayName','Lower nodes number');
plot(episodes_Nodesplus, rewards_Nodesplus, 'g', 'LineWidth',2, 'DisplayName','Higher nodes number');
yline(550, 'k--', 'LineWidth',1, 'DisplayName', 'threshold')
xlabel('Episodes');
ylabel('Average reward');
title('Comparison of average reward for different number of nodes values');
legend('show','Location','best');
saveas(gcf, fullfile('SimOut_Media', 'SensAnalysis_Nodes.jpg'));

%% Discount factor

% Nominal
load(pwd + "/SimOut_Data/trainingStats_Nominal.mat"); 
episodes_nominal = trainingStats.EpisodeIndex;
rewards_nominal = trainingStats.AverageReward;

% higher gamma
load(pwd + "/SimOut_Data/trainingStats_Gammaplus.mat"); 
episodes_Gammaplus = trainingStats.EpisodeIndex;
rewards_Gammaplus = trainingStats.AverageReward;

% lower gamma
load(pwd + "/SimOut_Data/trainingStats_Gammaminus.mat"); 
episodes_Gammaminus = trainingStats.EpisodeIndex;
rewards_Gammaminus = trainingStats.AverageReward;

% Plot
figure; hold on; grid on;
plot(episodes_nominal, rewards_nominal, 'b', 'LineWidth',2, 'DisplayName','Nominal');
plot(episodes_Gammaminus, rewards_Gammaminus, 'r', 'LineWidth',2, 'DisplayName','\gamma = 0.900');
plot(episodes_Gammaplus, rewards_Gammaplus, 'g', 'LineWidth',2, 'DisplayName','\gamma = 0.999');
yline(550, 'k--', 'LineWidth',1, 'DisplayName', 'threshold')
xlabel('Episodes');
ylabel('Average reward');
title('Comparison of average reward for different \gamma values');
legend('show','Location','best');
saveas(gcf, fullfile('SimOut_Media', 'SensAnalysis_Gamma.jpg'));

% Performances comparison ------------------------------------------------
    
load(pwd + "/SimOut_Agents/agentRef_Nominal", 'agent');
Anom = agent;
load(pwd + "/SimOut_Agents/agentRef_Gammaplus", 'agent');
Aplus = agent;
load(pwd + "/SimOut_Agents/agentRef_Gammaminus", 'agent');
Amin = agent;

% Settings
settings.RandomStart = false;
settings.h0 = 10;
figure; hold on; grid on;

% Nominal
ResetHandleSim = @() funcReset(settings);
StepHandleSim  = @(Action,LoggedSignals) funcStep(Action,LoggedSignals,settings);
env = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandleSim,ResetHandleSim);
experience = sim(env, Anom, simOpts);
t_nom = experience.Observation.(obsField).Time;
X_nom = squeeze(experience.Observation.(obsField).Data);

% Gamma plus
ResetHandleSim = @() funcReset(settings);
StepHandleSim  = @(Action,LoggedSignals) funcStep(Action,LoggedSignals,settings);
env = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandleSim,ResetHandleSim);
experience = sim(env, Aplus, simOpts);
t_plus = experience.Observation.(obsField).Time;
X_plus = squeeze(experience.Observation.(obsField).Data);

% Gamma minus
ResetHandleSim = @() funcReset(settings);
StepHandleSim  = @(Action,LoggedSignals) funcStep(Action,LoggedSignals,settings);
env = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandleSim,ResetHandleSim);
experience = sim(env, Amin, simOpts);
t_min = experience.Observation.(obsField).Time;
X_min = squeeze(experience.Observation.(obsField).Data);

% Plot
plot(t_nom, X_nom(5,:), 'b-', 'LineWidth',1.5, 'DisplayName','Nominal');
plot(t_min, X_min(5,:), 'r-', 'LineWidth',1.5, 'DisplayName','\gamma = 0.900');
plot(t_plus, X_plus(5,:), 'g-', 'LineWidth',1.5, 'DisplayName','\gamma = 0.999');
xlabel('Time [s]');
ylabel('Altitude deviation \Delta h [m]');
title('Performance comparison for different \gamma values');
legend('show','Location','best');
saveas(gcf, fullfile('SimOut_Media', 'Gamma_h_perform.jpg'));
