%% ULTIMATE GLM MODEL - Maximum R² Strategy
% Combining: Smoothing + Rolling Stats + Polynomial + Interactions + Non-Linear Models

%% Configuration
TOTAL_TIME = 300;
FILTER_ORDER = 3;
FILTER_WINDOW = 21;
ROLLING_WINDOW = 10;

%% Load Data
try
    brain_data = load(fullfile("I:\data\LRI-110044\2024\Aug24\082424_TC\GBM_Group1_BaselineSess1\152\Processed Data\Brain\Allen\rawBlue_rawViolet_rawHemodynamicSubtr\MOp1_bilateral_rawBlue_rawViolet_rawHemodynamicSubtr.mat"));
    brain_activity_raw = brain_data.MOp1_bilateral_hemodynamicSubtr_raw;
    
    pupil_data = load(fullfile("I:\data\LRI-110044\2024\Aug24\082424_TC\GBM_Group1_BaselineSess1\152\Processed Data\HMM\behaviorVars\pupilDia.mat"));
    pupilDia = pupil_data.pupilDia;
    
    speed_data = load(fullfile("I:\data\LRI-110044\2024\Aug24\082424_TC\GBM_Group1_BaselineSess1\152\Processed Data\HMM\behaviorVars\airballSpeed.mat"));
    orbital_speed_raw = speed_data.subsampledOrbitalAirballSpeed;
    
    fprintf('✓ All data loaded successfully\n');
catch ME
    error('Failed to load data: %s', ME.message);
end

%% Resample
n_brain = length(brain_activity_raw);
time_brain = linspace(0, TOTAL_TIME, n_brain);
time_pupil = linspace(0, TOTAL_TIME, length(pupilDia));
time_speed = linspace(0, TOTAL_TIME, length(orbital_speed_raw));

pupil_resampled = interp1(time_pupil, pupilDia, time_brain, 'linear')';
orbital_resampled = interp1(time_speed, orbital_speed_raw, time_brain, 'linear')';

%% Normalize THEN Smooth (Critical Order!)
brain_activity_norm = zscore(brain_activity_raw);
pupil_norm = zscore(pupil_resampled);
orbital_norm = zscore(orbital_resampled);

% Apply Savitzky-Golay smoothing
brain_activity = sgolayfilt(brain_activity_norm, FILTER_ORDER, FILTER_WINDOW);
pupil_smooth = sgolayfilt(pupil_norm, FILTER_ORDER, FILTER_WINDOW);
orbital_smooth = sgolayfilt(orbital_norm, FILTER_ORDER, FILTER_WINDOW);

fprintf('\n========================================\n');
fprintf('   ADVANCED MODEL TESTING\n');
fprintf('========================================\n');

%% Create Advanced Features on SMOOTHED data
% Linear
feat_pupil = pupil_smooth;
feat_orbital = orbital_smooth;

% Polynomial
feat_pupil_sq = pupil_smooth.^2;
feat_orbital_sq = orbital_smooth.^2;
feat_pupil_cube = pupil_smooth.^3;
feat_orbital_cube = orbital_smooth.^3;

% Interactions
feat_interaction = pupil_smooth .* orbital_smooth;
feat_interaction2 = (pupil_smooth.^2) .* orbital_smooth;
feat_interaction3 = pupil_smooth .* (orbital_smooth.^2);

% Rolling statistics on smoothed data
feat_pupil_roll_mean = movmean(pupil_smooth, ROLLING_WINDOW);
feat_orbital_roll_mean = movmean(orbital_smooth, ROLLING_WINDOW);
feat_pupil_roll_std = movstd(pupil_smooth, ROLLING_WINDOW);
feat_orbital_roll_std = movstd(orbital_smooth, ROLLING_WINDOW);

% Temporal derivatives
feat_pupil_vel = [0; diff(pupil_smooth)];
feat_orbital_vel = [0; diff(orbital_smooth)];
feat_pupil_accel = [0; 0; diff(diff(pupil_smooth))];
feat_orbital_accel = [0; 0; diff(diff(orbital_smooth))];

%% STRATEGY 1: Test GLM Feature Combinations
fprintf('\n=== Strategy 1: GLM with Different Feature Sets ===\n');

feature_combos = {
    struct('name', 'Smoothed Basic', ...
           'X', [feat_pupil, feat_orbital]), ...
    
    struct('name', 'Smoothed + Polynomial', ...
           'X', [feat_pupil, feat_orbital, feat_pupil_sq, feat_orbital_sq, ...
                 feat_interaction]), ...
    
    struct('name', 'Smoothed + Rolling Stats', ...
           'X', [feat_pupil, feat_orbital, ...
                 feat_pupil_roll_mean, feat_orbital_roll_mean, ...
                 feat_pupil_roll_std, feat_orbital_roll_std]), ...
    
    struct('name', 'Smoothed + Poly + Rolling', ...
           'X', [feat_pupil, feat_orbital, feat_pupil_sq, feat_orbital_sq, ...
                 feat_interaction, feat_pupil_roll_std, feat_orbital_roll_std]), ...
    
    struct('name', 'Smoothed + Temporal', ...
           'X', [feat_pupil, feat_orbital, feat_pupil_sq, feat_orbital_sq, ...
                 feat_interaction, feat_pupil_vel, feat_orbital_vel]), ...
    
    struct('name', 'COMPREHENSIVE (All Features)', ...
           'X', [feat_pupil, feat_orbital, ...
                 feat_pupil_sq, feat_orbital_sq, feat_pupil_cube, feat_orbital_cube, ...
                 feat_interaction, feat_interaction2, feat_interaction3, ...
                 feat_pupil_roll_mean, feat_orbital_roll_mean, ...
                 feat_pupil_roll_std, feat_orbital_roll_std, ...
                 feat_pupil_vel, feat_orbital_vel])
};

%% Evaluate All GLM Combinations
n_folds = 10;
glm_results = [];

for i = 1:length(feature_combos)
    combo = feature_combos{i};
    fprintf('\nTesting: %s (%d features)\n', combo.name, size(combo.X, 2));
    
    % Fit model
    mdl = fitglm(combo.X, brain_activity, 'Distribution', 'normal');
    
    % Cross-validation
    cv = cvpartition(n_brain, 'KFold', n_folds);
    cv_R2 = zeros(n_folds, 1);
    
    for fold = 1:n_folds
        train_idx = training(cv, fold);
        test_idx = test(cv, fold);
        
        mdl_cv = fitglm(combo.X(train_idx,:), brain_activity(train_idx), ...
                        'Distribution', 'normal');
        pred_cv = predict(mdl_cv, combo.X(test_idx,:));
        
        cv_R2(fold) = 1 - sum((brain_activity(test_idx) - pred_cv).^2) / ...
                          sum((brain_activity(test_idx) - mean(brain_activity(test_idx))).^2);
    end
    
    glm_results(i).name = combo.name;
    glm_results(i).n_features = size(combo.X, 2);
    glm_results(i).R2_train = mdl.Rsquared.Ordinary;
    glm_results(i).R2_adj = mdl.Rsquared.Adjusted;
    glm_results(i).R2_cv_mean = mean(cv_R2);
    glm_results(i).R2_cv_std = std(cv_R2);
    glm_results(i).model = mdl;
    glm_results(i).X = combo.X;
    
    fprintf('  Train R²: %.4f | Adj R²: %.4f | CV R²: %.4f ± %.4f\n', ...
            mdl.Rsquared.Ordinary, mdl.Rsquared.Adjusted, mean(cv_R2), std(cv_R2));
end

%% Find Best GLM
[best_glm_R2, best_glm_idx] = max([glm_results.R2_cv_mean]);
best_glm_model = glm_results(best_glm_idx);

fprintf('\n🏆 Best GLM: %s (CV R² = %.4f ± %.4f)\n', ...
        best_glm_model.name, best_glm_model.R2_cv_mean, best_glm_model.R2_cv_std);

%% STRATEGY 2: Random Forest
fprintf('\n=== Strategy 2: Random Forest ===\n');

% Use comprehensive feature set
X_comprehensive = [feat_pupil, feat_orbital, ...
                   feat_pupil_sq, feat_orbital_sq, ...
                   feat_interaction, ...
                   feat_pupil_roll_std, feat_orbital_roll_std, ...
                   feat_pupil_vel, feat_orbital_vel];

feature_names_rf = {'Pupil', 'Orbital', 'Pupil²', 'Orbital²', 'P×O', ...
                    'PupilStd', 'OrbitalStd', 'PupilVel', 'OrbitalVel'};

fprintf('Training Random Forest with %d features...\n', size(X_comprehensive, 2));

% Random Forest with feature importance
n_trees = 200;
rf_model = TreeBagger(n_trees, X_comprehensive, brain_activity, ...
                      'Method', 'regression', ...
                      'OOBPrediction', 'on', ...
                      'OOBPredictorImportance', 'on', ...  % FIXED: Added this line
                      'MinLeafSize', 5, ...
                      'NumPredictorsToSample', 3);

% Out-of-bag R²
oob_predictions = oobPredict(rf_model);
oob_R2 = 1 - sum((brain_activity - oob_predictions).^2) / ...
             sum((brain_activity - mean(brain_activity)).^2);

fprintf('Random Forest OOB R²: %.4f\n', oob_R2);

% Cross-validation for RF
fprintf('Cross-validating Random Forest...\n');
cv_R2_rf = zeros(10, 1);
cv = cvpartition(n_brain, 'KFold', 10);

for fold = 1:10
    train_idx = training(cv, fold);
    test_idx = test(cv, fold);
    
    rf_cv = TreeBagger(n_trees, X_comprehensive(train_idx,:), ...
                       brain_activity(train_idx), ...
                       'Method', 'regression', ...
                       'MinLeafSize', 5, ...
                       'NumPredictorsToSample', 3);
    pred_rf = predict(rf_cv, X_comprehensive(test_idx,:));
    
    cv_R2_rf(fold) = 1 - sum((brain_activity(test_idx) - pred_rf).^2) / ...
                         sum((brain_activity(test_idx) - mean(brain_activity(test_idx))).^2);
end

fprintf('Random Forest CV R²: %.4f ± %.4f\n', mean(cv_R2_rf), std(cv_R2_rf));

% Feature importance (NOW IT WILL WORK)
feature_importance = rf_model.OOBPermutedPredictorDeltaError;
[sorted_importance, imp_idx] = sort(feature_importance, 'descend');

fprintf('\nRandom Forest Feature Importance:\n');
for i = 1:length(feature_names_rf)
    fprintf('  %s: %.4f\n', feature_names_rf{imp_idx(i)}, sorted_importance(i));
end

%% STRATEGY 3: Support Vector Regression
fprintf('\n=== Strategy 3: Support Vector Regression ===\n');

kernels = {'linear', 'gaussian', 'polynomial'};
svr_results = struct();

for k = 1:length(kernels)
    kernel_name = kernels{k};
    fprintf('\nTesting SVR with %s kernel...\n', kernel_name);
    
    % Cross-validation
    cv_R2_svr = zeros(10, 1);
    cv = cvpartition(n_brain, 'KFold', 10);
    
    for fold = 1:10
        train_idx = training(cv, fold);
        test_idx = test(cv, fold);
        
        try
            if strcmp(kernel_name, 'polynomial')
                svr_model = fitrsvm(X_comprehensive(train_idx,:), ...
                                   brain_activity(train_idx), ...
                                   'KernelFunction', kernel_name, ...
                                   'PolynomialOrder', 2, ...
                                   'Standardize', true, ...
                                   'KernelScale', 'auto');
            else
                svr_model = fitrsvm(X_comprehensive(train_idx,:), ...
                                   brain_activity(train_idx), ...
                                   'KernelFunction', kernel_name, ...
                                   'Standardize', true, ...
                                   'KernelScale', 'auto');
            end
            
            pred_svr = predict(svr_model, X_comprehensive(test_idx,:));
            
            cv_R2_svr(fold) = 1 - sum((brain_activity(test_idx) - pred_svr).^2) / ...
                                  sum((brain_activity(test_idx) - mean(brain_activity(test_idx))).^2);
        catch
            cv_R2_svr(fold) = NaN;
        end
    end
    
    svr_results(k).kernel = kernel_name;
    svr_results(k).cv_R2_mean = nanmean(cv_R2_svr);
    svr_results(k).cv_R2_std = nanstd(cv_R2_svr);
    
    fprintf('  CV R²: %.4f ± %.4f\n', svr_results(k).cv_R2_mean, svr_results(k).cv_R2_std);
end

% Find best SVR
[best_svr_R2, best_svr_idx] = max([svr_results.cv_R2_mean]);
fprintf('\n🏆 Best SVR: %s kernel (CV R² = %.4f)\n', ...
        svr_results(best_svr_idx).kernel, best_svr_R2);

%% STRATEGY 4: Gaussian Process Regression
fprintf('\n=== Strategy 4: Gaussian Process Regression ===\n');

% Due to computational cost, use subset of data for GP
subset_ratio = 0.3;  % Use 30% of data
subset_size = round(n_brain * subset_ratio);
subset_idx = randperm(n_brain, subset_size);

X_gp_subset = X_comprehensive(subset_idx, :);
y_gp_subset = brain_activity(subset_idx);

fprintf('Training GP on %d samples (%.0f%% of data)...\n', subset_size, subset_ratio*100);

try
    gp_model = fitrgp(X_gp_subset, y_gp_subset, ...
                      'KernelFunction', 'ardsquaredexponential', ...
                      'Standardize', true);
    
    % Predict on full data
    gp_predictions = predict(gp_model, X_comprehensive);
    gp_R2 = 1 - sum((brain_activity - gp_predictions).^2) / ...
                sum((brain_activity - mean(brain_activity)).^2);
    
    fprintf('Gaussian Process R² (on full data): %.4f\n', gp_R2);
catch ME
    fprintf('GP failed: %s\n', ME.message);
    gp_R2 = NaN;
end

%% COMPREHENSIVE COMPARISON
fprintf('\n========================================\n');
fprintf('   FINAL MODEL COMPARISON\n');
fprintf('========================================\n');
fprintf('%-35s | %12s\n', 'Model', 'CV R²');
fprintf('%s\n', repmat('-', 1, 55));

% Previous results
fprintf('%-35s | %.4f ± %.4f\n', 'Basic (No Smoothing)', 0.3357, 0.0379);
fprintf('%-35s | %.4f ± %.4f\n', 'Rolling Stats (No Smoothing)', 0.4248, 0.0548);

% New GLM results
for i = 1:length(glm_results)
    fprintf('%-35s | %.4f ± %.4f\n', glm_results(i).name, ...
            glm_results(i).R2_cv_mean, glm_results(i).R2_cv_std);
end

% Non-linear results
fprintf('%-35s | %.4f ± %.4f\n', 'Random Forest', mean(cv_R2_rf), std(cv_R2_rf));
for k = 1:length(svr_results)
    fprintf('%-35s | %.4f ± %.4f\n', sprintf('SVR (%s)', svr_results(k).kernel), ...
            svr_results(k).cv_R2_mean, svr_results(k).cv_R2_std);
end
if ~isnan(gp_R2)
    fprintf('%-35s | %.4f (subset)\n', 'Gaussian Process', gp_R2);
end

fprintf('%s\n', repmat('-', 1, 55));

%% Find Overall Best Model
all_results = struct();
idx = 1;

% Add GLM results
for i = 1:length(glm_results)
    all_results(idx).name = glm_results(i).name;
    all_results(idx).type = 'GLM';
    all_results(idx).cv_R2 = glm_results(i).R2_cv_mean;
    all_results(idx).cv_std = glm_results(i).R2_cv_std;
    idx = idx + 1;
end

% Add RF
all_results(idx).name = 'Random Forest';
all_results(idx).type = 'Ensemble';
all_results(idx).cv_R2 = mean(cv_R2_rf);
all_results(idx).cv_std = std(cv_R2_rf);
idx = idx + 1;

% Add SVR
for k = 1:length(svr_results)
    all_results(idx).name = sprintf('SVR (%s)', svr_results(k).kernel);
    all_results(idx).type = 'SVR';
    all_results(idx).cv_R2 = svr_results(k).cv_R2_mean;
    all_results(idx).cv_std = svr_results(k).cv_R2_std;
    idx = idx + 1;
end

% Find best
[max_R2, max_idx] = max([all_results.cv_R2]);
best_overall = all_results(max_idx);

fprintf('\n🏆🏆🏆 OVERALL WINNER 🏆🏆🏆\n');
fprintf('Model: %s (%s)\n', best_overall.name, best_overall.type);
fprintf('CV R²: %.4f ± %.4f\n', best_overall.cv_R2, best_overall.cv_std);
fprintf('\nImprovement over baseline: %.1f%%\n', ...
        (best_overall.cv_R2 - 0.3357) / 0.3357 * 100);

%% Visualization
figure('Position', [50 50 1600 800]);

% Plot 1: Model Comparison Bar Chart
subplot(2,3,1);
bar([all_results.cv_R2]);
hold on;
errorbar(1:length(all_results), [all_results.cv_R2], [all_results.cv_std], ...
         'k.', 'LineWidth', 2);
set(gca, 'XTickLabel', {all_results.name}, 'XTickLabelRotation', 45);
ylabel('CV R²');
title('Model Performance Comparison');
grid on;
ylim([0 1]);

% Plot 2: Best Model Predictions
subplot(2,3,2);
if strcmp(best_overall.type, 'GLM')
    best_pred = predict(best_glm_model.model, best_glm_model.X);
elseif strcmp(best_overall.type, 'Ensemble')
    best_pred = oob_predictions;
else
    best_pred = brain_activity; % Placeholder
end

scatter(brain_activity, best_pred, 20, 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot(xlim, xlim, 'k--', 'LineWidth', 2);
xlabel('Actual Brain Activity');
ylabel('Predicted Brain Activity');
title(sprintf('%s\n(CV R² = %.3f)', best_overall.name, best_overall.cv_R2));
axis equal; grid on;

% Plot 3: Feature Importance (Random Forest)
subplot(2,3,3);
barh(sorted_importance);
set(gca, 'YTick', 1:length(feature_names_rf), ...
         'YTickLabel', feature_names_rf(imp_idx));
xlabel('Importance');
title('Random Forest Feature Importance');
grid on;

% Plot 4: Time Series Comparison (Best Model)
subplot(2,3,4);
t = (1:length(brain_activity)) * TOTAL_TIME / length(brain_activity);
plot(t, brain_activity, 'b', 'LineWidth', 1); hold on;
plot(t, best_pred, 'r', 'LineWidth', 1);
xlabel('Time (seconds)');
ylabel('Brain Activity (z-scored)');
legend('Actual', 'Predicted', 'Location', 'best');
title('Best Model: Time Series');
grid on;

% Plot 5: Residuals
subplot(2,3,5);
residuals = brain_activity - best_pred;
histogram(residuals, 50, 'Normalization', 'pdf', 'FaceAlpha', 0.7);
xlabel('Residual Value');
ylabel('Density');
title(sprintf('Residuals (RMSE = %.3f)', sqrt(mean(residuals.^2))));
grid on;

% Plot 6: Performance by Model Type
subplot(2,3,6);
types = unique({all_results.type});
type_R2 = zeros(length(types), 1);
for i = 1:length(types)
    type_idx = strcmp({all_results.type}, types{i});
    type_R2(i) = max([all_results(type_idx).cv_R2]);
end
bar(type_R2);
set(gca, 'XTickLabel', types);
ylabel('Best CV R²');
title('Best Performance by Model Type');
grid on;
ylim([0 1]);

%% Save Results
save('Ultimate_Model_Comparison.mat', 'all_results', 'best_overall', ...
     'glm_results', 'rf_model', 'svr_results');
fprintf('\n✓ All results saved to Ultimate_Model_Comparison.mat\n');