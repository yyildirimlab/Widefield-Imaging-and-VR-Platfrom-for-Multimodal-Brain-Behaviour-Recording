%% BATCH RANDOM FOREST ANALYSIS FOR MULTIPLE BRAIN REGIONS
% Processes bilateral, left, and right hemisphere data separately
% Outputs 3 Excel files with R² scores and performance metrics

clear; clc; close all;

%% Configuration
TOTAL_TIME = 300;
FILTER_ORDER = 3;
FILTER_WINDOW = 21;
ROLLING_WINDOW = 10;
N_TREES = 300;
MIN_LEAF = 5;
N_PREDICTORS_SAMPLE = 3;

% Base directory for brain data
BASE_DIR = "I:\data\LRI-110044\2025\April25\042225_TC\GEMM_6s_Group1_BlenderSess1\481\processed data\brain\rawBlue_rawViolet_rawHemodynamicSubtr";

% Output directory
OUTPUT_DIR = "I:\data\LRI-110044\2025\April25\042225_TC\GEMM_6s_Group1_BlenderSess1\481\processed data\glm";

% Define brain regions to analyze
brain_regions = {'MOp1','MOs1', 'RSPagl1', 'RSPd1', 'SSp_ll1','SSp_tr1','SSp_ul1','VISa1','VISam1','VISpm1'};

% Define hemisphere types
hemisphere_types = {'bilateral', 'left', 'right'};

%% Load Behavioral Data (Same for all regions)
fprintf('=== Loading Behavioral Data ===\n');
try
    pupil_data = load(fullfile("I:\data\LRI-110044\2025\April25\042225_TC\GEMM_6s_Group1_BlenderSess1\481\processed data\corrBrainBeh\behaviorVars\pupilDia.mat"));
    pupilDia = pupil_data.pupilDia;
    
    speed_data = load(fullfile("I:\data\LRI-110044\2025\April25\042225_TC\GEMM_6s_Group1_BlenderSess1\481\processed data\corrBrainBeh\behaviorVars\airballSpeed.mat"));
    orbital_speed_raw = speed_data.subsampledOrbitalAirballSpeed_smooth;
    
    fprintf('✓ Behavioral data loaded successfully\n\n');
catch ME
    error('Failed to load behavioral data: %s', ME.message);
end

%% Initialize Results Tables
column_names = {'Region', 'OOB_R2', 'OOB_RMSE', 'OOB_MAE', ...
                'CV_R2_Mean', 'CV_R2_Std', 'CV_RMSE_Mean', 'CV_RMSE_Std', ...
                'Top_Feature_1', 'Top_Feature_2', 'Top_Feature_3'};

results_bilateral = cell(length(brain_regions), length(column_names));
results_left = cell(length(brain_regions), length(column_names));
results_right = cell(length(brain_regions), length(column_names));

results_tables = struct();
results_tables.bilateral = results_bilateral;
results_tables.left = results_left;
results_tables.right = results_right;

%% Process Each Brain Region and Hemisphere
fprintf('========================================\n');
fprintf('   BATCH PROCESSING START\n');
fprintf('========================================\n\n');

for region_idx = 1:length(brain_regions)
    region = brain_regions{region_idx};
    fprintf('>>> Processing Region: %s (%d/%d)\n', region, region_idx, length(brain_regions));
    
    for hemi_idx = 1:length(hemisphere_types)
        hemisphere = hemisphere_types{hemi_idx};
        fprintf('  → Hemisphere: %s\n', hemisphere);
        
        % Construct filename
        filename = sprintf('%s_%s_rawBlue_rawViolet_rawHemodynamicSubtr.mat', ...
                          region, hemisphere);
        filepath = fullfile(BASE_DIR, filename);
        
        % Check if file exists
        if ~isfile(filepath)
            fprintf('    ⚠ File not found: %s\n', filename);
            continue;
        end
        
        try
            %% Load Brain Data
            brain_data = load(filepath);
            
            % Find the brain activity variable (it should contain 'hemodynamicSubtr_raw')
            fields = fieldnames(brain_data);
            brain_var = fields{contains(fields, 'hemodynamicSubtr_raw')};
            brain_activity_raw = brain_data.(brain_var);
            
            %% Preprocessing
            n_brain = length(brain_activity_raw);
            time_brain = linspace(0, TOTAL_TIME, n_brain);
            time_pupil = linspace(0, TOTAL_TIME, length(pupilDia));
            time_speed = linspace(0, TOTAL_TIME, length(orbital_speed_raw));
            
            % Resample
            pupil_resampled = interp1(time_pupil, pupilDia, time_brain, 'linear')';
            orbital_resampled = interp1(time_speed, orbital_speed_raw, time_brain, 'linear')';
            
            % Normalize THEN Smooth
            brain_activity_norm = zscore(brain_activity_raw);
            pupil_norm = zscore(pupil_resampled);
            orbital_norm = zscore(orbital_resampled);
            
            brain_activity = sgolayfilt(brain_activity_norm, FILTER_ORDER, FILTER_WINDOW);
            pupil_smooth = sgolayfilt(pupil_norm, FILTER_ORDER, FILTER_WINDOW);
            orbital_smooth = sgolayfilt(orbital_norm, FILTER_ORDER, FILTER_WINDOW);
            
            %% Feature Engineering
            X_features = [
                pupil_smooth, ...
                orbital_smooth, ...
                pupil_smooth.^2, ...
                orbital_smooth.^2, ...
                pupil_smooth .* orbital_smooth, ...
                movstd(pupil_smooth, ROLLING_WINDOW), ...
                movstd(orbital_smooth, ROLLING_WINDOW), ...
                [0; diff(pupil_smooth)], ...
                [0; diff(orbital_smooth)]
            ];
            
            feature_names = {'Pupil', 'Orbital', 'Pupil²', 'Orbital²', 'P×O', ...
                           'PupilStd', 'OrbitalStd', 'PupilVel', 'OrbitalVel'};
            
            %% Train Random Forest
            rf_model = TreeBagger(N_TREES, X_features, brain_activity, ...
                                 'Method', 'regression', ...
                                 'OOBPrediction', 'on', ...
                                 'OOBPredictorImportance', 'on', ...
                                 'MinLeafSize', MIN_LEAF, ...
                                 'NumPredictorsToSample', N_PREDICTORS_SAMPLE);
            
            %% Evaluate Model
            % OOB Performance
            oob_predictions = oobPredict(rf_model);
            oob_R2 = 1 - sum((brain_activity - oob_predictions).^2) / ...
                         sum((brain_activity - mean(brain_activity)).^2);
            oob_RMSE = sqrt(mean((brain_activity - oob_predictions).^2));
            oob_MAE = mean(abs(brain_activity - oob_predictions));
            
            % Cross-Validation
            n_folds = 10;
            cv = cvpartition(n_brain, 'KFold', n_folds);
            cv_R2 = zeros(n_folds, 1);
            cv_RMSE = zeros(n_folds, 1);
            
            for fold = 1:n_folds
                train_idx = training(cv, fold);
                test_idx = test(cv, fold);
                
                rf_cv = TreeBagger(N_TREES, X_features(train_idx,:), ...
                                  brain_activity(train_idx), ...
                                  'Method', 'regression', ...
                                  'MinLeafSize', MIN_LEAF, ...
                                  'NumPredictorsToSample', N_PREDICTORS_SAMPLE);
                
                pred_cv = predict(rf_cv, X_features(test_idx,:));
                
                cv_R2(fold) = 1 - sum((brain_activity(test_idx) - pred_cv).^2) / ...
                                 sum((brain_activity(test_idx) - mean(brain_activity(test_idx))).^2);
                cv_RMSE(fold) = sqrt(mean((brain_activity(test_idx) - pred_cv).^2));
            end
            
            %% Feature Importance
            importance = rf_model.OOBPermutedPredictorDeltaError;
            [sorted_imp, imp_idx] = sort(importance, 'descend');
            
            top_features = cell(1, 3);
            for i = 1:3
                top_features{i} = sprintf('%s (%.3f)', feature_names{imp_idx(i)}, sorted_imp(i));
            end
            
            %% Store Results
            result_row = {region, oob_R2, oob_RMSE, oob_MAE, ...
                         mean(cv_R2), std(cv_R2), mean(cv_RMSE), std(cv_RMSE), ...
                         top_features{1}, top_features{2}, top_features{3}};
            
            if strcmp(hemisphere, 'bilateral')
                results_tables.bilateral(region_idx, :) = result_row;
            elseif strcmp(hemisphere, 'left')
                results_tables.left(region_idx, :) = result_row;
            elseif strcmp(hemisphere, 'right')
                results_tables.right(region_idx, :) = result_row;
            end
            
            fprintf('    ✓ R² = %.4f, RMSE = %.4f, MAE = %.4f\n', ...
                   oob_R2, oob_RMSE, oob_MAE);
            
        catch ME
            fprintf('    ✗ Error processing %s: %s\n', filename, ME.message);
            
            % Store error indicator
            error_row = {region, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 'ERROR', 'ERROR', 'ERROR'};
            if strcmp(hemisphere, 'bilateral')
                results_tables.bilateral(region_idx, :) = error_row;
            elseif strcmp(hemisphere, 'left')
                results_tables.left(region_idx, :) = error_row;
            elseif strcmp(hemisphere, 'right')
                results_tables.right(region_idx, :) = error_row;
            end
        end
    end
    fprintf('\n');
end

%% Create and Save Excel Files
fprintf('========================================\n');
fprintf('   SAVING RESULTS TO EXCEL\n');
fprintf('========================================\n\n');

% Create tables
T_bilateral = cell2table([column_names; results_tables.bilateral], ...
                        'VariableNames', column_names);
T_bilateral = T_bilateral(2:end, :); % Remove header row

T_left = cell2table([column_names; results_tables.left], ...
                    'VariableNames', column_names);
T_left = T_left(2:end, :);

T_right = cell2table([column_names; results_tables.right], ...
                     'VariableNames', column_names);
T_right = T_right(2:end, :);

% Save to Excel in OUTPUT_DIR
try
    writetable(T_bilateral, fullfile(OUTPUT_DIR, 'RF_Results_Bilateral.xlsx'), 'Sheet', 'Bilateral');
    fprintf('✓ Bilateral results saved to: %s\n', fullfile(OUTPUT_DIR, 'RF_Results_Bilateral.xlsx'));
    
    writetable(T_left, fullfile(OUTPUT_DIR, 'RF_Results_Left.xlsx'), 'Sheet', 'Left');
    fprintf('✓ Left hemisphere results saved to: %s\n', fullfile(OUTPUT_DIR, 'RF_Results_Left.xlsx'));
    
    writetable(T_right, fullfile(OUTPUT_DIR, 'RF_Results_Right.xlsx'), 'Sheet', 'Right');
    fprintf('✓ Right hemisphere results saved to: %s\n', fullfile(OUTPUT_DIR, 'RF_Results_Right.xlsx'));
catch ME
    warning('Excel save failed: %s\nSaving as CSV instead...', ME.message);
    
    writetable(T_bilateral, fullfile(OUTPUT_DIR, 'RF_Results_Bilateral.csv'));
    writetable(T_left, fullfile(OUTPUT_DIR, 'RF_Results_Left.csv'));
    writetable(T_right, fullfile(OUTPUT_DIR, 'RF_Results_Right.csv'));
    
    fprintf('✓ Results saved as CSV files in: %s\n', OUTPUT_DIR);
end

%% Summary Statistics
fprintf('\n========================================\n');
fprintf('   SUMMARY STATISTICS\n');
fprintf('========================================\n\n');

fprintf('BILATERAL HEMISPHERE:\n');
bilateral_r2 = cell2mat(results_tables.bilateral(:, 2));
fprintf('  Mean R²: %.4f (± %.4f)\n', mean(bilateral_r2, 'omitnan'), std(bilateral_r2, 'omitnan'));
fprintf('  Best R²: %.4f (%s)\n', max(bilateral_r2), brain_regions{bilateral_r2 == max(bilateral_r2)});

fprintf('\nLEFT HEMISPHERE:\n');
left_r2 = cell2mat(results_tables.left(:, 2));
fprintf('  Mean R²: %.4f (± %.4f)\n', mean(left_r2, 'omitnan'), std(left_r2, 'omitnan'));
fprintf('  Best R²: %.4f (%s)\n', max(left_r2), brain_regions{left_r2 == max(left_r2)});

fprintf('\nRIGHT HEMISPHERE:\n');
right_r2 = cell2mat(results_tables.right(:, 2));
fprintf('  Mean R²: %.4f (± %.4f)\n', mean(right_r2, 'omitnan'), std(right_r2, 'omitnan'));
fprintf('  Best R²: %.4f (%s)\n', max(right_r2), brain_regions{right_r2 == max(right_r2)});

fprintf('\n========================================\n');
fprintf('   BATCH PROCESSING COMPLETE!\n');
fprintf('========================================\n');

%% Visualization: Comparison Across Regions
fig = figure('Position', [100 100 1400 800], 'Name', 'Multi-Region RF Performance');

% R² Comparison
subplot(2,2,1);
bar_data = [bilateral_r2, left_r2, right_r2];
bar(bar_data);
set(gca, 'XTickLabel', brain_regions, 'XTickLabelRotation', 45);
xlabel('Brain Region');
ylabel('R² Score');
title('R² Score Comparison Across Regions');
legend({'Bilateral', 'Left', 'Right'}, 'Location', 'best');
grid on;

% RMSE Comparison
subplot(2,2,2);
bilateral_rmse = cell2mat(results_tables.bilateral(:, 3));
left_rmse = cell2mat(results_tables.left(:, 3));
right_rmse = cell2mat(results_tables.right(:, 3));
bar_data = [bilateral_rmse, left_rmse, right_rmse];
bar(bar_data);
set(gca, 'XTickLabel', brain_regions, 'XTickLabelRotation', 45);
xlabel('Brain Region');
ylabel('RMSE');
title('RMSE Comparison Across Regions');
legend({'Bilateral', 'Left', 'Right'}, 'Location', 'best');
grid on;

% MAE Comparison
subplot(2,2,3);
bilateral_mae = cell2mat(results_tables.bilateral(:, 4));
left_mae = cell2mat(results_tables.left(:, 4));
right_mae = cell2mat(results_tables.right(:, 4));
bar_data = [bilateral_mae, left_mae, right_mae];
bar(bar_data);
set(gca, 'XTickLabel', brain_regions, 'XTickLabelRotation', 45);
xlabel('Brain Region');
ylabel('MAE');
title('MAE Comparison Across Regions');
legend({'Bilateral', 'Left', 'Right'}, 'Location', 'best');
grid on;

% CV R² with Error Bars
subplot(2,2,4);
bilateral_cv_mean = cell2mat(results_tables.bilateral(:, 5));
bilateral_cv_std = cell2mat(results_tables.bilateral(:, 6));
left_cv_mean = cell2mat(results_tables.left(:, 5));
left_cv_std = cell2mat(results_tables.left(:, 6));
right_cv_mean = cell2mat(results_tables.right(:, 5));
right_cv_std = cell2mat(results_tables.right(:, 6));

x = 1:length(brain_regions);
errorbar(x-0.2, bilateral_cv_mean, bilateral_cv_std, 'o-', 'LineWidth', 1.5); hold on;
errorbar(x, left_cv_mean, left_cv_std, 's-', 'LineWidth', 1.5);
errorbar(x+0.2, right_cv_mean, right_cv_std, '^-', 'LineWidth', 1.5);
set(gca, 'XTick', x, 'XTickLabel', brain_regions, 'XTickLabelRotation', 45);
xlabel('Brain Region');
ylabel('Cross-Validation R²');
title('CV R² with Standard Deviation');
legend({'Bilateral', 'Left', 'Right'}, 'Location', 'best');
grid on;

saveas(fig, fullfile(OUTPUT_DIR, 'MultiRegion_RF_Comparison.png'));
fprintf('\n✓ Comparison figure saved to: %s\n', fullfile(OUTPUT_DIR, 'MultiRegion_RF_Comparison.png'));