%% Limpieza inicial
clc; clear; close all;
global X_train Y_train X_val Y_val nD

%% Carga de datos
Dataset2 = readtable("heart_limpio.csv");
x = Dataset2{:, 1:end-1};
y = Dataset2{:, end};
nombre_features = Dataset2.Properties.VariableNames(1:end-1);
[n_muestras, n_features] = size(x);
nD = n_features;

%% Configuración previa para algoritmos
n_features_a_seleccionar = floor(n_features / 2);
step_rfe = 1;

% BSFOA
Npop   = 30;
Max_it = 50;
lb = -4; ub = 4;

%% Parámetros del bucle externo
N_runs = 5;  % Ajustado a 5 ejecuciones
K_folds = 10; % Definido explicitamente para usar en partición y tiempos
carpeta_imagenes = 'Graficos_Heart'; % Carpeta para exportar .eps

%% Sistema de guardado / reanudación
archivo_progreso = 'progreso_FeatureMainDATA2.mat';

if isfile(archivo_progreso)
    fprintf('>> Archivo de progreso encontrado. Reanudando...\n');
    load(archivo_progreso); 
    if ~exist('curvas_v1_total', 'var')
        curvas_original_total = zeros(N_runs, K_folds, Max_it);
        curvas_v1_total = zeros(N_runs, K_folds, Max_it); 
        curvas_v2_total = zeros(N_runs, K_folds, Max_it);
        curvas_v3_total = zeros(N_runs, K_folds, Max_it);
    end
else
    % Acumuladores por ejecución (runs × features)
    acum_rfe            = zeros(N_runs, n_features);
    acum_pi             = zeros(N_runs, n_features);
    acum_sfoa_original_total = zeros(N_runs, n_features);
    acum_cbsfoa_forest  = zeros(N_runs, n_features);
    acum_cbsfoa_forest2 = zeros(N_runs, n_features);
    acum_cbsfoa_forest3 = zeros(N_runs, n_features);
    
    % Tiempos totales por algoritmo y ejecución (runs × 1)
    tiempos_rfe            = zeros(N_runs, 1);
    tiempos_pi             = zeros(N_runs, 1);
    tiempos_sfoa_original_total = zeros(N_runs,1);
    tiempos_cbsfoa_forest  = zeros(N_runs, 1);
    tiempos_cbsfoa_forest2 = zeros(N_runs, 1);
    tiempos_cbsfoa_forest3 = zeros(N_runs, 1);
    
    curvas_original_total = zeros(N_runs, K_folds, Max_it);
    curvas_v1_total = zeros(N_runs, K_folds, Max_it);
    curvas_v2_total = zeros(N_runs, K_folds, Max_it);
    curvas_v3_total = zeros(N_runs, K_folds, Max_it);

    
    ultima_run  = 0;
    ultima_fold = 0;
    fprintf('>> Iniciando desde cero.\n');
end

%% ══════════════════════════════════════════════════════════════════════════
%%  BUCLE EXTERNO: N ejecuciones independientes
%% ══════════════════════════════════════════════════════════════════════════
for run = (ultima_run + 1) : N_runs
    fprintf('\n========================================\n');
    fprintf('  EJECUCIÓN %d de %d\n', run, N_runs);
    fprintf('========================================\n');
    
    % Semilla distinta por ejecución
    rng(run * 10);
    c = cvpartition(y, 'KFold', K_folds);
    
    es_reanudacion = (run == ultima_run + 1 && ultima_fold > 0);
    if es_reanudacion
        fold_inicio = ultima_fold + 1;
        fprintf('   (Reanudando variables parciales desde fold %d)\n', fold_inicio);
    else
        fold_inicio = 1;
        fprintf('   (Iniciando run desde cero)\n');
        
        cont_rfe            = zeros(1, n_features);
        cont_pi             = zeros(1, n_features);
        cont_sfoa_original  = zeros(1, nD);
        cont_cbsfoa_forest  = zeros(1, nD);
        cont_cbsfoa_forest2 = zeros(1, nD);
        cont_cbsfoa_forest3 = zeros(1, nD);
        
        t_rfe_run   = 0;  
        t_pi_run = 0;
        t_sfoa_run = 0;
        t_cbsf1_run = 0;  
        t_cbsf2_run  = 0;
        t_cbsf3_run  = 0;
    end
    
    %% ── BUCLE INTERNO: KFold ────────────────────────────────────────────
    for i = fold_inicio : c.NumTestSets
        fprintf('\n  [Run %d] Ejecutando Fold %d de %d\n', run, i, c.NumTestSets);
        try
            trainIdx = training(c, i);
            testIdx  = test(c, i);
            X_train = x(trainIdx, :);
            Y_train = y(trainIdx, :);
            X_val   = x(testIdx, :);
            Y_val   = y(testIdx, :);
            X_test  = X_val;
            Y_test  = Y_val;
            
            %% ── RFE ─────────────────────────────────────────────────────
            fprintf('    > RFE... ');
            tic;
            fn_importancia_rfe = @(X_sub, y_sub) obtener_importancia_rf(X_sub, y_sub);
            [support_rfe, ~] = rfe(X_train, Y_train, fn_importancia_rfe, n_features_a_seleccionar, step_rfe);
            t_rfe = toc;
            cont_rfe   = cont_rfe + support_rfe;
            t_rfe_run  = t_rfe_run + t_rfe;
            fprintf('%.2f s\n', t_rfe);
            
            %% ── Permutation Importance ──────────────────────────────────
            fprintf('    > Permutation Importance... ');
            tic;
            modelo_pi = TreeBagger(50, X_train, Y_train, 'Method', 'classification', 'OOBPredictorImportance', 'on');
            metric_fn = @(y_real, y_pred) sum(y_real == str2double(y_pred)) / length(y_real);
            res_pi = permutation_importance(modelo_pi, X_test, Y_test, metric_fn, 5);
            t_pi = toc;
            [~, idx_ordenado_pi] = sort(res_pi.importances_mean, 'descend');
            support_pi = false(1, n_features);
            support_pi(idx_ordenado_pi(1:n_features_a_seleccionar)) = true;
            cont_pi   = cont_pi + support_pi;
            t_pi_run  = t_pi_run + t_pi;
            fprintf('%.2f s\n', t_pi);
            %% ── SFOA OR ───────────────────────────────────────────────

            fprintf('    > SFOA Orginal (Mini-Forest)... ');
            tic;
            [~, ~, Curve_Or, Sf_Or, ~] = BSFOASig(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
            t_sfoa_or = toc;
            support_Original = false(1, nD); support_Original(Sf_Or) = true;
            cont_sfoa_original  = cont_sfoa_original + support_Original;
            t_sfoa_run        = t_sfoa_run + t_sfoa_or;
            curvas_original_total(run, i,:) = Curve_Or;
            fprintf('%.2f s\n', t_sfoa_or);
            
            %% ── CBSFOA v1 ───────────────────────────────────────────────
            fprintf('    > CBSFOA v1 (Mini-Forest)... ');
            tic;
            [~, ~, Curve_v1, Sf_forest1, ~] = CBSFOASig(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
            t_cbsf1 = toc;
            support_forest1 = false(1, nD); support_forest1(Sf_forest1) = true;
            cont_cbsfoa_forest  = cont_cbsfoa_forest + support_forest1;
            t_cbsf1_run         = t_cbsf1_run + t_cbsf1;
            curvas_v1_total(run, i,:) = Curve_v1;
            fprintf('%.2f s\n', t_cbsf1);
            
            %% ── CBSFOA v2 ───────────────────────────────────────────────
            fprintf('    > CBSFOA v2 (Mini-Forest + Chaos explore)... ');
            tic;
            [~, ~, Curve_v2, Sf_forest2, ~] = CBSFOASig2(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
            t_cbsf2 = toc;
            support_forest2 = false(1, nD); support_forest2(Sf_forest2) = true;
            cont_cbsfoa_forest2  = cont_cbsfoa_forest2 + support_forest2;
            t_cbsf2_run          = t_cbsf2_run + t_cbsf2;
            curvas_v2_total(run, i,:) = Curve_v2;
            fprintf('%.2f s\n', t_cbsf2);
            
            %% ── CBSFOA v3 ───────────────────────────────────────────────
            fprintf('    > CBSFOA v3... ');
            tic;
            [~, ~, Curve_v3, Sf_3, ~] = CBSFOASig3(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
            t_3 = toc;
            support_3 = false(1, nD); support_3(Sf_3) = true;
            cont_cbsfoa_forest3 = cont_cbsfoa_forest3 + support_3;
            t_cbsf3_run = t_cbsf3_run + t_3;
            curvas_v3_total(run, i, :) = Curve_v3;
            fprintf('%.2f s\n', t_3);
            
            %% ── Guardado de seguridad tras cada fold ────────────────────
            ultima_fold = i;
            save(archivo_progreso, 'acum_rfe', 'acum_pi', ...
                 'acum_sfoa_original_total', 'acum_cbsfoa_forest', 'acum_cbsfoa_forest2','acum_cbsfoa_forest3', ...
                 'tiempos_rfe', 'tiempos_pi', ...
                 'tiempos_sfoa_original_total', 'tiempos_cbsfoa_forest', 'tiempos_cbsfoa_forest2', 'tiempos_cbsfoa_forest3', ...
                 'cont_rfe', 'cont_pi', ...
                 'cont_sfoa_original', 'cont_cbsfoa_forest', 'cont_cbsfoa_forest2','cont_cbsfoa_forest3', ...
                 't_rfe_run', 't_pi_run', 't_sfoa_run', ...
                 't_cbsf1_run', 't_cbsf2_run', 't_cbsf3_run', ...
                 'curvas_original_total', 'curvas_v1_total','curvas_v2_total','curvas_v3_total', 'ultima_run', 'ultima_fold');
            fprintf('    [Fold %d asegurado en disco]\n', i);
        catch ME
            fprintf('\n  !! Error en Run %d, Fold %d: %s\n', run, i, ME.message);
            fprintf('     Guardando estado y deteniendo...\n');
            return;
        end
    end % ── fin KFold
    
%% Almacenar resultados del run completo
    acum_rfe(run, :)            = cont_rfe;
    acum_pi(run, :)             = cont_pi;
    acum_sfoa_original_total(run, :) = cont_sfoa_original;
    acum_cbsfoa_forest(run, :)  = cont_cbsfoa_forest;
    acum_cbsfoa_forest2(run, :) = cont_cbsfoa_forest2;
    acum_cbsfoa_forest3(run, :) = cont_cbsfoa_forest3;
    
    tiempos_rfe(run)            = t_rfe_run;
    tiempos_pi(run)             = t_pi_run;
    tiempos_sfoa_original_total(run) = t_sfoa_run;
    tiempos_cbsfoa_forest(run)  = t_cbsf1_run;
    tiempos_cbsfoa_forest2(run) = t_cbsf2_run;
    tiempos_cbsfoa_forest3(run) = t_cbsf3_run;
    
    ultima_run  = run;
    ultima_fold = 0; 
    
    save(archivo_progreso, 'acum_rfe', 'acum_pi', ...
         'acum_sfoa_original_total', 'acum_cbsfoa_forest', 'acum_cbsfoa_forest2', 'acum_cbsfoa_forest3',...
         'tiempos_rfe', 'tiempos_pi', ...
         'tiempos_sfoa_original_total', 'tiempos_cbsfoa_forest', 'tiempos_cbsfoa_forest2','tiempos_cbsfoa_forest3', ...
         'cont_rfe', 'cont_pi', ...
         'cont_sfoa_original', 'cont_cbsfoa_forest', 'cont_cbsfoa_forest2','cont_cbsfoa_forest3', ...
         't_rfe_run', 't_pi_run', 't_sfoa_run', ...
         't_cbsf1_run', 't_cbsf2_run', 't_cbsf3_run', ...
         'curvas_original_total', 'curvas_v1_total','curvas_v2_total','curvas_v3_total', ...
         'ultima_run', 'ultima_fold');
    fprintf('\n  [Run %d completo y guardado]\n', run);
end % ── fin bucle N_runs
%% ══════════════════════════════════════════════════════════════════════════
%%  PROMEDIOS FINALES Y PREPARACIÓN DE DATOS PARA GRÁFICOS
%% ══════════════════════════════════════════════════════════════════════════
media_rfe            = mean(acum_rfe,            1);
media_pi             = mean(acum_pi,             1);
media_sfoa_original  = mean(acum_sfoa_original_total, 1);
media_cbsfoa_forest  = mean(acum_cbsfoa_forest,  1);
media_cbsfoa_forest2 = mean(acum_cbsfoa_forest2, 1);
media_cbsfoa_forest3 = mean(acum_cbsfoa_forest3, 1);

umbral = c.NumTestSets * 0.5;

sel_rfe            = media_rfe            >= umbral;
sel_pi             = media_pi             >= umbral;
sel_sfoa_original  = media_sfoa_original  >= umbral;
sel_cbsfoa_forest  = media_cbsfoa_forest  >= umbral;
sel_cbsfoa_forest2 = media_cbsfoa_forest2 >= umbral;
sel_cbsfoa_forest3 = media_cbsfoa_forest3 >= umbral;

n_sel_rfe          = sum(sel_rfe);
n_sel_pi           = sum(sel_pi);
n_sel_sfoa_original= sum(sel_sfoa_original);
n_sel_cbsf1        = sum(sel_cbsfoa_forest);
n_sel_cbsf2        = sum(sel_cbsfoa_forest2);
n_sel_cbsf3        = sum(sel_cbsfoa_forest3);

% Extraer datos para los gráficos CBSFOA y Original
curva_promedio_or = squeeze(mean(mean(curvas_original_total, 1), 2));
curva_promedio_v1 = squeeze(mean(mean(curvas_v1_total, 1), 2));
curva_promedio_v2 = squeeze(mean(mean(curvas_v2_total, 1), 2));
curva_promedio_v3 = squeeze(mean(mean(curvas_v3_total, 1), 2));

fitness_final_or = reshape(curvas_original_total(:, :, end), 1, []);
fitness_final_v1 = reshape(curvas_v1_total(:, :, end), 1, []);
fitness_final_v2 = reshape(curvas_v2_total(:, :, end), 1, []);
fitness_final_v3 = reshape(curvas_v3_total(:, :, end), 1, []);

% Tiempo promedio dividido entre K_folds exclusivo para la figura 3
t_medios_cbsf = [mean(tiempos_sfoa_original_total), mean(tiempos_cbsfoa_forest), mean(tiempos_cbsfoa_forest2), mean(tiempos_cbsfoa_forest3)] / K_folds;
%% ══════════════════════════════════════════════════════════════════════════
%%  TABLA DE REDUCCIÓN DE CARACTERÍSTICAS (CONSOLA)
%% ══════════════════════════════════════════════════════════════════════════
fprintf('\n\n======================================================\n');
fprintf('       TABLA DE REDUCCIÓN DE CARACTERÍSTICAS\n');
fprintf('======================================================\n');
fprintf('%-25s %10s %10s %10s\n', 'Algoritmo', 'Features', 'Reduccion', 'Reduccion%');
fprintf('------------------------------------------------------\n');
algoritmos = {'RFE', 'Perm. Importance', 'SFOA Original', 'CBSFOA v1', 'CBSFOA v2', 'CBSFOA v3'};
n_seleccionadas = [n_sel_rfe, n_sel_pi, n_sel_sfoa_original, n_sel_cbsf1, n_sel_cbsf2, n_sel_cbsf3];

for k = 1:length(algoritmos)
    reduccion   = n_features - n_seleccionadas(k);
    pct         = (reduccion / n_features) * 100;
    fprintf('%-25s %10d %10d %9.1f%%\n', algoritmos{k}, n_seleccionadas(k), reduccion, pct);
end
fprintf('------------------------------------------------------\n');
fprintf('%-25s %10d\n', 'Total features originales', n_features);
fprintf('======================================================\n\n');

%% ══════════════════════════════════════════════════════════════════════════
%%  TABLA DE TIEMPOS PROMEDIO (CONSOLA)
%% ══════════════════════════════════════════════════════════════════════════
t_medios_global = [mean(tiempos_rfe), mean(tiempos_pi), mean(tiempos_sfoa_original_total), ...
            mean(tiempos_cbsfoa_forest), mean(tiempos_cbsfoa_forest2), mean(tiempos_cbsfoa_forest3)];
            
fprintf('======================================================\n');
fprintf('       TABLA DE TIEMPOS PROMEDIO POR EJECUCIÓN (TOTAL)\n');
fprintf('======================================================\n');
fprintf('%-25s %12s\n', 'Algoritmo', 'Tiempo (s)');
fprintf('------------------------------------------------------\n');
for k = 1:length(algoritmos)
    fprintf('%-25s %12.2f\n', algoritmos{k}, t_medios_global(k));
end
fprintf('======================================================\n\n');

%% ══════════════════════════════════════════════════════════════════════════
%%  GENERACIÓN Y EXPORTACIÓN DE GRÁFICOS (EPS vectoriales)
%% ══════════════════════════════════════════════════════════════════════════
if ~exist(carpeta_imagenes, 'dir')
    mkdir(carpeta_imagenes);
end

% 1. Gráfico de Curvas de Convergencia Promedio
fig1 = figure('Name', 'Convergencia', 'Color', 'w', 'Position', [100 100 700 500]);
plot(1:Max_it, curva_promedio_or, '-d', 'LineWidth', 1.5, 'MarkerIndices', 1:5:Max_it, 'Color', [0.5 0.5 0.5]); hold on;
plot(1:Max_it, curva_promedio_v1, '-o', 'LineWidth', 1.5, 'MarkerIndices', 1:5:Max_it);
plot(1:Max_it, curva_promedio_v2, '-s', 'LineWidth', 1.5, 'MarkerIndices', 1:5:Max_it);
plot(1:Max_it, curva_promedio_v3, '-^', 'LineWidth', 1.5, 'MarkerIndices', 1:5:Max_it);
grid on; box on;
xlabel('Iteraciones', 'FontWeight', 'bold');
ylabel('Fitness (Tasa de Error Global Promedio)', 'FontWeight', 'bold');
legend('SFOA Original', 'CBSFOA v1 (Estándar)', 'CBSFOA v2 (Caos Init+Expl)', 'CBSFOA v3 (Caos Expl)', 'Location', 'best');
title('Curvas de Convergencia Promedio (K-Fold x Runs)', 'FontSize', 12);
print(fig1, fullfile(carpeta_imagenes, 'fig_curva_convergencia_heart.eps'), '-depsc');

% 2. Gráfico de Boxplot del Fitness Final (Robustez / Estabilidad)
fig2 = figure('Name', 'Robustez (Boxplot)', 'Color', 'w', 'Position', [820 100 600 500]);
data_boxplot = [fitness_final_or', fitness_final_v1', fitness_final_v2', fitness_final_v3'];
boxplot(data_boxplot, 'Labels', {'SFOA Org.', 'CBSFOA v1', 'CBSFOA v2', 'CBSFOA v3'});
ylabel('Valor de Fitness Final', 'FontWeight', 'bold');
title('Distribución del Fitness Final (Evaluando Estabilidad)', 'FontSize', 12);
grid on;
print(fig2, fullfile(carpeta_imagenes, 'fig_boxplot_robustez_heart.eps'), '-depsc');

% 3. Gráfico de Barras: Comparación de Tiempos (Costo Computacional por Fold)
fig3 = figure('Name', 'Tiempos de Ejecución', 'Color', 'w', 'Position', [100 650 450 400]);
b = bar(t_medios_cbsf, 'FaceColor', 'flat');
b.CData(1,:) = [0.5 0.5 0.5]; 
b.CData(2,:) = [0.2 0.6 0.8]; 
b.CData(3,:) = [0.8 0.2 0.2];
b.CData(4,:) = [0.2 0.8 0.2];
set(gca, 'XTickLabel', {'Org.', 'v1', 'v2', 'v3'});
ylabel('Tiempo Promedio por Fold (s)', 'FontWeight', 'bold');
title('Costo Computacional (Variantes CBSFOA)', 'FontSize', 12);
grid on;
print(fig3, fullfile(carpeta_imagenes, 'fig_costo_computacional_heart.eps'), '-depsc');

% 4. Gráfico de Barras: Características Seleccionadas (Reducción)
fig4 = figure('Name', 'Seleccion Features', 'Color', 'w', 'Position', [580 650 450 400]);
b2 = bar([n_sel_sfoa_original, n_sel_cbsf1, n_sel_cbsf2, n_sel_cbsf3], 'FaceColor', 'flat');
b2.CData(1,:) = [0.5 0.5 0.5];
b2.CData(2,:) = [0.2 0.6 0.8];
b2.CData(3,:) = [0.8 0.2 0.2];
b2.CData(4,:) = [0.2 0.8 0.2];
set(gca, 'XTickLabel', {'Org.', 'v1', 'v2', 'v3'});
ylabel('Cantidad de Características Retenidas', 'FontWeight', 'bold');
title('Reducción de Dimensionalidad (Variantes)', 'FontSize', 12);
grid on;
print(fig4, fullfile(carpeta_imagenes, 'fig_reduccion_features_heart.eps'), '-depsc');
%% ══════════════════════════════════════════════════════════════════════════
%%  GUARDADO FINAL COMPLETO (.MAT)
%% ══════════════════════════════════════════════════════════════════════════
save('ComparacionFeature_Final_DATA2.mat', ...
     'acum_rfe', 'acum_pi', ...
     'acum_sfoa_original_total', 'acum_cbsfoa_forest', 'acum_cbsfoa_forest2', 'acum_cbsfoa_forest3', ... 
     'media_rfe', 'media_pi', ...
     'media_sfoa_original', 'media_cbsfoa_forest', 'media_cbsfoa_forest2', 'media_cbsfoa_forest3', ... 
     'sel_rfe', 'sel_pi', ...
     'sel_sfoa_original', 'sel_cbsfoa_forest', 'sel_cbsfoa_forest2', 'sel_cbsfoa_forest3', ...
     'tiempos_rfe', 'tiempos_pi', ...
     'tiempos_sfoa_original_total', 'tiempos_cbsfoa_forest', 'tiempos_cbsfoa_forest2','tiempos_cbsfoa_forest3', ...
     'curvas_original_total', 'curvas_v1_total','curvas_v2_total','curvas_v3_total',...
     't_medios_global', 't_medios_cbsf', 'nombre_features', 'n_features', 'N_runs', 'K_folds', 'Max_it', 'Npop');

fprintf('>> Resultados finales guardados en ComparacionFeature_Final_DATA2.mat\n');
fprintf('>> Los gráficos vectoriales .eps se guardaron con éxito en la carpeta "%s".\n', carpeta_imagenes);

%% ── Función local ─────────────────────────────────────────────────────────
function importancias = obtener_importancia_rf(X_train, y_train)
    Mdl = TreeBagger(50, X_train, y_train, 'Method', 'classification', 'OOBPredictorImportance', 'on');
    importancias = Mdl.OOBPermutedPredictorDeltaError;
end
