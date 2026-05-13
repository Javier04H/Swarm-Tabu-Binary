%% Limpieza inicial
clc; clear; close all;
global X_train Y_train X_val Y_val nD

%% Carga de datos
Dataset2 = readtable("sleep_health_limpio.csv");
x = Dataset2{:, 1:end-1};
y = Dataset2{:, end};
nombre_features = Dataset2.Properties.VariableNames(1:end-1);
[n_muestras, n_features] = size(x);
nD = n_features;

%% Configuración previa para algoritmos
n_features_a_seleccionar = max(1, floor(n_features / 3));
step_rfe = 1;
% BSFOA
Npop   = 30;
Max_it = 50;
lb = -4; ub = 4;

%% Parámetros del bucle externo
N_runs = 4;  % Número de ejecuciones completas eran 5  pero ahora serán 4 para test

%% Sistema de guardado / reanudación
archivo_progreso = 'progreso_FeatureMain.mat';

if isfile(archivo_progreso)
    fprintf('>> Archivo de progreso encontrado. Reanudando...\n');
    load(archivo_progreso);   % carga: acum_*, tiempos_*, ultima_run, ultima_fold
else
    % Acumuladores por ejecución (runs × features)
    acum_rfe            = zeros(N_runs, n_features);
    acum_boruta         = zeros(N_runs, n_features);
    acum_pi             = zeros(N_runs, n_features);
    acum_cbsfoa_forest  = zeros(N_runs, n_features);
    acum_cbsfoa_forest2 = zeros(N_runs, n_features);

    % Tiempos totales por algoritmo y ejecución (runs × 1)
    tiempos_rfe            = zeros(N_runs, 1);
    tiempos_boruta         = zeros(N_runs, 1);
    tiempos_pi             = zeros(N_runs, 1);
    tiempos_cbsfoa_forest  = zeros(N_runs, 1);
    tiempos_cbsfoa_forest2 = zeros(N_runs, 1);

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
    if run == 2
        fprintf('  [Saltando la Run 2 de forma segura...]\n');
        continue; % Esto aborta la iteración actual y pasa directamente a la Run 3
    end
    % Semilla distinta por ejecución
    rng(run * 10);
    c = cvpartition(y, 'KFold', 10);

    % Determinamos si estamos reanudando una ejecución a medias
    es_reanudacion = (run == ultima_run + 1 && ultima_fold > 0);

    if es_reanudacion
        % --- MODO REANUDACIÓN ---
        fold_inicio = ultima_fold + 1;
        fprintf('   (Reanudando variables parciales desde fold %d)\n', fold_inicio);
        % valores que se cargaron previamente con el load(archivo_progreso).
        
    else
        % --- MODO RUN NUEVO ---
        fold_inicio = 1;
        fprintf('   (Iniciando run desde cero)\n');
        
        % Solo reseteamos a cero si es un Run completamente nuevo
        cont_rfe            = zeros(1, n_features);
        cont_boruta         = zeros(1, n_features);
        cont_pi             = zeros(1, n_features);
        cont_cbsfoa_forest  = zeros(1, nD);
        cont_cbsfoa_forest2 = zeros(1, nD);

        % También reseteamos los tiempos parciales de la run
        t_rfe_run   = 0;  
        t_boruta_run = 0;  
        t_pi_run = 0;
        t_cbsf1_run = 0;  
        t_cbsf2_run  = 0;
    end

    %% ── BUCLE INTERNO: KFold ────────────────────────────────────────────
    for i = fold_inicio : c.NumTestSets
        fprintf('\n  [Run %d] Ejecutando Fold %d de %d\n', run, i, c.NumTestSets);

        try
            % Extracción de índices
            trainIdx = training(c, i);
            testIdx  = test(c, i);

            % Separación de datos (también actualiza las globales)
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
            [support_rfe, ~] = rfe(X_train, Y_train, fn_importancia_rfe, ...
                                   n_features_a_seleccionar, step_rfe);
            t_rfe = toc;
            cont_rfe   = cont_rfe + support_rfe;
            t_rfe_run  = t_rfe_run + t_rfe;
            fprintf('%.2f s\n', t_rfe);

            %% ── Boruta ──────────────────────────────────────────────────
            fprintf('    > Boruta... ');
            tic;
            [decision_boruta, ~] = boruta(X_train, Y_train, 0.01, 50);
            t_boruta = toc;
            cont_boruta   = cont_boruta + (decision_boruta == 1);
            t_boruta_run  = t_boruta_run + t_boruta;
            fprintf('%.2f s\n', t_boruta);

            %% ── Permutation Importance ──────────────────────────────────
            fprintf('    > Permutation Importance... ');
            tic;
            modelo_pi = TreeBagger(50, X_train, Y_train, ...
                                   'Method', 'classification', ...
                                   'OOBPredictorImportance', 'on');
            metric_fn = @(y_real, y_pred) ...
                sum(y_real == str2double(y_pred)) / length(y_real);
            res_pi = permutation_importance(modelo_pi, X_test, Y_test, metric_fn, 5);
            t_pi = toc;
            [~, idx_ordenado_pi] = sort(res_pi.importances_mean, 'descend');
            support_pi = false(1, n_features);
            support_pi(idx_ordenado_pi(1:n_features_a_seleccionar)) = true;
            cont_pi   = cont_pi + support_pi;
            t_pi_run  = t_pi_run + t_pi;
            fprintf('%.2f s\n', t_pi);

            %% ── CBSFOA v1 ───────────────────────────────────────────────
            fprintf('    > CBSFOA v1 (Mini-Forest)... ');
            tic;
            [~, ~, ~, Sf_forest1, ~] = CBSFOASig(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
            t_cbsf1 = toc;
            support_forest1 = false(1, nD);
            support_forest1(Sf_forest1) = true;
            cont_cbsfoa_forest  = cont_cbsfoa_forest + support_forest1;
            t_cbsf1_run         = t_cbsf1_run + t_cbsf1;
            fprintf('%.2f s\n', t_cbsf1);

            %% ── CBSFOA v2 ───────────────────────────────────────────────
            fprintf('    > CBSFOA v2 (Mini-Forest + Chaos explore)... ');
            tic;
            [~, ~, ~, Sf_forest2, ~] = CBSFOASig2(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
            t_cbsf2 = toc;
            support_forest2 = false(1, nD);
            support_forest2(Sf_forest2) = true;
            cont_cbsfoa_forest2  = cont_cbsfoa_forest2 + support_forest2;
            t_cbsf2_run          = t_cbsf2_run + t_cbsf2;
            fprintf('%.2f s\n', t_cbsf2);

            %% ── Guardado de seguridad tras cada fold ────────────────────
            ultima_fold = i;
            % Guardamos el estado parcial del run actual
            save(archivo_progreso, ...
                 'acum_rfe', 'acum_boruta', 'acum_pi', ...
                 'acum_cbsfoa_forest', 'acum_cbsfoa_forest2', ...
                 'tiempos_rfe', 'tiempos_boruta', 'tiempos_pi', ...
                 'tiempos_cbsfoa_forest', 'tiempos_cbsfoa_forest2', ...
                 'cont_rfe', 'cont_boruta', 'cont_pi', ...
                 'cont_cbsfoa_forest', 'cont_cbsfoa_forest2', ...
                 't_rfe_run', 't_boruta_run', 't_pi_run', ...
                 't_cbsf1_run', 't_cbsf2_run', ...
                 'ultima_run', 'ultima_fold');
            fprintf('    [Fold %d asegurado en disco]\n', i);

        catch ME
            fprintf('\n  !! Error en Run %d, Fold %d: %s\n', run, i, ME.message);
            fprintf('     Guardando estado y deteniendo...\n');
            save(archivo_progreso, ...
                 'acum_rfe', 'acum_boruta', 'acum_pi', ...
                 'acum_cbsfoa_forest', 'acum_cbsfoa_forest2', ...
                 'tiempos_rfe', 'tiempos_boruta', 'tiempos_pi', ...
                 'tiempos_cbsfoa_forest', 'tiempos_cbsfoa_forest2', ...
                 'cont_rfe', 'cont_boruta', 'cont_pi', ...
                 'cont_cbsfoa_forest', 'cont_cbsfoa_forest2', ...
                 't_rfe_run', 't_boruta_run', 't_pi_run', ...
                 't_cbsf1_run', 't_cbsf2_run', ...
                 'ultima_run', 'ultima_fold');
            return;
        end
    end % ── fin KFold

    %% Almacenar resultados del run completo
    acum_rfe(run, :)            = cont_rfe;
    acum_boruta(run, :)         = cont_boruta;
    acum_pi(run, :)             = cont_pi;
    acum_cbsfoa_forest(run, :)  = cont_cbsfoa_forest;
    acum_cbsfoa_forest2(run, :) = cont_cbsfoa_forest2;

    tiempos_rfe(run)            = t_rfe_run;
    tiempos_boruta(run)         = t_boruta_run;
    tiempos_pi(run)             = t_pi_run;
    tiempos_cbsfoa_forest(run)  = t_cbsf1_run;
    tiempos_cbsfoa_forest2(run) = t_cbsf2_run;

    ultima_run  = run;
    ultima_fold = 0;  % reset: el próximo run empieza desde fold 1

    save(archivo_progreso, ...
         'acum_rfe', 'acum_boruta', 'acum_pi', ...
         'acum_cbsfoa_forest', 'acum_cbsfoa_forest2', ...
         'tiempos_rfe', 'tiempos_boruta', 'tiempos_pi', ...
         'tiempos_cbsfoa_forest', 'tiempos_cbsfoa_forest2', ...
         'cont_rfe', 'cont_boruta', 'cont_pi', ...
         'cont_cbsfoa_forest', 'cont_cbsfoa_forest2', ...
         't_rfe_run', 't_boruta_run', 't_pi_run', ...
         't_cbsf1_run', 't_cbsf2_run', ...
         'ultima_run', 'ultima_fold');
    fprintf('\n  [Run %d completo y guardado]\n', run);

end % ── fin bucle N_runs

%% ══════════════════════════════════════════════════════════════════════════
%%  PROMEDIOS FINALES
%% ══════════════════════════════════════════════════════════════════════════

% Frecuencia media de selección por feature (entre 0 y 10 = num folds)
media_rfe            = mean(acum_rfe,            1);
media_boruta         = mean(acum_boruta,         1);
media_pi             = mean(acum_pi,             1);
media_cbsfoa_forest  = mean(acum_cbsfoa_forest,  1);
media_cbsfoa_forest2 = mean(acum_cbsfoa_forest2, 1);

% Umbral de selección: feature elegida en >50% de los folds en promedio
umbral = c.NumTestSets * 0.5;
sel_rfe            = media_rfe            >= umbral;
sel_boruta         = media_boruta         >= umbral;
sel_pi             = media_pi             >= umbral;
sel_cbsfoa_forest  = media_cbsfoa_forest  >= umbral;
sel_cbsfoa_forest2 = media_cbsfoa_forest2 >= umbral;

n_sel_rfe    = sum(sel_rfe);
n_sel_boruta = sum(sel_boruta);
n_sel_pi     = sum(sel_pi);
n_sel_cbsf1  = sum(sel_cbsfoa_forest);
n_sel_cbsf2  = sum(sel_cbsfoa_forest2);

%% ══════════════════════════════════════════════════════════════════════════
%%  TABLA DE REDUCCIÓN DE CARACTERÍSTICAS
%% ══════════════════════════════════════════════════════════════════════════
fprintf('\n\n======================================================\n');
fprintf('       TABLA DE REDUCCIÓN DE CARACTERÍSTICAS\n');
fprintf('======================================================\n');
fprintf('%-25s %10s %10s %10s\n', 'Algoritmo', 'Features', 'Reduccion', 'Reduccion%');
fprintf('------------------------------------------------------\n');

algoritmos   = {'RFE', 'Boruta', 'Perm. Importance', 'CBSFOA v1', 'CBSFOA v2'};
n_seleccionadas = [n_sel_rfe, n_sel_boruta, n_sel_pi, n_sel_cbsf1, n_sel_cbsf2];

for k = 1:length(algoritmos)
    reduccion   = n_features - n_seleccionadas(k);
    pct         = (reduccion / n_features) * 100;
    fprintf('%-25s %10d %10d %9.1f%%\n', algoritmos{k}, n_seleccionadas(k), reduccion, pct);
end
fprintf('------------------------------------------------------\n');
fprintf('%-25s %10d\n', 'Total features originales', n_features);
fprintf('======================================================\n\n');

%% ══════════════════════════════════════════════════════════════════════════
%%  TABLA DE TIEMPOS PROMEDIO
%% ══════════════════════════════════════════════════════════════════════════
t_medios = [mean(tiempos_rfe), mean(tiempos_boruta), mean(tiempos_pi), ...
            mean(tiempos_cbsfoa_forest), mean(tiempos_cbsfoa_forest2)];

fprintf('======================================================\n');
fprintf('       TABLA DE TIEMPOS PROMEDIO POR EJECUCIÓN\n');
fprintf('======================================================\n');
fprintf('%-25s %12s\n', 'Algoritmo', 'Tiempo (s)');
fprintf('------------------------------------------------------\n');
for k = 1:length(algoritmos)
    fprintf('%-25s %12.2f\n', algoritmos{k}, t_medios(k));
end
fprintf('======================================================\n\n');

%% Guardado final completo
save('ComparacionFeature_Final.mat', ...
     'acum_rfe', 'acum_boruta', 'acum_pi', ...
     'acum_cbsfoa_forest', 'acum_cbsfoa_forest2', ...
     'media_rfe', 'media_boruta', 'media_pi', ...
     'media_cbsfoa_forest', 'media_cbsfoa_forest2', ...
     'sel_rfe', 'sel_boruta', 'sel_pi', ...
     'sel_cbsfoa_forest', 'sel_cbsfoa_forest2', ...
     'tiempos_rfe', 'tiempos_boruta', 'tiempos_pi', ...
     'tiempos_cbsfoa_forest', 'tiempos_cbsfoa_forest2', ...
     't_medios', 'nombre_features', 'n_features', 'N_runs');

fprintf('>> Resultados finales guardados en ComparacionFeature_Final.mat\n');

%% ── Función local ─────────────────────────────────────────────────────────
function importancias = obtener_importancia_rf(X_train, y_train)
% Entrena un Random Forest de 50 árboles y devuelve la importancia OOB
% de cada variable para alimentar al algoritmo RFE.
    Mdl = TreeBagger(50, X_train, y_train, ...
                     'Method', 'classification', ...
                     'OOBPredictorImportance', 'on');
    importancias = Mdl.OOBPermutedPredictorDeltaError;
end