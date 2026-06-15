%% Limpieza inicial
clc; clear; close all;
try
    delete(gcp('nocreate'));
catch
    % Si no hay nada, continúa sin lanzar error
end

if isempty(gcp('nocreate'))
    parpool('local', 8);
end
%% Carga de datos
Dataset2 = readtable("sleep_health_limpio.csv");
x = Dataset2{:, 1:end-1};
y = Dataset2{:, end};
nombre_features = Dataset2.Properties.VariableNames(1:end-1);
[n_muestras, n_features] = size(x);

%% Parámetros del bucle externo y Validación
N_runs = 1;     % Número de ejecuciones completas (4 para test)
num_folds = 10; % KFold original

%% Sistema de guardado / reanudación (Exclusivo Boruta)
archivo_progreso = 'progreso_Boruta.mat';

if isfile(archivo_progreso)
    fprintf('>> Archivo de progreso de Boruta encontrado. Reanudando...\n');
    load(archivo_progreso); % carga: acum_boruta, tiempos_boruta, ultima_run, ultima_fold
else
    % Acumuladores exclusivos para Boruta
    acum_boruta    = zeros(N_runs, n_features);
    tiempos_boruta = zeros(N_runs, 1);

    ultima_run  = 0;
    ultima_fold = 0;
    fprintf('>> Iniciando Boruta desde cero.\n');
end

%% ══════════════════════════════════════════════════════════════════════════
%%  BUCLE EXTERNO: N ejecuciones independientes
%% ══════════════════════════════════════════════════════════════════════════
for run = (ultima_run + 1) : N_runs
    fprintf('\n========================================\n');
    fprintf('  EJECUCIÓN %d de %d (SOLO BORUTA)\n', run, N_runs);
    fprintf('========================================\n');
    
    % Semilla distinta por ejecución
    rng(run * 10);
    c = cvpartition(y, 'KFold', num_folds);

    % Determinamos si estamos reanudando una ejecución a medias
    es_reanudacion = (run == ultima_run + 1 && ultima_fold > 0);

    if es_reanudacion
        % --- MODO REANUDACIÓN ---
        fold_inicio = ultima_fold + 1;
        fprintf('   (Reanudando variables parciales desde fold %d)\n', fold_inicio);
    else
        % --- MODO RUN NUEVO ---
        fold_inicio = 1;
        fprintf('   (Iniciando run desde cero)\n');
        
        cont_boruta  = zeros(1, n_features);
        t_boruta_run = 0;  
    end

    %% ── BUCLE INTERNO: KFold ────────────────────────────────────────────
    for i = fold_inicio : c.NumTestSets
        fprintf('\n  [Run %d] Ejecutando Fold %d de %d\n', run, i, c.NumTestSets);

        try
            % Extracción de índices
            trainIdx = training(c, i);
            X_train  = x(trainIdx, :);
            Y_train  = y(trainIdx, :);

            %% ── Boruta ──────────────────────────────────────────────────
            fprintf('    > Ejecutando Boruta... ');
            tic;
            [decision_boruta, ~] = boruta(X_train, Y_train, 0.01, 100);
            t_boruta = toc;
            
            cont_boruta   = cont_boruta + (decision_boruta == 1);
            t_boruta_run  = t_boruta_run + t_boruta;
            fprintf('%.2f s\n', t_boruta);

            %% ── Guardado de seguridad tras cada fold ────────────────────
            ultima_fold = i;
            save(archivo_progreso, ...
                 'acum_boruta', 'tiempos_boruta', 'cont_boruta', ...
                 't_boruta_run', 'ultima_run', 'ultima_fold');
            fprintf('    [Fold %d asegurado en disco]\n', i);

        catch ME
            fprintf('\n  !! Error en Run %d, Fold %d: %s\n', run, i, ME.message);
            fprintf('     Guardando estado y deteniendo...\n');
            return;
        end
    end % ── fin KFold

    %% Almacenar resultados del run completo
    acum_boruta(run, :) = cont_boruta;
    tiempos_boruta(run) = t_boruta_run;

    ultima_run  = run;
    ultima_fold = 0;  % El próximo run empieza desde fold 1

    save(archivo_progreso, ...
         'acum_boruta', 'tiempos_boruta', 'cont_boruta', ...
         't_boruta_run', 'ultima_run', 'ultima_fold');
    fprintf('\n  [Run %d completo y guardado]\n', run);

end % ── fin bucle N_runs

%% ══════════════════════════════════════════════════════════════════════════
%%  PROMEDIOS Y CORTE FINAL
%% ══════════════════════════════════════════════════════════════════════════

% Frecuencia media de selección por feature (entre 0 y 10)
media_boruta = mean(acum_boruta, 1);

% Umbral de selección: elegido en >50% de los folds en promedio
umbral = num_folds * 0.5;
sel_boruta = media_boruta >= umbral;
n_sel_boruta = sum(sel_boruta);

%% ══════════════════════════════════════════════════════════════════════════
%%  REPORTE DE RESULTADOS (SOLO BORUTA)
%% ══════════════════════════════════════════════════════════════════════════
fprintf('\n\n======================================================\n');
fprintf('       RESUMEN DE REDUCCIÓN DE CARACTERÍSTICAS\n');
fprintf('======================================================\n');
reduccion = n_features - n_sel_boruta;
pct = (reduccion / n_features) * 100;

fprintf('Características Originales : %d\n', n_features);
fprintf('Características Seleccionadas: %d\n', n_sel_boruta);
fprintf('Variables Eliminadas        : %d\n', reduccion);
fprintf('Porcentaje de Reducción     : %.1f%%\n', pct);
fprintf('Tiempo Promedio por Run     : %.2f s\n', mean(tiempos_boruta));
fprintf('======================================================\n\n');

%% Guardado final completo de Boruta
save('Resultado_Boruta_Final.mat', ...
     'acum_boruta', 'media_boruta', 'sel_boruta', ...
     'tiempos_boruta', 'nombre_features', 'n_features', 'N_runs');

fprintf('>> Resultados de Boruta guardados en Resultado_Boruta_Final.mat\n');