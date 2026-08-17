%% Limpieza inicial
clc; clear; close all;
global X_train Y_train X_val Y_val nD

%% 1. Carga de datos (Requerido para reconstruir dimensiones)
Dataset2 = readtable("sleep_health_limpio.csv");
x = Dataset2{:, 1:end-1};
y = Dataset2{:, end};

[n_muestras, n_features] = size(x); 
nD = n_features;

%% 2. Carga de resultados previos (CBSFOA v1, v2, v3)
archivo_final = 'Comparacion_CBSFOAs_Final.mat';
if isfile(archivo_final)
    load(archivo_final);
    fprintf('>> Archivo "%s" cargado exitosamente. Datos previos recuperados.\n', archivo_final);
else
    error('No se encontró "%s". Asegúrate de que el archivo final esté en la carpeta.', archivo_final);
end

%% Configuración de parámetros faltantes
lb = -4; ub = 4;
carpeta_imagenes = '99-Imagenes';
if ~exist(carpeta_imagenes, 'dir')
    mkdir(carpeta_imagenes);
end

%% 3. Sistema de guardado provisional para BSFOASig (Algoritmo Base)
archivo_progreso_base = 'progreso_BSFOA_Base.mat';

if isfile(archivo_progreso_base)
    fprintf('>> Archivo de progreso base encontrado. Reanudando...\n');
    load(archivo_progreso_base); 
    % Verificación extra por si la variable no existía en un guardado antiguo
    if ~exist('curvas_bsfoa_total', 'var')
        curvas_bsfoa_total = zeros(N_runs, K_folds, Max_it); 
    end
else
    acum_bsfoa         = zeros(N_runs, n_features);
    tiempos_bsfoa      = zeros(N_runs, 1);
    curvas_bsfoa_total = zeros(N_runs, K_folds, Max_it);
    
    ultima_run_base  = 0;
    ultima_fold_base = 0;
    fprintf('>> Iniciando corrida de BSFOASig (Base) desde cero.\n');
end

%% ══════════════════════════════════════════════════════════════════════════
%%  BUCLE EXTERNO: Evaluación del algoritmo original (BSFOASig)
%% ══════════════════════════════════════════════════════════════════════════
fprintf('\n======================================================\n');
fprintf('  INICIANDO EVALUACIÓN DEL ALGORITMO BASE (BSFOASig)\n');
fprintf('======================================================\n');

for run = (ultima_run_base + 1) : N_runs
    fprintf('\n========================================\n');
    fprintf('  EJECUCIÓN %d de %d (BSFOASig)\n', run, N_runs);
    fprintf('========================================\n');
    
    rng(run * 10);
    c = cvpartition(y, 'KFold', K_folds);
    
    % LÓGICA DE REANUDACIÓN (Igual a MainFinal)
    es_reanudacion = (run == ultima_run_base + 1 && ultima_fold_base > 0);
    if es_reanudacion
        fold_inicio = ultima_fold_base + 1;
        fprintf('   (Reanudando variables parciales desde fold %d)\n', fold_inicio);
    else
        fold_inicio = 1;
        fprintf('   (Iniciando run desde cero)\n');
        
        cont_bsfoa  = zeros(1, n_features);
        t_bsfoa_run = 0;  
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

            %% ── Ejecución de BSFOASig ───────────────────────────────────────
            fprintf('    > BSFOASig... ');
            tic;
            [~, ~, Curve_bsfoa, Sf_bsfoa, ~] = BSFOASig(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
            t_b = toc;
            
            support_b = false(1, nD); 
            support_b(Sf_bsfoa) = true;
            
            cont_bsfoa = cont_bsfoa + support_b;
            t_bsfoa_run = t_bsfoa_run + t_b;
            curvas_bsfoa_total(run, i, :) = Curve_bsfoa;
            
            fprintf('%.2f s\n', t_b);
            
            %% ── Guardado de seguridad tras cada fold ────────────────────
            ultima_fold_base = i;
            save(archivo_progreso_base, ...
                 'acum_bsfoa', 'tiempos_bsfoa', 'curvas_bsfoa_total', ...
                 'cont_bsfoa', 't_bsfoa_run', ...
                 'ultima_run_base', 'ultima_fold_base');
            fprintf('    [Fold %d asegurado en disco]\n', i);
            
        catch ME
            fprintf('\n  !! Error en Run %d, Fold %d: %s\n', run, i, ME.message);
            fprintf('     Guardando estado y deteniendo...\n');
            return;
        end
    end 

    %% Almacenar resultados del run completo para el Base
    acum_bsfoa(run, :) = cont_bsfoa;
    tiempos_bsfoa(run) = t_bsfoa_run;
    
    ultima_run_base  = run;
    ultima_fold_base = 0; 
    
    save(archivo_progreso_base, ...
         'acum_bsfoa', 'tiempos_bsfoa', 'curvas_bsfoa_total', ...
         'ultima_run_base', 'ultima_fold_base');
    fprintf('\n  [Run %d completo y guardado]\n', run);
end 

%% ══════════════════════════════════════════════════════════════════════════
%%  PROCESAMIENTO DE DATOS PARA COMPARACIÓN TOTAL
%% ══════════════════════════════════════════════════════════════════════════
media_bsfoa = mean(acum_bsfoa, 1);
umbral = K_folds * 0.5; 
n_sel_bsfoa = sum(media_bsfoa >= umbral);

t_medio_bsfoa = mean(tiempos_bsfoa);
curva_promedio_bsfoa = squeeze(mean(mean(curvas_bsfoa_total, 1), 2));
fitness_final_bsfoa = reshape(curvas_bsfoa_total(:, :, end), 1, []);

% Consolidar variables para gráficos (Base + v1 + v2 + v3)
t_medios_todos = [t_medio_bsfoa, t_medios(1), t_medios(2), t_medios(3)];
n_sel_todos    = [n_sel_bsfoa, n_sel_1, n_sel_2, n_sel_3];

%% ══════════════════════════════════════════════════════════════════════════
%%  TABLAS DE RESULTADOS POR CONSOLA
%% ══════════════════════════════════════════════════════════════════════════
fprintf('\n\n======================================================\n');
fprintf('   RESUMEN FINAL: BSFOASig vs VARIANTES CBSFOA\n');
fprintf('======================================================\n');
fprintf('%-15s %12s %10s %12s\n', 'Algoritmo', 'Features Sel.', 'Reducc.%', 'Tiempo (s)');
fprintf('------------------------------------------------------\n');
fprintf('%-15s %12d %9.1f%% %12.2f\n', 'BSFOASig (Base)', n_sel_bsfoa, ((n_features-n_sel_bsfoa)/n_features)*100, t_medios_todos(1));
fprintf('%-15s %12d %9.1f%% %12.2f\n', 'CBSFOA v1', n_sel_1, ((n_features-n_sel_1)/n_features)*100, t_medios_todos(2));
fprintf('%-15s %12d %9.1f%% %12.2f\n', 'CBSFOA v2', n_sel_2, ((n_features-n_sel_2)/n_features)*100, t_medios_todos(3));
fprintf('%-15s %12d %9.1f%% %12.2f\n', 'CBSFOA v3', n_sel_3, ((n_features-n_sel_3)/n_features)*100, t_medios_todos(4));
fprintf('------------------------------------------------------\n');
fprintf('Features originales: %d | Folds: %d | Runs: %d\n', n_features, K_folds, N_runs);
fprintf('======================================================\n\n');

%% ══════════════════════════════════════════════════════════════════════════
%%  GENERACIÓN Y EXPORTACIÓN DE GRÁFICOS (Actualizados para 4 algoritmos)
%% ══════════════════════════════════════════════════════════════════════════

% 1. Gráfico de Curvas de Convergencia Promedio
fig1 = figure('Name', 'Convergencia', 'Color', 'w', 'Position', [100 100 700 500]);
plot(1:Max_it, curva_promedio_bsfoa, '-d', 'LineWidth', 1.5, 'MarkerIndices', 1:5:Max_it, 'Color', [0.5 0.5 0.5]); hold on;
plot(1:Max_it, curva_promedio_v1, '-o', 'LineWidth', 1.5, 'MarkerIndices', 1:5:Max_it);
plot(1:Max_it, curva_promedio_v2, '-s', 'LineWidth', 1.5, 'MarkerIndices', 1:5:Max_it);
plot(1:Max_it, curva_promedio_v3, '-^', 'LineWidth', 1.5, 'MarkerIndices', 1:5:Max_it);
grid on; box on;
xlabel('Iteraciones', 'FontWeight', 'bold');
ylabel('Fitness (Tasa de Error Global Promedio)', 'FontWeight', 'bold');
legend('BSFOASig (Base)', 'CBSFOA v1 (Estándar)', 'CBSFOA v2 (Caos Init+Expl)', 'CBSFOA v3 (Caos Expl)', 'Location', 'best');
title('Curvas de Convergencia Promedio (K-Fold x Runs)', 'FontSize', 12);
print(fig1, fullfile(carpeta_imagenes, 'fig_curva_convergencia_completa.eps'), '-depsc');

% 2. Gráfico de Boxplot del Fitness Final (Robustez / Estabilidad)
fig2 = figure('Name', 'Robustez (Boxplot)', 'Color', 'w', 'Position', [820 100 600 500]);
data_boxplot = [fitness_final_bsfoa', fitness_final_v1', fitness_final_v2', fitness_final_v3'];
boxplot(data_boxplot, 'Labels', {'Base', 'CBSFOA v1', 'CBSFOA v2', 'CBSFOA v3'});
ylabel('Valor de Fitness Final', 'FontWeight', 'bold');
title('Distribución del Fitness Final (Evaluando Estabilidad)', 'FontSize', 12);
grid on;
print(fig2, fullfile(carpeta_imagenes, 'fig_boxplot_robustez_completa.eps'), '-depsc');

% 3. Gráfico de Barras: Comparación de Tiempos (Costo Computacional)
fig3 = figure('Name', 'Tiempos de Ejecución', 'Color', 'w', 'Position', [100 650 450 400]);
b = bar(t_medios_todos, 'FaceColor', 'flat');
b.CData(1,:) = [0.5 0.5 0.5]; 
b.CData(2,:) = [0.2 0.6 0.8]; 
b.CData(3,:) = [0.8 0.2 0.2];
b.CData(4,:) = [0.2 0.8 0.2];
set(gca, 'XTickLabel', {'Base', 'v1', 'v2', 'v3'});
ylabel('Tiempo Total Promedio por Ejecución (s)', 'FontWeight', 'bold');
title('Costo Computacional', 'FontSize', 12);
grid on;
print(fig3, fullfile(carpeta_imagenes, 'fig_costo_computacional_completo.eps'), '-depsc');

% 4. Gráfico de Barras: Características Seleccionadas (Reducción)
fig4 = figure('Name', 'Seleccion Features', 'Color', 'w', 'Position', [580 650 450 400]);
b2 = bar(n_sel_todos, 'FaceColor', 'flat');
b2.CData(1,:) = [0.5 0.5 0.5];
b2.CData(2,:) = [0.2 0.6 0.8];
b2.CData(3,:) = [0.8 0.2 0.2];
b2.CData(4,:) = [0.2 0.8 0.2];
set(gca, 'XTickLabel', {'Base', 'v1', 'v2', 'v3'});
ylabel('Cantidad de Características Retenidas', 'FontWeight', 'bold');
title('Reducción de Dimensionalidad', 'FontSize', 12);
grid on;
print(fig4, fullfile(carpeta_imagenes, 'fig_reduccion_features_completa.eps'), '-depsc');

%% Guardado del consolidado Final (4 Algoritmos)
save('Comparacion_Total_4_Algoritmos.mat', ...
     'acum_bsfoa', 'media_bsfoa', 'n_sel_bsfoa', 'tiempos_bsfoa', 't_medio_bsfoa', ...
     'curvas_bsfoa_total', 'curva_promedio_bsfoa', 'fitness_final_bsfoa', ...
     't_medios_todos', 'n_sel_todos', ...
     'acum_cbsf1', 'acum_cbsf2', 'acum_cbsf3', ...
     'media_cbsf1', 'media_cbsf2', 'media_cbsf3', ...
     'n_sel_1', 'n_sel_2', 'n_sel_3', ...
     'tiempos_cbsf1', 'tiempos_cbsf2', 'tiempos_cbsf3', ...
     'curvas_v1_total', 'curvas_v2_total', 'curvas_v3_total', ...
     'curva_promedio_v1', 'curva_promedio_v2', 'curva_promedio_v3', ...
     'fitness_final_v1', 'fitness_final_v2', 'fitness_final_v3', ...
     'nombre_features', 'n_features', 'N_runs', 'K_folds', 'Max_it', 'Npop');

fprintf('>> Resultados de los 4 algoritmos guardados en "Comparacion_Total_4_Algoritmos.mat"\n');
fprintf('>> Gráficos vectoriales completos guardados en la carpeta "%s".\n', carpeta_imagenes);