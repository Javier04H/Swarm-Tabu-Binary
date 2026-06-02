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
Npop   = 30;
Max_it = 50;
lb = -4; ub = 4;

%% Parámetros del bucle externo
N_runs = 4;  
K_folds = 10; 

%% Gestión de carpetas y archivos de salida
archivo_progreso = 'Comparacion_CBSFOAs_progreso.mat';
carpeta_imagenes = '99-Imagenes';

% Crear la carpeta para las figuras si no existe
if ~exist(carpeta_imagenes, 'dir')
    mkdir(carpeta_imagenes);
    fprintf('>> Carpeta "%s" creada exitosamente.\n', carpeta_imagenes);
end

if isfile(archivo_progreso)
    fprintf('>> Archivo de progreso encontrado. Reanudando...\n');
    load(archivo_progreso); 
else
    acum_cbsf1 = zeros(N_runs, n_features);
    acum_cbsf2 = zeros(N_runs, n_features);
    acum_cbsf3 = zeros(N_runs, n_features);

    tiempos_cbsf1 = zeros(N_runs, 1);
    tiempos_cbsf2 = zeros(N_runs, 1);
    tiempos_cbsf3 = zeros(N_runs, 1);

    curvas_v1_total = zeros(N_runs, K_folds, Max_it);
    curvas_v2_total = zeros(N_runs, K_folds, Max_it);
    curvas_v3_total = zeros(N_runs, K_folds, Max_it);

    ultima_run  = 0;
    ultima_fold = 0;
    fprintf('>> Iniciando experimentos desde cero.\n');
end

%% ══════════════════════════════════════════════════════════════════════════
%%  BUCLE EXTERNO: N ejecuciones independientes
%% ══════════════════════════════════════════════════════════════════════════
for run = (ultima_run + 1) : N_runs
    fprintf('\n========================================\n');
    fprintf('  EJECUCIÓN %d de %d\n', run, N_runs);
    fprintf('========================================\n');
    
    rng(run * 10);
    c = cvpartition(y, 'KFold', K_folds);

    es_reanudacion = (run == ultima_run + 1 && ultima_fold > 0);

    if es_reanudacion
        fold_inicio = ultima_fold + 1;
        fprintf('   (Reanudando variables parciales desde fold %d)\n', fold_inicio);
    else
        fold_inicio = 1;
        fprintf('   (Iniciando run desde cero)\n');
        
        cont_cbsf1 = zeros(1, n_features);
        cont_cbsf2 = zeros(1, n_features);
        cont_cbsf3 = zeros(1, n_features);

        t_cbsf1_run = 0;  
        t_cbsf2_run = 0;
        t_cbsf3_run = 0;
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

            %% ── CBSFOA v1 ───────────────────────────────────────────────
            fprintf('    > CBSFOA v1... ');
            tic;
            [~, ~, Curve_v1, Sf_1, ~] = CBSFOASig(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
            t_1 = toc;
            
            support_1 = false(1, nD); support_1(Sf_1) = true;
            cont_cbsf1 = cont_cbsf1 + support_1;
            t_cbsf1_run = t_cbsf1_run + t_1;
            curvas_v1_total(run, i, :) = Curve_v1;
            fprintf('%.2f s\n', t_1);

            %% ── CBSFOA v2 ───────────────────────────────────────────────
            fprintf('    > CBSFOA v2... ');
            tic;
            [~, ~, Curve_v2, Sf_2, ~] = CBSFOASig2(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
            t_2 = toc;
            
            support_2 = false(1, nD); support_2(Sf_2) = true;
            cont_cbsf2 = cont_cbsf2 + support_2;
            t_cbsf2_run = t_cbsf2_run + t_2;
            curvas_v2_total(run, i, :) = Curve_v2;
            fprintf('%.2f s\n', t_2);

            %% ── CBSFOA v3 ───────────────────────────────────────────────
            fprintf('    > CBSFOA v3... ');
            tic;
            [~, ~, Curve_v3, Sf_3, ~] = CBSFOASig3(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
            t_3 = toc;
            
            support_3 = false(1, nD); support_3(Sf_3) = true;
            cont_cbsf3 = cont_cbsf3 + support_3;
            t_cbsf3_run = t_cbsf3_run + t_3;
            curvas_v3_total(run, i, :) = Curve_v3;
            fprintf('%.2f s\n', t_3);

            %% ── Guardado de seguridad tras cada fold ────────────────────
            ultima_fold = i;
            save(archivo_progreso, ...
                 'acum_cbsf1', 'acum_cbsf2', 'acum_cbsf3', ...
                 'tiempos_cbsf1', 'tiempos_cbsf2', 'tiempos_cbsf3', ...
                 'cont_cbsf1', 'cont_cbsf2', 'cont_cbsf3', ...
                 't_cbsf1_run', 't_cbsf2_run', 't_cbsf3_run', ...
                 'curvas_v1_total', 'curvas_v2_total', 'curvas_v3_total', ...
                 'ultima_run', 'ultima_fold');
            fprintf('    [Fold %d asegurado en disco]\n', i);

        catch ME
            fprintf('\n  !! Error en Run %d, Fold %d: %s\n', run, i, ME.message);
            fprintf('     Guardando estado y deteniendo...\n');
            return;
        end
    end 

    %% Almacenar resultados del run completo
    acum_cbsf1(run, :) = cont_cbsf1;
    acum_cbsf2(run, :) = cont_cbsf2;
    acum_cbsf3(run, :) = cont_cbsf3;

    tiempos_cbsf1(run) = t_cbsf1_run;
    tiempos_cbsf2(run) = t_cbsf2_run;
    tiempos_cbsf3(run) = t_cbsf3_run;

    ultima_run  = run;
    ultima_fold = 0;  

    save(archivo_progreso, ...
         'acum_cbsf1', 'acum_cbsf2', 'acum_cbsf3', ...
         'tiempos_cbsf1', 'tiempos_cbsf2', 'tiempos_cbsf3', ...
         'curvas_v1_total', 'curvas_v2_total', 'curvas_v3_total', ...
         'ultima_run', 'ultima_fold');
    fprintf('\n  [Run %d completo y guardado]\n', run);

end 

%% ══════════════════════════════════════════════════════════════════════════
%%  PROCESAMIENTO DE DATOS PARA EL PAPER
%% ══════════════════════════════════════════════════════════════════════════
media_cbsf1 = mean(acum_cbsf1, 1);
media_cbsf2 = mean(acum_cbsf2, 1);
media_cbsf3 = mean(acum_cbsf3, 1);

umbral = K_folds * 0.5; 
n_sel_1 = sum(media_cbsf1 >= umbral);
n_sel_2 = sum(media_cbsf2 >= umbral);
n_sel_3 = sum(media_cbsf3 >= umbral);

t_medios = [mean(tiempos_cbsf1), mean(tiempos_cbsf2), mean(tiempos_cbsf3)];

curva_promedio_v1 = squeeze(mean(mean(curvas_v1_total, 1), 2));
curva_promedio_v2 = squeeze(mean(mean(curvas_v2_total, 1), 2));
curva_promedio_v3 = squeeze(mean(mean(curvas_v3_total, 1), 2));

fitness_final_v1 = reshape(curvas_v1_total(:, :, end), 1, []);
fitness_final_v2 = reshape(curvas_v2_total(:, :, end), 1, []);
fitness_final_v3 = reshape(curvas_v3_total(:, :, end), 1, []);

%% ══════════════════════════════════════════════════════════════════════════
%%  TABLAS DE RESULTADOS POR CONSOLA
%% ══════════════════════════════════════════════════════════════════════════
fprintf('\n\n======================================================\n');
fprintf('   RESUMEN FINAL PARA COMPARACIÓN DE VARIANTES CBSFOA\n');
fprintf('======================================================\n');
fprintf('%-15s %12s %10s %12s\n', 'Variante', 'Features Sel.', 'Reducc.%', 'Tiempo (s)');
fprintf('------------------------------------------------------\n');
fprintf('%-15s %12d %9.1f%% %12.2f\n', 'CBSFOA v1', n_sel_1, ((n_features-n_sel_1)/n_features)*100, t_medios(1));
fprintf('%-15s %12d %9.1f%% %12.2f\n', 'CBSFOA v2', n_sel_2, ((n_features-n_sel_2)/n_features)*100, t_medios(2));
fprintf('%-15s %12d %9.1f%% %12.2f\n', 'CBSFOA v3', n_sel_3, ((n_features-n_sel_3)/n_features)*100, t_medios(3));
fprintf('------------------------------------------------------\n');
fprintf('Features originales: %d | Folds: %d | Runs: %d\n', n_features, K_folds, N_runs);
fprintf('======================================================\n\n');

%% ══════════════════════════════════════════════════════════════════════════
%%  GENERACIÓN Y EXPORTACIÓN DE GRÁFICOS (EPS vectoriales)
%% ══════════════════════════════════════════════════════════════════════════

% 1. Gráfico de Curvas de Convergencia Promedio
fig1 = figure('Name', 'Convergencia', 'Color', 'w', 'Position', [100 100 700 500]);
plot(1:Max_it, curva_promedio_v1, '-o', 'LineWidth', 1.5, 'MarkerIndices', 1:5:Max_it); hold on;
plot(1:Max_it, curva_promedio_v2, '-s', 'LineWidth', 1.5, 'MarkerIndices', 1:5:Max_it);
plot(1:Max_it, curva_promedio_v3, '-^', 'LineWidth', 1.5, 'MarkerIndices', 1:5:Max_it);
grid on; box on;
xlabel('Iteraciones', 'FontWeight', 'bold');
ylabel('Fitness (Tasa de Error Global Promedio)', 'FontWeight', 'bold');
legend('CBSFOA v1 (Estándar)', 'CBSFOA v2 (Caos Init+Expl)', 'CBSFOA v3 (Caos Expl)', 'Location', 'best');
title('Curvas de Convergencia Promedio (K-Fold x Runs)', 'FontSize', 12);
print(fig1, fullfile(carpeta_imagenes, 'fig_curva_convergencia.eps'), '-depsc');

% 2. Gráfico de Boxplot del Fitness Final (Robustez / Estabilidad)
fig2 = figure('Name', 'Robustez (Boxplot)', 'Color', 'w', 'Position', [820 100 600 500]);
data_boxplot = [fitness_final_v1', fitness_final_v2', fitness_final_v3'];
boxplot(data_boxplot, 'Labels', {'CBSFOA v1', 'CBSFOA v2', 'CBSFOA v3'});
ylabel('Valor de Fitness Final', 'FontWeight', 'bold');
title('Distribución del Fitness Final (Evaluando Estabilidad)', 'FontSize', 12);
grid on;
print(fig2, fullfile(carpeta_imagenes, 'fig_boxplot_robustez.eps'), '-depsc');

% 3. Gráfico de Barras: Comparación de Tiempos (Costo Computacional)
fig3 = figure('Name', 'Tiempos de Ejecución', 'Color', 'w', 'Position', [100 650 450 400]);
b = bar(t_medios, 'FaceColor', 'flat');
b.CData(1,:) = [0.2 0.6 0.8]; 
b.CData(2,:) = [0.8 0.2 0.2];
b.CData(3,:) = [0.2 0.8 0.2];
set(gca, 'XTickLabel', {'v1', 'v2', 'v3'});
ylabel('Tiempo Total Promedio por Ejecución (s)', 'FontWeight', 'bold');
title('Costo Computacional', 'FontSize', 12);
grid on;
print(fig3, fullfile(carpeta_imagenes, 'fig_costo_computacional.eps'), '-depsc');

% 4. Gráfico de Barras: Características Seleccionadas (Reducción)
fig4 = figure('Name', 'Seleccion Features', 'Color', 'w', 'Position', [580 650 450 400]);
b2 = bar([n_sel_1, n_sel_2, n_sel_3], 'FaceColor', 'flat');
b2.CData(1,:) = [0.2 0.6 0.8];
b2.CData(2,:) = [0.8 0.2 0.2];
b2.CData(3,:) = [0.2 0.8 0.2];
set(gca, 'XTickLabel', {'v1', 'v2', 'v3'});
ylabel('Cantidad de Características Retenidas', 'FontWeight', 'bold');
title('Reducción de Dimensionalidad', 'FontSize', 12);
grid on;
print(fig4, fullfile(carpeta_imagenes, 'fig_reduccion_features.eps'), '-depsc');

%% Guardado final completo en formato consolidado
save('Comparacion_CBSFOAs_Final.mat', ...
     'acum_cbsf1', 'acum_cbsf2', 'acum_cbsf3', ...
     'media_cbsf1', 'media_cbsf2', 'media_cbsf3', ...
     'n_sel_1', 'n_sel_2', 'n_sel_3', ...
     'tiempos_cbsf1', 'tiempos_cbsf2', 'tiempos_cbsf3', ...
     't_medios', ...
     'curvas_v1_total', 'curvas_v2_total', 'curvas_v3_total', ...
     'curva_promedio_v1', 'curva_promedio_v2', 'curva_promedio_v3', ...
     'fitness_final_v1', 'fitness_final_v2', 'fitness_final_v3', ...
     'nombre_features', 'n_features', 'N_runs', 'K_folds', 'Max_it', 'Npop');

fprintf('>> Resultados y datos guardados en Comparacion_CBSFOAs_Final.mat\n');
fprintf('>> Los 4 gráficos vectoriales .eps se guardaron con éxito en la carpeta "%s".\n', carpeta_imagenes);