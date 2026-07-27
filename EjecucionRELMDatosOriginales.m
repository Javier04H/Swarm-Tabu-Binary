clc; close all; clear;

%%% EVALUACIÓN DE DATASETS ORIGINALES MEDIANTE RELM %%%

% 1. Definir los archivos CSV originales a procesar
archivos_csv = {'sleep_health_limpio.csv', 'heart_limpio.csv'};

N_runs = 5; % Cambiar a 5 para la validación final
if ~exist('ultima_run','var')
    ultima_run = 0;
end
elm_type = 1;                  
act_func = 'sig';              
grid_neurons = [10, 50, 100, 200, 500]; 
grid_C = logspace(-5, 5, 6);   
K = 10; 

resultados_finales = struct();

% Bucle externo para procesar ambos archivos .csv
for idx_arch = 1:length(archivos_csv)
    archivo_actual = archivos_csv{idx_arch};
    
    fprintf('\n**************************************************\n');
    fprintf('       PROCESANDO ARCHIVO: %s\n', archivo_actual);
    fprintf('**************************************************\n');
    
    % Asegurarse de que el archivo existe antes de cargarlo
    if ~exist(archivo_actual, 'file')
        fprintf('Advertencia: El archivo %s no existe en el directorio. Saltando...\n', archivo_actual);
        continue;
    end
    
    % 2. Cargar el dataset desde el CSV
    % Leemos como tabla y lo convertimos a matriz numérica
    tabla_datos = readtable(archivo_actual);
    matriz_datos = table2array(tabla_datos);
    
    % Separar X (todas las columnas excepto la última) e Y (última columna)
    X_original = matriz_datos(:, 1:end-1);
    y = matriz_datos(:, end);
    
    % Mapear etiquetas reales a índices 1, 2, ..., N (evita errores en matriz de confusión)
    [~, ~, y] = unique(y);
    num_classes = length(unique(y)); 
    
    % Extraemos el nombre limpio del archivo para guardar los resultados organizados
    [~, nombre_base, ~] = fileparts(archivo_actual);
    
    % Configuramos la evaluación para que corra una sola vez usando el dataset original
    datasets = {X_original};
    dataset_names = {'Original_Completo'};
    num_datasets = length(datasets);
    
    % Bucle interno (en este caso solo iterará 1 vez por CSV)
    for d = 1:num_datasets
        fprintf('\n==================================================\n');
        fprintf('Evaluando Dataset: %s (Archivo: %s)\n', dataset_names{d}, archivo_actual);
        fprintf('==================================================\n');
        
        X_current = datasets{d};
        
        % Matrices 4D: (Neuronas x C x Folds x Runs)
        acc_matrix_4D = zeros(length(grid_neurons), length(grid_C), K, N_runs);
        mcc_matrix_4D = zeros(length(grid_neurons), length(grid_C), K, N_runs);
        gmean_matrix_4D = zeros(length(grid_neurons), length(grid_C), K, N_runs);
        time_matrix_4D = zeros(length(grid_neurons), length(grid_C), K, N_runs);
        
        for run = (ultima_run + 1) : N_runs
            fprintf('--- Iniciando Run %d/%d ---\n', run, N_runs);
            rng(run * 10); % Garantiza particiones y pesos iniciales distintos por run
            c = cvpartition(y, 'KFold', K);
            
            for fold = 1:K
                trainIdx = training(c, fold);
                testIdx  = test(c, fold);
                
                X_train = X_current(trainIdx, :);
                Y_train = y(trainIdx, :);
                X_test  = X_current(testIdx, :);
                Y_test  = y(testIdx, :);
                
                % ══════════════════════════════════════════════════════════
                % NORMALIZACIÓN INTERNA MIN-MAX (RANGO 0-1) POR FOLD
                % ══════════════════════════════════════════════════════════
                % Calculamos los límites usando ÚNICAMENTE los datos de Train
                min_val = min(X_train, [], 1);
                max_val = max(X_train, [], 1);
                rango = max_val - min_val;
                
                % Evitamos división por cero si alguna característica es constante
                rango(rango == 0) = 1; 
                
                % Aplicamos la transformación a Train y Test
                X_train_norm = (X_train - min_val) ./ rango;
                X_test_norm  = (X_test - min_val) ./ rango; 
                % ══════════════════════════════════════════════════════════
                
                for i = 1:length(grid_neurons)
                    Nh = grid_neurons(i);
                    for j = 1:length(grid_C)
                        C_val = grid_C(j);
                        
                        tic;
                        [W, b, Beta] = train_relm(X_train_norm, Y_train, Nh, C_val, act_func);
                        train_time = toc;
                        
                        % Obtener predicciones
                        Y_pred = predict_relm(X_test_norm, W, b, Beta, act_func);
                        
                        % 1. Calcular Accuracy
                        accuracy = sum(Y_pred == Y_test) / length(Y_test);
                        
                        % 2. Calcular MCC y G-Mean utilizando la función local
                        [~, mcc_val, gmean_val] = calc_avanzadas_relm(Y_test, Y_pred, num_classes);
                        
                        % Guardar todo en las matrices 4D
                        acc_matrix_4D(i, j, fold, run) = accuracy;
                        mcc_matrix_4D(i, j, fold, run) = mcc_val;
                        gmean_matrix_4D(i, j, fold, run) = gmean_val;
                        time_matrix_4D(i, j, fold, run) = train_time;
                    end
                end
            end
        end
        
        % --- Procesamiento Estadístico Global ---
        mean_acc = mean(mean(acc_matrix_4D, 3), 4);
        mean_mcc = mean(mean(mcc_matrix_4D, 3), 4);
        mean_gmean = mean(mean(gmean_matrix_4D, 3), 4);
        mean_time = mean(mean(time_matrix_4D, 3), 4);
        
        % Calcular desviación estándar para cada métrica aplanando los resultados
        std_acc = zeros(length(grid_neurons), length(grid_C));
        std_mcc = zeros(length(grid_neurons), length(grid_C));
        std_gmean = zeros(length(grid_neurons), length(grid_C));
        
        for i = 1:length(grid_neurons)
            for j = 1:length(grid_C)
                muestras_acc = squeeze(acc_matrix_4D(i, j, :, :));
                muestras_mcc = squeeze(mcc_matrix_4D(i, j, :, :));
                muestras_gmean = squeeze(gmean_matrix_4D(i, j, :, :));
                
                std_acc(i, j) = std(muestras_acc(:));
                std_mcc(i, j) = std(muestras_mcc(:));
                std_gmean(i, j) = std(muestras_gmean(:));
            end
        end
        
        % Buscar el máximo basado en MCC
        [best_mcc, linear_idx] = max(mean_mcc(:));
        [best_n_idx, best_c_idx] = ind2sub(size(mean_mcc), linear_idx);
        
        best_Nh = grid_neurons(best_n_idx);
        best_C = grid_C(best_c_idx);
        
        best_acc = mean_acc(best_n_idx, best_c_idx);
        best_gmean = mean_gmean(best_n_idx, best_c_idx);
        
        fprintf('\n-> RESULTADOS GLOBALES PARA %s (%d Runs, %d-Fold CV):\n', dataset_names{d}, N_runs, K);
        fprintf('Mejor combinación (Maximizando MCC): Neuronas = %d, C = %e\n', best_Nh, best_C);
        fprintf('  > MCC Test     (Media ± Std): %.4f ± %.4f\n', best_mcc, std_mcc(best_n_idx, best_c_idx));
        fprintf('  > G-Mean Test  (Media ± Std): %.4f ± %.4f\n', best_gmean, std_gmean(best_n_idx, best_c_idx));
        fprintf('  > Accuracy Test(Media ± Std): %.4f ± %.4f\n', best_acc, std_acc(best_n_idx, best_c_idx));
        
        % Guardar en estructura organizando por Archivo -> Algoritmo (Original)
        resultados_finales.(nombre_base).(dataset_names{d}).mean_acc = mean_acc;
        resultados_finales.(nombre_base).(dataset_names{d}).mean_mcc = mean_mcc;
        resultados_finales.(nombre_base).(dataset_names{d}).mean_gmean = mean_gmean;
        resultados_finales.(nombre_base).(dataset_names{d}).best_params = [best_Nh, best_C];
    end
end

%% ================= FUNCIONES NÚCLEO RELM ================= %%
function [W, b, Beta] = train_relm(X, Y, Nh, C, act_func)
    Num_Samples = size(X, 1);
    Num_Features = size(X, 2);
    
    W = rand(Num_Features, Nh) * 2 - 1;
    b = rand(1, Nh);
    tempH = (X * W) + repmat(b, Num_Samples, 1);
    
    switch lower(act_func)
        case {'sig', 'sigmoid'}
            H = 1 ./ (1 + exp(-tempH));
        case {'sin', 'sine'}
            H = sin(tempH);
        case {'hardlim'}
            H = double(hardlim(tempH));
    end
    
    labels = unique(Y);
    num_classes = length(labels);
    T = zeros(Num_Samples, num_classes);
    for i = 1:num_classes
        T(Y == labels(i), i) = 1;
    end
    T = T * 2 - 1; 
    
    I = eye(size(H, 2));
    Beta = (H' * H + I/C) \ (H' * T);
end

function Y_pred = predict_relm(X, W, b, Beta, act_func)
    Num_Samples = size(X, 1);
    tempH = (X * W) + repmat(b, Num_Samples, 1);
    
    switch lower(act_func)
        case {'sig', 'sigmoid'}
            H = 1 ./ (1 + exp(-tempH));
        case {'sin', 'sine'}
            H = sin(tempH);
        case {'hardlim'}
            H = double(hardlim(tempH));
    end
    
    TY = H * Beta;
    [~, max_idx] = max(TY, [], 2);
    Y_pred = max_idx; 
end

function [macro_F1, MCC, G_mean] = calc_avanzadas_relm(Y_true, Y_pred, num_classes)
    C_mat = confusionmat(Y_true, Y_pred, 'Order', 1:num_classes);
    
    precision = zeros(num_classes, 1);
    recall = zeros(num_classes, 1);
    f1_class = zeros(num_classes, 1);
    
    for i = 1:num_classes
        TP = C_mat(i,i);
        FP = sum(C_mat(:,i)) - TP;
        FN = sum(C_mat(i,:)) - TP;
        
        if (TP + FP) == 0; precision(i) = 0; else; precision(i) = TP / (TP + FP); end
        if (TP + FN) == 0; recall(i) = 0; else; recall(i) = TP / (TP + FN); end
        
        if (precision(i) + recall(i)) == 0
            f1_class(i) = 0;
        else
            f1_class(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
        end
    end
    
    macro_F1 = mean(f1_class);
    G_mean = nthroot(prod(recall), num_classes);
    
    s = sum(C_mat(:));          
    c = sum(diag(C_mat));       
    t = sum(C_mat, 1);          
    p = sum(C_mat, 2)';         
    
    numerador = (c * s) - sum(p .* t);
    denominador = sqrt((s^2 - sum(p.^2)) * (s^2 - sum(t.^2)));
    
    if denominador == 0; MCC = 0; else; MCC = numerador / denominador; end
end