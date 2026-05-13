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

%% Contadores
cont_rfe    = zeros(1, n_features);
cont_boruta = zeros(1, n_features);
cont_pi     = zeros(1, n_features);
cont_cbsfoa_forest = zeros(1,nD);
cont_cbsfoa_forest2 = zeros(1,nD);


%% Configuración previa para algoritmos
n_features_a_seleccionar = max(1, floor(n_features / 3));
step_rfe = 1;
%BSFOA
Npop = 30;    
Max_it = 50;
lb = -4; ub = 4;


%% KFold
rng(10); % Semilla de randomización para que la partición sea siempre la misma
c = cvpartition(y, 'KFold', 10);

%% Inicio Algoritmos
for i = 1:c.NumTestSets
    fprintf('\nEjecutando Iteración %d de %d\n', i, c.NumTestSets);

    % Extracción de índices
    trainIdx = training(c, i);
    testIdx  = test(c, i);

    % Separación de datos
    X_train = x(trainIdx, :);
    Y_train = y(trainIdx, :);
    X_val  = x(testIdx, :);
    Y_val  = y(testIdx, :);

   %Alias locales por claridad (opcionales)
    X_test = X_val;
    Y_test = Y_val;

    % ── RFE ──────────────────────────────────────────────────────────────
    fprintf('\n> Ejecutando RFE\n');
    fn_importancia_rfe = @(X_sub, y_sub) obtener_importancia_rf(X_sub, y_sub);
    % CORRECCIÓN: usar X_train / Y_train, no X / Y
    [support_rfe, ~] = rfe(X_train, Y_train, fn_importancia_rfe, ...
                           n_features_a_seleccionar, step_rfe);
    cont_rfe = cont_rfe + support_rfe;


    
    % ── Boruta ───────────────────────────────────────────────────────────
    fprintf('\n> Ejecutando Boruta\n');
    % CORRECCIÓN: usar X_train / Y_train, no X / Y
    [decision_boruta, ~] = boruta(X_train, Y_train, 0.01, 50);
    % 1 = Confirmed, 0 = Tentative, -1 = Rejected
    cont_boruta = cont_boruta + (decision_boruta == 1);



    % ── Permutation Importance ───────────────────────────────────────────
    fprintf('\n> Ejecutando Permutation Importance\n');
    % CORRECCIÓN: entrenar el modelo dentro del fold con X_train / Y_train
    modelo_pi = TreeBagger(50, X_train, Y_train, ...
                           'Method', 'classification', ...
                           'OOBPredictorImportance', 'on');

    % Métrica: accuracy (TreeBagger predice celdas de texto → str2double)
    metric_fn = @(y_real, y_pred) ...
        sum(y_real == str2double(y_pred)) / length(y_real);

    % Evaluar PI sobre X_test / Y_test (datos no vistos en el fold)
    res_pi = permutation_importance(modelo_pi, X_test, Y_test, metric_fn, 5);

    % Seleccionar el Top-N con mayor importancia
    [~, idx_ordenado_pi] = sort(res_pi.importances_mean, 'descend');
    support_pi = false(1, n_features);
    support_pi(idx_ordenado_pi(1:n_features_a_seleccionar)) = true;
    cont_pi = cont_pi + support_pi;


    % CBSFOAS
    % --- Ejecución 1: BSFOA con Mini-Forest y Chaos PieceWise en iniciacion (@fobj_miniforest) ---
    [~, ~, ~, Sf_forest1, ~] = CBSFOASig(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
    support_forest1 = false(1, nD);
    support_forest1(Sf_forest1) = true;
    cont_cbsfoa_forest = cont_cbsfoa_forest + support_forest1;

    % --- Ejecución 2: BSFOA con Mini-Forest y Chaos PieceWise en iniciacion y exploracion (@fobj_miniforest) ---
    [~, ~, ~, Sf_forest2, ~] = CBSFOASig2(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
    support_forest2 = false(1, nD);
    support_forest2(Sf_forest2) = true;
    cont_cbsfoa_forest2 = cont_cbsfoa_forest2 + support_forest2;
end

save('ComparacionFeature.mat');

%% ── Función local ────────────────────────────────────────────────────────
function importancias = obtener_importancia_rf(X_train, y_train)
% Entrena un Random Forest de 50 árboles y devuelve la importancia OOB
% de cada variable para alimentar al algoritmo RFE.
    Mdl = TreeBagger(50, X_train, y_train, ...
                     'Method', 'classification', ...
                     'OOBPredictorImportance', 'on');
    importancias = Mdl.OOBPermutedPredictorDeltaError;
end