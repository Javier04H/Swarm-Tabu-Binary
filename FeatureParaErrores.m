clc; clear; close all;

global X_train Y_train X_val Y_val nD


Dataset = readtable ("sleep_health_limpio.csv");

%carga datos
x = Dataset{:,1:end -1};
y = Dataset{:,end};
nombre_features = Dataset.Properties.VariableNames(1:end-1);
[n_muestras, n_features] = size(x);
nD = n_features;

%% Configuración previa para algoritmos
n_features_a_seleccionar = max(1, floor(n_features / 3));
step_rfe = 1;
% BSFOA
Npop   = 30;
Max_it = 50;
lb = -4; ub = 4;

N_runs= 1;
%
acum_rfe            = zeros(N_runs, n_features);
acum_boruta         = zeros(N_runs, n_features);
acum_pi             = zeros(N_runs, n_features);
acum_cbsfoa_forest  = zeros(N_runs, n_features);
acum_cbsfoa_forest2 = zeros(N_runs, n_features);

%unica run, acá se multiplica la run que se desea generar por 10 ene este
%caso la run 5
rng(50);
c = cvpartition(y, 'KFold',10);

% Contadores internos de este run (se acumulan fold a fold)
cont_rfe            = zeros(1, n_features);
cont_boruta         = zeros(1, n_features);
cont_pi             = zeros(1, n_features);
cont_cbsfoa_forest  = zeros(1, nD);
cont_cbsfoa_forest2 = zeros(1, nD);


for i = 1: c.NumTestSets
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
            fn_importancia_rfe = @(X_sub, y_sub) obtener_importancia_rf(X_sub, y_sub);
            [support_rfe, ~] = rfe(X_train, Y_train, fn_importancia_rfe, ...
                                   n_features_a_seleccionar, step_rfe);

            cont_rfe   = cont_rfe + support_rfe;

                        %% ── Permutation Importance ──────────────────────────────────
            fprintf('    > Permutation Importance... ');

            modelo_pi = TreeBagger(50, X_train, Y_train, ...
                                   'Method', 'classification', ...
                                   'OOBPredictorImportance', 'on');
            metric_fn = @(y_real, y_pred) ...
                sum(y_real == str2double(y_pred)) / length(y_real);
            res_pi = permutation_importance(modelo_pi, X_test, Y_test, metric_fn, 5);
            [~, idx_ordenado_pi] = sort(res_pi.importances_mean, 'descend');
            support_pi = false(1, n_features);
            support_pi(idx_ordenado_pi(1:n_features_a_seleccionar)) = true;
            cont_pi   = cont_pi + support_pi;

                        %% ── Boruta ──────────────────────────────────────────────────
            fprintf('    > Boruta... ');
            [decision_boruta, ~] = boruta(X_train, Y_train, 0.01, 50);
            cont_boruta   = cont_boruta + (decision_boruta == 1);

                        %% ── CBSFOA v1 ───────────────────────────────────────────────
            fprintf('    > CBSFOA v1 (Mini-Forest)... ');

            [~, ~, ~, Sf_forest1, ~] = CBSFOASig(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
         
            support_forest1 = false(1, nD);
            support_forest1(Sf_forest1) = true;
            cont_cbsfoa_forest  = cont_cbsfoa_forest + support_forest1;

            %% ── CBSFOA v2 ───────────────────────────────────────────────
            fprintf('    > CBSFOA v2 (Mini-Forest + Chaos explore)... ');
            [~, ~, ~, Sf_forest2, ~] = CBSFOASig2(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
            support_forest2 = false(1, nD);
            support_forest2(Sf_forest2) = true;
            cont_cbsfoa_forest2  = cont_cbsfoa_forest2 + support_forest2;
end

    acum_rfe(1, :)            = cont_rfe;
    acum_boruta(1, :)         = cont_boruta;
    acum_pi(1, :)             = cont_pi;
    acum_cbsfoa_forest(1, :)  = cont_cbsfoa_forest;
    acum_cbsfoa_forest2(1, :) = cont_cbsfoa_forest2;

    save ComparacionReintentoRun5.mat

function importancias = obtener_importancia_rf(X_train, y_train)
% Entrena un Random Forest de 50 árboles y devuelve la importancia OOB
% de cada variable para alimentar al algoritmo RFE.
    Mdl = TreeBagger(50, X_train, y_train, ...
                     'Method', 'classification', ...
                     'OOBPredictorImportance', 'on');
    importancias = Mdl.OOBPermutedPredictorDeltaError;
end