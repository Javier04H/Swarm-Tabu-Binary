%% script: main.m
clc; clear; close all;

% 1. Ejecutar el script de limpieza (crea X, Y y T en el workspace)
disp('=== 1. INICIANDO LIMPIEZA DE DATOS ===');
LimpiezaDatos;

% Extraer los nombres de las características directamente de la tabla
nombres_features = T.Properties.VariableNames(1:end-1);
[n_muestras, n_features] = size(X);

% 2. Configuración del experimento
num_iteraciones = 10;
% Hacemos que RFE y PI seleccionen la mitad de las variables (puedes cambiarlo)
n_features_a_seleccionar = max(1, floor(n_features / 2)); 
step_rfe = 1;

% Contadores para guardar cuántas veces se seleccionó cada característica (inician en 0)
cont_rfe = zeros(1, n_features);
cont_boruta = zeros(1, n_features);
cont_pi = zeros(1, n_features);

disp(' ');
disp('=== 2. INICIANDO CICLO DE SELECCIÓN DE CARACTERÍSTICAS ===');
fprintf('Se ejecutarán %d iteraciones...\n', num_iteraciones);

for iter = 1:num_iteraciones
    fprintf('\n--- Iteración %d de %d ---\n', iter, num_iteraciones);
    
    % ==========================================
    % A. RFE (Recursive Feature Elimination)
    % ==========================================
    fprintf('  > Ejecutando RFE...\n');
    % Usamos un Random Forest rápido (función local al final) para calcular importancias
    fn_importancia_rfe = @(X_sub, y_sub) obtener_importancia_rf(X_sub, y_sub);
    [support_rfe, ~] = rfe(X, Y, fn_importancia_rfe, n_features_a_seleccionar, step_rfe);
    
    % Sumar 1 a las características que sobrevivieron
    cont_rfe = cont_rfe + support_rfe;
    
    % ==========================================
    % B. BORUTA
    % ==========================================
    fprintf('  > Ejecutando Boruta...\n');
    [decision_boruta, ~] = boruta(X, Y, 0.01, 50);
    
    % Boruta devuelve 1 (Confirmed), 0 (Tentative), -1 (Rejected).
    % Sumamos 1 solo a las variables que fueron estrictamente confirmadas.
    cont_boruta = cont_boruta + (decision_boruta == 1);
    
    % ==========================================
    % C. PERMUTATION IMPORTANCE
    % ==========================================
    fprintf('  > Ejecutando Permutation Importance...\n');
    % Entrenar un modelo base (Random Forest de 50 árboles)
    modelo_pi = TreeBagger(50, X, Y, 'Method', 'classification');
    
    % Definir métrica: Accuracy (Precisión). 
    % Nota: TreeBagger devuelve predicciones como celdas de texto (ej. {'2'}), 
    % por eso usamos str2double para compararlo numéricamente con 'Y'.
    metric_fn = @(y_real, y_pred) sum(y_real == str2double(y_pred)) / length(y_real);
    
    % Ejecutar permutación (5 desordenamientos por columna)
    res_pi = permutation_importance(modelo_pi, X, Y, metric_fn, 5);
    
    % PI nos da números puros. Seleccionamos el Top N con mayor importancia
    [~, idx_ordenado_pi] = sort(res_pi.importances_mean, 'descend');
    support_pi = false(1, n_features);
    support_pi(idx_ordenado_pi(1:n_features_a_seleccionar)) = true;
    
    cont_pi = cont_pi + support_pi;
end

% 3. Resultados y cálculo de porcentajes
disp(' ');
disp('=== 3. RESULTADOS FINALES (% de retención en 10 iteraciones) ===');

pct_rfe = (cont_rfe / num_iteraciones) * 100;
pct_boruta = (cont_boruta / num_iteraciones) * 100;
pct_pi = (cont_pi / num_iteraciones) * 100;

% Crear una tabla estructurada para visualización clara
TablaResultados = table(nombres_features', pct_rfe', pct_boruta', pct_pi', ...
    'VariableNames', {'Caracteristica', 'RFE_Pct', 'Boruta_Pct', 'PermImportance_Pct'});

disp(TablaResultados);
%%
save('sleep_workspace.mat')
%%
% =========================================================================
% FUNCIONES LOCALES (Deben ir al final del script main)
% =========================================================================
function importancias = obtener_importancia_rf(X_train, y_train)
    % Entrena un modelo Random Forest rápido de 50 árboles 
    % y extrae la importancia nativa de las variables para alimentar al RFE.
    Mdl = TreeBagger(50, X_train, y_train, 'Method', 'classification', 'OOBPredictorImportance', 'on');
    importancias = Mdl.OOBPermutedPredictorDeltaError;
end


