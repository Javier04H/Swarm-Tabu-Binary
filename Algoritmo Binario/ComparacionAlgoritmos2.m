clc; clear; close all;

% 1. CARGAR LOS RESULTADOS PREVIOS
if exist('resultado_completo_con_bsfoa.mat', 'file')
    load('resultado_completo_con_bsfoa.mat');
    disp('=== 1. DATOS PREVIOS CARGADOS EXITOSAMENTE ===');
else
    error('No se encontró el archivo');
end

% AHORA NORMALIZAMOS
global X_train Y_train X_val Y_val nD
X = zscore(X);
nD = size(X, 2);

% PARTICIÓN GLOBAL (70% Train, 30% Test)
cv = cvpartition(Y, 'HoldOut', 0.3);
X_train = X(cv.training, :);
Y_train = Y(cv.training);
X_val   = X(cv.test, :);
Y_val   = Y(cv.test);

% 2. CONFIGURACIÓN DE BSFOA
Npop = 30;    
Max_it = 50;
lb = -4; ub = 4;

% Reiniciar contadores para ambas versiones de BSFOA

cont_cbsfoa_forest = zeros(1,nD);cont_cbsfoa_forest2 = zeros(1,nD);

disp(' ');
disp('=== 2. EJECUTANDO BSFOA: ÁRBOL SIMPLE vs MINI-FOREST ===');
fprintf('Se ejecutarán %d iteraciones...\n', num_iteraciones);

for iter = 1:num_iteraciones
    fprintf('  > Iteración %d de %d...\n', iter, num_iteraciones);
    [~, ~, ~, Sf_forest1, ~] = CBSFOASig(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
    support_forest1 = false(1, nD);
    support_forest1(Sf_forest1) = true;
    cont_cbsfoa_forest = cont_cbsfoa_forest + support_forest1;

    % --- Ejecución 2: BSFOA con Mini-Forest (@fobj_miniforest) ---
    [~, ~, ~, Sf_forest2, ~] = CBSFOASig2(Npop, Max_it, lb, ub, nD, @fobj_miniforest);
    support_forest2 = false(1, nD);
    support_forest2(Sf_forest2) = true;
    cont_cbsfoa_forest2 = cont_cbsfoa_forest2 + support_forest2;
end

% 3. CÁLCULO DE PORCENTAJES FINALES
disp(' ');
disp('=== 3. COMPARATIVA FINAL ===');

% Calculamos porcentajes de ambos métodos
pct_cbsfoa_forest = (cont_cbsfoa_forest / num_iteraciones) * 100;
pct_cbsfoa_forest2 = (cont_cbsfoa_forest2 / num_iteraciones) * 100;

% Asegurar que existan los otros métodos, sino columnas vacías
if ~exist('pct_rfe', 'var'), pct_rfe = zeros(1,nD); end
if ~exist('pct_boruta', 'var'), pct_boruta = zeros(1,nD); end
if ~exist('pct_pi', 'var'), pct_pi = zeros(1,nD); end
if ~exist('pct_bsfoa_tree', 'var'), pct_bsfoa_tree = zeros(1,nD); end
if ~exist('pct_bsfoa_forest', 'var'), pct_bsfoa_forest = zeros(1,nD); end
if ~exist('nombres_features', 'var'), nombres_features = string(1:nD); end

% Crear la tabla agregando la nueva columna del Mini-Forest
TablaComparativa = table(nombres_features', pct_rfe', pct_boruta', pct_pi', pct_bsfoa_tree', pct_bsfoa_forest',pct_cbsfoa_forest',pct_cbsfoa_forest2', ...
    'VariableNames', {'Caracteristica', 'RFE_Pct', 'Boruta_Pct', 'PermImportance_Pct', 'BSFOA_Tree_Pct', 'BSFOA_Forest_Pct','CBSFOA_F1','CBSFOA_F2'});
%%
% Ordenar basándonos en los resultados del Mini-Forest (o el de Árbol si prefieres)
TablaComparativa = sortrows(TablaComparativa, 'BSFOA_Forest_Pct', 'descend');
disp(TablaComparativa);
%%
% 4. GUARDAR
save('resultado_completo_con_chaosbsfoa.mat');
disp('Resultados guardados exitosamente.');