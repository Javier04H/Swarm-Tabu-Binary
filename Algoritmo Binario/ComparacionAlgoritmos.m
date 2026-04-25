clc; clear; close all;

% 1. CARGAR LOS RESULTADOS PREVIOS
if exist('sleep_workspace.mat', 'file')
    load('sleep_workspace.mat');
    disp('=== 1. DATOS PREVIOS CARGADOS EXITOSAMENTE ===');
else
    error('No se encontró el archivo sleep_workspace.mat. Asegúrate de haber corrido los otros métodos primero.');
end

% 2. CONFIGURACIÓN DE BSFOA (Ajusta según tu necesidad)
Npop = 10000; 
Max_it = 100;
lb = -4; ub = 4; % Rango para la Tangente Hiperbólica
HO = 0.2;


% Reiniciar contador de BSFOA (por si ya existía en el mat)
cont_bsfoa = zeros(1, n_features);

disp(' ');
disp('=== 2. EJECUTANDO BSFOA (Solo este método) ===');
fprintf('Se ejecutarán %d iteraciones de BSFOA...\n', num_iteraciones);

for iter = 1:num_iteraciones
    fprintf('  > Iteración %d de %d...\n', iter, num_iteraciones);
    
    % Ejecutar el algoritmo binarizado
    [~, ~, ~, Sf, ~] = BSFOASig(Npop, Max_it, lb, ub, n_features, @fobj);
    
    % Actualizar contador de BSFOA
    support_bsfoa = false(1, n_features);
    support_bsfoa(Sf) = true;
    cont_bsfoa = cont_bsfoa + support_bsfoa;
end

% 3. CÁLCULO DE PORCENTAJES FINALES
disp(' ');
disp('=== 3. COMPARATIVA FINAL ===');

% Calculamos el porcentaje de BSFOA
pct_bsfoa = (cont_bsfoa / num_iteraciones) * 100;

% Los otros porcentajes (pct_rfe, pct_boruta, pct_pi) ya están en el Workspace
% gracias al comando 'load' del principio.

% Crear Tabla Comparativa Final
TablaComparativa = table(nombres_features', pct_rfe', pct_boruta', pct_pi', pct_bsfoa', ...
    'VariableNames', {'Caracteristica', 'RFE_Pct', 'Boruta_Pct', 'PermImportance_Pct', 'BSFOA_Pct'});

% Ordenar por la característica que más seleccionó BSFOA para ver relevancia
TablaComparativa = sortrows(TablaComparativa, 'BSFOA_Pct', 'descend');

disp(TablaComparativa);

% 4. GUARDAR TODO ACTUALIZADO
save('resultado_completo_con_bsfoa.mat');
disp('Resultados guardados en resultado_completo_con_bsfoa.mat');
disp(s)