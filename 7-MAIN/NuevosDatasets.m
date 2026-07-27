function NuevosDatasets
clc; close all; clear

csv = readtable('sleep_health_limpio.csv');

% Inicializamos fijando el tamaño estricto en 5x30
columBoruta  = zeros(5, 30);
columCBSFOA1 = zeros(5, 30);
columCBSFOA2 = zeros(5, 30);
columCBSFOA3 = zeros(5, 30);
columPi      = zeros(5, 30);
columRFE     = zeros(5, 30);

for i = 1:5
    if i == 2
      load("Intento1_fallido.mat", 'acum_boruta', 'acum_cbsfoa_forest', 'acum_cbsfoa_forest2', 'acum_pi', 'acum_rfe');  
        columBoruta(i,:)  = acum_boruta(i, :);
        columCBSFOA1(i,:) = acum_cbsfoa_forest(i, :);
        columCBSFOA2(i,:) = acum_cbsfoa_forest2(i, :);
        columPi(i,:)      = acum_pi(i, :);
        columRFE(i,:)     = acum_rfe(i, :);
        continue
    end
    
    if i == 5
        % CLAVE: Cargamos ÚNICAMENTE las variables 'acum_', NO todo el archivo.
        % Así, si el archivo tiene un 'columBoruta' de 10x30 guardado, MATLAB lo ignorará.
        load("ComparacionReintentoRun5.mat", 'acum_boruta', 'acum_cbsfoa_forest', 'acum_cbsfoa_forest2', 'acum_pi', 'acum_rfe');
        
        columBoruta(i,:)  = acum_boruta(1, :);
        columCBSFOA1(i,:) = acum_cbsfoa_forest(1, :);
        columCBSFOA2(i,:) = acum_cbsfoa_forest2(1, :);
        columPi(i,:)      = acum_pi(1, :);
        columRFE(i,:)     = acum_rfe(1, :);    
    else
        % Lo mismo aquí: cargamos solo los vectores fila que necesitamos
        load("ComparacionFeature_Final.mat", 'acum_boruta', 'acum_cbsfoa_forest', 'acum_cbsfoa_forest2', 'acum_pi', 'acum_rfe'); 
        
        columBoruta(i,:)  = acum_boruta(i, :);
        columCBSFOA1(i,:) = acum_cbsfoa_forest(i, :);
        columCBSFOA2(i,:) = acum_cbsfoa_forest2(i, :);
        columPi(i,:)      = acum_pi(i, :);
        columRFE(i,:)     = acum_rfe(i, :);
    end
end

%% ══════════════════════════════════════════════════════════════════════════
%%  PROMEDIOS FINALES
%% ══════════════════════════════════════════════════════════════════════════
% Frecuencia media de selección por feature (entre 0 y 10 = num folds)
media_rfe            = mean(columRFE,            1);
media_boruta         = mean(columBoruta,         1);
media_pi             = mean(columPi,             1);
media_cbsfoa_forest  = mean(columCBSFOA1,  1);
media_cbsfoa_forest2 = mean(columCBSFOA2, 1);

% Umbral de selección: feature elegida en >50% de los folds en promedio
umbral = 10 * 0.5;

sel_rfe            = media_rfe            >= umbral;
sel_boruta         = media_boruta         >= umbral;
sel_pi             = media_pi             >= umbral;
sel_cbsfoa_forest  = media_cbsfoa_forest  >= umbral;
sel_cbsfoa_forest2 = media_cbsfoa_forest2 >= umbral;

datos = [7, 2.25, 2, 10, 1.5, 10, 1, 7.5, 1, 9.75, 10, 5, 8.5, 1.75, 1.75, 2.5, 1, 9.75, 1.25, 2.25, 10, 1.5, 1.25, 9.5, 0.75, 1.25, 1.25, 1.75, 0.5, 1.25];
% CORRECCIÓN: Dejarlo como operación lógica, sin el double(), para que funcione como índice.
sel_cbsfoa_forest3 = (datos >= 5); 

n_sel_rfe    = sum(sel_rfe);
n_sel_boruta = sum(sel_boruta);
n_sel_pi     = sum(sel_pi);
n_sel_cbsf1  = sum(sel_cbsfoa_forest);
n_sel_cbsf2  = sum(sel_cbsfoa_forest2);
n_sel_cbsf3  = sum(sel_cbsfoa_forest3);

%% ══════════════════════════════════════════════════════════════════════════
%%  TABLA DE REDUCCIÓN DE CARACTERÍSTICAS
%% ══════════════════════════════════════════════════════════════════════════
fprintf('\n\n======================================================\n');
fprintf('       TABLA DE REDUCCIÓN DE CARACTERÍSTICAS\n');
fprintf('======================================================\n');
fprintf('%-25s %10s %10s %10s\n', 'Algoritmo', 'Features', 'Reduccion', 'Reduccion%');
fprintf('------------------------------------------------------\n');

n_features = width(csv) - 1;
algoritmos   = {'RFE', 'Boruta', 'Perm. Importance', 'CBSFOA v1', 'CBSFOA v2', 'CBSFOA v3'};
n_seleccionadas = [n_sel_rfe, n_sel_boruta, n_sel_pi, n_sel_cbsf1, n_sel_cbsf2, n_sel_cbsf3];

for k = 1:length(algoritmos)
    reduccion   = n_features - n_seleccionadas(k);
    pct         = (reduccion / n_features) * 100;
    fprintf('%-25s %10d %10d %9.1f%%\n', algoritmos{k}, n_seleccionadas(k), reduccion, pct);
end
fprintf('------------------------------------------------------\n');
fprintf('%-25s %10d\n', 'Total features originales', n_features);
fprintf('======================================================\n\n')

% 1. Extraemos los nombres de las 30 características originales
nombres_columnas = csv.Properties.VariableNames(1:end-1);

% 2. Juntamos todos los vectores 'sel_' en una sola matriz numérica (6 filas x 30 columnas)
% Usamos 'double' para convertir los 'true/false' lógicos en '1/0' para la tabla visual
matriz_binaria = [
    double(sel_boruta);
    double(sel_cbsfoa_forest);
    double(sel_cbsfoa_forest2);
    double(sel_pi);
    double(sel_rfe);
    double(sel_cbsfoa_forest3);
];

% 3. Definimos los nombres de las filas (los métodos)
nombres_filas = {'Boruta', 'CBSFOA_Forest_1', 'CBSFOA_Forest_2', 'Perm_Importance_Pi', 'RFE','CBSFOA_Forest_3'};

% 4. Creamos la tabla oficial de MATLAB
tabla_comparativa = array2table(matriz_binaria, ...
    'VariableNames', nombres_columnas, ...
    'RowNames', nombres_filas);

% 5. La mostramos en la consola
disp(tabla_comparativa);

% --- 1. PREPARACIÓN GLOBAL (SIN NORMALIZAR) ---
X_tabla = csv(:, 1:end-1); 
y = csv{:, end}; % Tu vector de etiquetas (clases)

% Convertimos a matriz 
X = table2array(X_tabla);

% --- 2. FILTRADO DIRECTO PARA LA RELM (Matrices numéricas) ---
% Usamos los vectores lógicos directamente sobre la matriz original X
X_RELM_boruta   = X(:, sel_boruta);
X_RELM_cbsfoa1  = X(:, sel_cbsfoa_forest);
X_RELM_cbsfoa2  = X(:, sel_cbsfoa_forest2);
X_RELM_pi       = X(:, sel_pi);
X_RELM_rfe      = X(:, sel_rfe);
X_RELM_cbsfoa3  = X(:, sel_cbsfoa_forest3); % Ahora esto funcionará correctamente

writetable(tabla_comparativa, 'Comparacionfeature.csv');
save('CSV_LIMPIOS',"X_RELM_rfe","X_RELM_pi","X_RELM_cbsfoa2","X_RELM_cbsfoa1",'X_RELM_boruta','X_RELM_cbsfoa3',"y")

end