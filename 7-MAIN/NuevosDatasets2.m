function NuevosDatasets2
clc; close all; clear

% 1. Cargar el dataset base
csv = readtable('heart_limpio.csv');
n_features = width(csv) - 1;

% 2. Cargar las variables desde los nuevos archivos .mat especificados
% Cargar Boruta
load('Resultado_Boruta_Final.mat', 'acum_boruta');

% Cargar el resto de algoritmos (incluyendo la nueva variante CBSFOA v3)
load('ComparacionFeature_Final_DATA2.mat', ...
     'acum_cbsfoa_forest', 'acum_cbsfoa_forest2', 'acum_cbsfoa_forest3', 'acum_pi', 'acum_rfe');

%% ══════════════════════════════════════════════════════════════════════════
%%  PROMEDIOS FINALES Y GENERACIÓN DE VECTORES DE SELECCIÓN
%% ══════════════════════════════════════════════════════════════════════════
% Frecuencia media de selección por feature (promedio a lo largo de las ejecuciones)
media_boruta         = mean(acum_boruta,         1);
media_cbsfoa_forest  = mean(acum_cbsfoa_forest,  1);
media_cbsfoa_forest2 = mean(acum_cbsfoa_forest2, 1);
media_cbsfoa_forest3 = mean(acum_cbsfoa_forest3, 1);
media_pi             = mean(acum_pi,             1);
media_rfe            = mean(acum_rfe,            1);

% Umbral de selección: feature elegida en >= 50% de los folds en promedio
umbral = 10 * 0.5;

sel_boruta         = media_boruta         >= umbral;
sel_cbsfoa_forest  = media_cbsfoa_forest  >= umbral;
sel_cbsfoa_forest2 = media_cbsfoa_forest2 >= umbral;
sel_cbsfoa_forest3 = media_cbsfoa_forest3 >= umbral;
sel_pi             = media_pi             >= umbral;
sel_rfe            = media_rfe            >= umbral;

% Conteo de características seleccionadas por cada método
n_sel_boruta = sum(sel_boruta);
n_sel_cbsf1  = sum(sel_cbsfoa_forest);
n_sel_cbsf2  = sum(sel_cbsfoa_forest2);
n_sel_cbsf3  = sum(sel_cbsfoa_forest3);
n_sel_pi     = sum(sel_pi);
n_sel_rfe    = sum(sel_rfe);

%% ══════════════════════════════════════════════════════════════════════════
%%  TABLA DE REDUCCIÓN DE CARACTERÍSTICAS
%% ══════════════════════════════════════════════════════════════════════════
fprintf('\n\n======================================================\n');
fprintf('       TABLA DE REDUCCIÓN DE CARACTERÍSTICAS\n');
fprintf('======================================================\n');
fprintf('%-25s %10s %10s %10s\n', 'Algoritmo', 'Features', 'Reduccion', 'Reduccion%');
fprintf('------------------------------------------------------\n');

algoritmos = {'Boruta', 'CBSFOA v1', 'CBSFOA v2', 'CBSFOA v3', 'Perm. Importance', 'RFE'};
n_seleccionadas = [n_sel_boruta, n_sel_cbsf1, n_sel_cbsf2, n_sel_cbsf3, n_sel_pi, n_sel_rfe];

for k = 1:length(algoritmos)
    reduccion = n_features - n_seleccionadas(k);
    pct       = (reduccion / n_features) * 100;
    fprintf('%-25s %10d %10d %9.1f%%\n', algoritmos{k}, n_seleccionadas(k), reduccion, pct);
end
fprintf('------------------------------------------------------\n');
fprintf('%-25s %10d\n', 'Total features originales', n_features);
fprintf('======================================================\n\n');

%% ══════════════════════════════════════════════════════════════════════════
%%  TABLA COMPARATIVA BINARIA (OFICIAL MATLAB)
%% ══════════════════════════════════════════════════════════════════════════
% 1. Extraemos los nombres de las características originales
nombres_columnas = csv.Properties.VariableNames(1:end-1);

% 2. Juntamos todos los vectores 'sel_' incluyendo la nueva variante
matriz_binaria = [
    double(sel_boruta);
    double(sel_cbsfoa_forest);
    double(sel_cbsfoa_forest2);
    double(sel_cbsfoa_forest3);
    double(sel_pi);
    double(sel_rfe)
];

% 3. Definimos los nombres de las filas correspondientes
nombres_filas = {'Boruta', 'CBSFOA_Forest_1', 'CBSFOA_Forest_2', 'CBSFOA_Forest_3', 'Perm_Importance_Pi', 'RFE'};

% 4. Creamos la tabla oficial de MATLAB
tabla_comparativa = array2table(matriz_binaria, ...
    'VariableNames', nombres_columnas, ...
    'RowNames', nombres_filas);

% 5. Mostrar la matriz binaria en la consola de comandos
disp(tabla_comparativa);

%% ══════════════════════════════════════════════════════════════════════════
%%  PREPARACIÓN Y FILTRADO DIRECTO (SIN NORMALIZAR)
%% ══════════════════════════════════════════════════════════════════════════
X_tabla = csv(:, 1:end-1); 
y = csv{:, end}; % Vector de etiquetas (clases)

% Convertimos la tabla a una matriz numérica pura (sin alterar sus rangos)
X = table2array(X_tabla);

% Filtrado directo para la RELM utilizando los vectores lógicos obtenidos
X_RELM_boruta  = X(:, sel_boruta);
X_RELM_cbsfoa1 = X(:, sel_cbsfoa_forest);
X_RELM_cbsfoa2 = X(:, sel_cbsfoa_forest2);
X_RELM_cbsfoa3 = X(:, sel_cbsfoa_forest3);
X_RELM_pi      = X(:, sel_pi);
X_RELM_rfe     = X(:, sel_rfe);

% Guardar reporte de comparación en un archivo CSV físico
writetable(tabla_comparativa, 'Comparacionfeature2.csv', 'WriteRowNames', true);

% Guardar los datasets limpios y filtrados en el nuevo archivo solicitado (.mat)
save('csv_limpio2.mat', ...
     'X_RELM_boruta', 'X_RELM_cbsfoa1', 'X_RELM_cbsfoa2', 'X_RELM_cbsfoa3', 'X_RELM_pi', 'X_RELM_rfe', 'y');

end