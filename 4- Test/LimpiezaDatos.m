%% =============================================
%  LIMPIEZA Y CODIFICACIÓN DEL DATASET
%  sleep_health_dataset.csv - MATLAB 2022
%% =============================================

%% 1. CARGAR
T = readtable('sleep_health_dataset.csv', 'TextType', 'string');
disp('Tamaño original:'); disp(size(T))

%% 2. ELIMINAR ID
T.person_id = [];

%% 3. GUARDAR sleep_disorder_risk ANTES DEL LOOP
% Lo separamos para recodificarlo correctamente al final
T2 = readtable('sleep_health_dataset.csv', 'TextType', 'string');
sleep_disorder_raw = T2.sleep_disorder_risk;  % guardamos el string original

%% 4. CONVERTIR TODAS LAS COLUMNAS STRING A DOUBLE AUTOMÁTICAMENTE
varNames = T.Properties.VariableNames;
for i = 1:length(varNames)
    col = varNames{i};
    if isstring(T.(col))
        T.(col) = double(categorical(T.(col)));
    end
end

%% 5. RECODIFICAR sleep_disorder_risk CON ORDEN CORRECTO
orden = ["Healthy","Mild","Moderate","Severe"];
T.sleep_disorder_risk = double(categorical(sleep_disorder_raw, orden, 'Ordinal', true));

disp('Verificación sleep_disorder_risk (debe ser 1 2 3 4):')
disp(unique(T.sleep_disorder_risk)')

%% 6. MOVER sleep_disorder_risk AL FINAL DE LA TABLA
% Extraer la columna target
target = T.sleep_disorder_risk;
T.sleep_disorder_risk = [];      % eliminar de su posición actual
T.sleep_disorder_risk = target;  % agregar al final

disp('Última columna (debe ser sleep_disorder_risk):')
disp(T.Properties.VariableNames(end))

%% 7. ELIMINAR NULOS
disp('Nulos por columna:')
disp(sum(ismissing(T)))
T = rmmissing(T);
disp('Tamaño tras limpiar nulos:'); disp(size(T))

%% 8. CONVERTIR A MATRIZ NUMÉRICA
M = table2array(T);
disp('¿Todo numérico?'); disp(isnumeric(M))

%% 9. SEPARAR FEATURES Y TARGET
X = M(:, 1:end-1);   % todo menos la última columna
Y = M(:, end);        % sleep_disorder_risk → 1,2,3,4

fprintf('Features X: %d filas x %d columnas\n', size(X,1), size(X,2))
fprintf('Target  Y: %d filas x %d columnas\n', size(Y,1), size(Y,2))
fprintf('Clases únicas en Y: %s\n', num2str(unique(Y)'))

%% 10. GUARDAR CSV LIMPIO
writetable(T, 'sleep_health_limpio.csv');
disp('✔ Guardado: sleep_health_limpio.csv')