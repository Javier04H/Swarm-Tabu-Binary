%% =============================================
%  LIMPIEZA Y CODIFICACIÓN DEL DATASET
%  heart.csv - Adaptado del script de referencia
%% =============================================

%% 1. CARGAR
T = readtable('heart.csv', 'TextType', 'string');
disp('Tamaño original:'); disp(size(T))

%% 2. ELIMINAR ID
% El dataset heart.csv no tiene una columna de ID explícita, 
% por lo que omitimos este paso. Si existiera, sería: T.ID = [];

%% 3. GUARDAR TARGET ANTES DEL LOOP
% Aunque HeartDisease ya es numérica (0 y 1), la guardamos por consistencia
target_raw = T.HeartDisease;  

%% 4. CONVERTIR TODAS LAS COLUMNAS STRING A DOUBLE AUTOMÁTICAMENTE
% Esto codificará variables como Sex, ChestPainType, RestingECG, ExerciseAngina y ST_Slope
varNames = T.Properties.VariableNames;
for i = 1:length(varNames)
    col = varNames{i};
    if isstring(T.(col))
        T.(col) = double(categorical(T.(col)));
    end
end

disp('Verificación de variables codificadas (ejemplo Sex: 1=F, 2=M):')
disp(unique(T.Sex)')

%% 5. RECODIFICAR VARIABLES ORDINALES (OPCIONAL)
% A diferencia del sleep_disorder_risk, la mayoría de las categorías aquí son nominales.
% ST_Slope ("Up", "Flat", "Down") podría considerarse ordinal. Si deseas forzar un orden:
% orden_slope = ["Down", "Flat", "Up"];
% T.ST_Slope = double(categorical(T2.ST_Slope, orden_slope, 'Ordinal', true));

%% 6. MOVER TARGET AL FINAL DE LA TABLA
% Aseguramos que HeartDisease quede siempre como la última columna
target = T.HeartDisease;
T.HeartDisease = [];      % eliminar de su posición actual
T.HeartDisease = target;  % agregar al final

disp('Última columna (debe ser HeartDisease):')
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
X = M(:, 1:end-1);   % todo menos la última columna (Features)
Y = M(:, end);       % HeartDisease → 0, 1

fprintf('Features X: %d filas x %d columnas\n', size(X,1), size(X,2))
fprintf('Target  Y: %d filas x %d columnas\n', size(Y,1), size(Y,2))
fprintf('Clases únicas en Y: %s\n', num2str(unique(Y)'))

%% 10. GUARDAR CSV LIMPIO
writetable(T, 'heart_limpio.csv');
disp('✔ Guardado: heart_limpio.csv')