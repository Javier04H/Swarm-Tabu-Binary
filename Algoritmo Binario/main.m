% Leer el archivo como una tabla
opts = detectImportOptions('diabetes_data_upload.csv');
T = readtable('diabetes_data_upload.csv', opts);

% 1. Convertir variables de texto (Yes/No, Male/Female) a numéricas
vars = T.Properties.VariableNames;
for i = 1:width(T)
    if iscell(T{:,i}) || isstring(T{:,i})
        T.(vars{i}) = double(contains(string(T{:,i}), {'Yes', 'Male', 'Positive'}, 'IgnoreCase', true));
    end
end

% 2. Normalizar la columna 'Age'
T.Age = (T.Age - min(T.Age)) / (max(T.Age) - min(T.Age));

% 3. Convertir a matriz numérica
data  = table2array(T);
feat  = data(:, 1:end-1);
label = data(:, end);

% --- VARIABLES GLOBALES (necesarias para fobj) ---
global X Y
X = feat;
Y = label;

% --- PARÁMETROS ---
nD     = size(feat, 2);
lb     = -4;   % rango continuo para tanh: valores fuera de [-4,4] saturan
ub     =  4;
Npop   = 30;
Max_it = 50;

% --- EJECUCIÓN ---
[xposbest, fvalbest, Curve, Sf, Nf] = BSFOA(Npop, Max_it, lb, ub, nD, @fobj);

% --- RESULTADOS ---
fprintf('\nOptimización terminada.\n');
fprintf('Características seleccionadas : %d de %d\n', Nf, nD);
fprintf('Índices seleccionados         : %s\n', num2str(Sf));
fprintf('Mejor Fitness (costo)         : %.4f\n', fvalbest);

% Curva de convergencia
figure;
plot(Curve, 'b-', 'LineWidth', 2);
xlabel('Iteración'); ylabel('Fitness');
title('Curva de Convergencia - BSFOA');
grid on;