%%% Carga de datasets
load("CSV_LIMPIOS.mat")

N_runs = 1;

if ~exist('ultima_run','var')
    ultima_run = 0;
end

elm_type = 1;                  % 0: Regresión, 1: Clasificación
act_func = 'sig';              % Función de activación

% 2. Definición de la grilla gruesa (Escala logarítmica)
grid_neurons = [10, 50, 100, 200, 500]; 
grid_C = 10.^[-5, -3, -1, 1, 3, 5]; 

% Inicializar matrices para almacenar la precisión/exactitud
% Filas: Neuronas, Columnas: Valores de C
training_matrix = zeros(length(grid_neurons), length(grid_C));
testing_matrix = zeros(length(grid_neurons), length(grid_C));

for run=(ultima_run +1) : N_runs

    rng(run * 10);
    c = cvpartition(y , "KFold", 10);

    for i = 1 : c.NumTestSets
        trainIdx = training(c, i);
        testIdx  = test(c, i);

        XBoru_train =  X_RELM_boruta(trainIdx);
        XBSFOA_train = X_RELM_cbsfoa1(trainIdx);
        XBPI_train = X_RELM_pi(trainIdx);
            
            
        XBoru_test =  X_RELM_boruta(testIdx);
        XBSFOA_test = X_RELM_cbsfoa1(testIdx);
        XBPI_test = X_RELM_pi(testIdx);
        for j = 1:length(grid_neurons)
            num_neurons = grid_neurons(j);
    
            for k = 1:length(grid_C)
                    C_val = grid_C(k);
        
        % Ejecutar R-ELM para la combinación actual
                    [TrainTime, TestTime, TrainAcc, TestAcc] = R_ELM(...
                    train_file, test_file, elm_type, num_neurons, act_func, C_val);
        
        % Almacenar resultados
                    training_matrix(i, j) = TrainAcc;
                    testing_matrix(i, j) = TestAcc;
        
                    fprintf('Neurons: %d | C: %e -> Test Acc/RMSE: %.4f\n', ...
                    num_neurons, C_val, TestAcc);

            end
        end
    if elm_type == 1
    [best_val, idx] = max(testing_matrix(:)); % Máxima precisión
    else
    [best_val, idx] = min(testing_matrix(:)); % Mínimo RMSE
    end
    end
end
            
 