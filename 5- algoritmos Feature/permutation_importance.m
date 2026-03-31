function result = permutation_importance(modelo, X, y, metric_fn, n_repeats)
    % X: Matriz de datos (n_muestras x n_caracteristicas)
    % y: Vector de respuestas reales
    % metric_fn: Función para evaluar el error (ej. @(y, y_pred) mse(y, y_pred))
    
    [n_samples, n_features] = size(X);
    
    % 1. Calcular el score base (baseline_score en el archivo de Python)
    y_pred_base = predict(modelo, X);
    baseline_score = metric_fn(y, y_pred_base);
    
    % Preparar matriz para guardar resultados (n_features x n_repeats)
    importances = zeros(n_features, n_repeats);
    
    % 2. El bucle paralelo (Equivalente al Parallel(delayed(...)) en Python)
    parfor col = 1:n_features
        
        scores_temp = zeros(1, n_repeats);
        
        % 3. El bucle de repeticiones (Equivalente a _calculate_permutation_scores)
        for rep = 1:n_repeats
            X_permuted = X; % Hacemos una copia temporal
            
            % Desordenar (shuffling_idx en Python)
            idx_desordenado = randperm(n_samples);
            X_permuted(:, col) = X_permuted(idx_desordenado, col);
            
            % Volver a predecir y calcular score
            y_pred_perm = predict(modelo, X_permuted);
            permuted_score = metric_fn(y, y_pred_perm);
            
            % Guardar la diferencia de error
            % (Si es accuracy, base - permuted. Si es MSE, permuted - base)
            scores_temp(rep) = baseline_score - permuted_score; 
        end
        importances(col, :) = scores_temp;
    end
    
    % 4. Armar el resultado final (Equivalente a _create_importances_bunch)
    result.importances_mean = mean(importances, 2);
    result.importances_std = std(importances, 0, 2);
    result.importances = importances;
end