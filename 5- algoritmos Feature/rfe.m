function [support, ranking] = rfe (X, y, train_and_get_importance, n_features_to_select, step)
    % independent_rfe_with_ranking: Eliminación Recursiva con soporte de Ranking exacto
    
    n_features = size(X, 2);
    
    % 1. Inicialización idéntica a Scikit-Learn
    support = true(1, n_features);   % Todas empiezan vivas (true)
    ranking = ones(1, n_features);   % Todas empiezan con rango 1
    
    while sum(support) > n_features_to_select
        % Filtrar características activas
        current_features = find(support);
        X_subset = X(:, current_features);
        
        % Entrenar modelo externo (ej: ELM) y obtener pesos
        importances = train_and_get_importance(X_subset, y);
        importances = abs(importances);
        
        % Calcular cuántas eliminar sin pasarnos del límite
        n_to_remove = min(step, sum(support) - n_features_to_select);
        
        % Ordenar las importancias de menor a mayor
        [~, sorted_indices] = sort(importances, 'ascend');
        
        % Identificar las peores variables de esta ronda
        worst_local_indices = sorted_indices(1:n_to_remove);
        features_to_remove = current_features(worst_local_indices);
        
        % 2. Eliminación
        support(features_to_remove) = false; 
        
        % 3. El truco del Ranking (Idéntico a ranking_[~support_] += 1)
        % A todas las variables que actualmente sean 'false' en el support, 
        % se les suma 1 a su ranking.
        ranking(~support) = ranking(~support) + 1;
        
        fprintf('Iteración completada. Características restantes: %d\n', sum(support));
    end
end