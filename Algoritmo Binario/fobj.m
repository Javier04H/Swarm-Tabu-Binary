function cost = fobj(binX)
    % Función objetivo para Feature Selection con BSFOA
    % binX: vector lógico/binario 1 x nD indicando features activas
    %
    % Minimiza: alpha * error_clasificacion + beta * fraccion_features

    alpha = 0.99;  % peso del error (mayor = prioriza precisión)
    beta  = 0.01;  % peso del número de features

    global X Y

    % Asegurar que binX sea vector fila lógico de tamaño correcto
    binX = logical(binX(:)');          % forzar fila
    nD   = size(X, 2);

    % Recortar o rellenar si hay desajuste de dimensiones
    if length(binX) > nD
        binX = binX(1:nD);
    elseif length(binX) < nD
        binX = [binX, false(1, nD - length(binX))];
    end

    % Penalización si no hay ninguna feature seleccionada
    if sum(binX) == 0
        cost = 1;
        return;
    end

    % Seleccionar columnas activas
    X_sel = X(:, binX);

    % Clasificador KNN con Leave-One-Out (rápido, sin hiperparámetros extra)
    mdl  = fitcknn(X_sel, Y, 'NumNeighbors', 5, 'Leaveout', 'on');
    loss = kfoldLoss(mdl);   % error de clasificación ∈ [0, 1]

    % Proporción de features usadas ∈ [0, 1]
    feat_ratio = sum(binX) / nD;

    % Costo total (minimizar)
    cost = alpha * loss + beta * feat_ratio;
end