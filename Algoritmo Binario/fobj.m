function cost = fobj(binX)
% Función objetivo escalar con frente de Pareto INCLINADO hacia sparse
% Compatible con BSFOA (retorna un solo escalar)
%
% IMPORTANTE: Ejecutar ANTES del algoritmo (una sola vez):
%   global X; X = zscore(X);

    global X Y

    % --- Sanidad del vector binario ---
    binX = logical(binX(:)');
    nD   = size(X, 2);

    if length(binX) > nD
        binX = binX(1:nD);
    elseif length(binX) < nD
        binX = [binX, false(1, nD - length(binX))];
    end

    % --- Sin features: peor caso ---
    if sum(binX) == 0
        cost = 1;
        return;
    end

    % --- Clasificación KNN con Leave-One-Out ---
    X_sel = X(:, binX);
    mdl   = fitcknn(X_sel, Y, 'NumNeighbors', 5, 'Leaveout', 'on');
    f1    = kfoldLoss(mdl);             % error ∈ [0, 1]

    % --- Fracción de features con penalización cuadrática ---
    % La cuadratura INCLINA el frente: pocas features = penalización mucho menor
    feat_ratio = sum(binX) / nD;
    f2 = feat_ratio^2;                  % ∈ [0, 1], presiona hacia soluciones sparse

    % --- Costo escalar agregado ---
    alpha = 0.90;   % prioriza precisión
    beta  = 0.10;   % penaliza features cuadráticamente
    cost  = alpha * f1 + beta * f2;

end