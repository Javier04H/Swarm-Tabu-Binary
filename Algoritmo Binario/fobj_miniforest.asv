function cost = fobj_miniforest(binX)
    global X_train Y_train X_val Y_val nD

    % --- Sanidad del vector binario ---
    binX = logical(binX(:)');

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

    % --- Selección de features activas ---
    X_tr_sel = X_train(:, binX);
    X_vl_sel = X_val(:, binX);

    try
        % Entrenar un Mini-Forest (Ensamble de 10 árboles)
        num_trees = 10;
        mdl = fitcensemble(X_tr_sel, Y_train, 'Method', 'Bag', 'NumLearningCycles', num_trees);
        
        % Predecir con datos de validación
        pred = predict(mdl, X_vl_sel);
        f1   = mean(pred ~= Y_val); % Error de clasificación más estable
    catch
        % Si falla (ej. por colinealidad extrema), asignamos peor error
        f1 = 1; 
    end

    % --- Fracción de features con penalización lineal suavizada ---
    feat_ratio = sum(binX) / nD;
    f2 = feat_ratio * 2;

    % --- Costo escalar agregado ---
    alpha = 0.97;
    beta  = 0.03;
    cost  = alpha * f1 + beta * f2;
end