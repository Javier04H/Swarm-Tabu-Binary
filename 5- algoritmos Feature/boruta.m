function [finalDecision, hitReg] = boruta(X, Y, pValue, maxRuns)
    % boruta_fiel: Implementación clon del algoritmo Boruta original en R
    % X: Matriz de predictores (Observaciones x Variables)
    % Y: Vector de respuesta
    % pValue: Nivel de confianza (Por defecto 0.01)
    % maxRuns: Límite de iteraciones (Por defecto 100)%CAMBIADO A 50
    
    if nargin < 3, pValue = 0.01; end
    if nargin < 4, maxRuns = 50; end
    
    [nObs, nAtt] = size(X);
    
    % finalDecision: 0 = Tentative, 1 = Confirmed, -1 = Rejected
    finalDecision = zeros(1, nAtt); 
    hitReg = zeros(1, nAtt);
    runs = 0;

    while any(finalDecision == 0) && runs < maxRuns
        runs = runs + 1;
        fprintf('Ejecutando iteración %d de %d...\n', runs, maxRuns);

        % 1. Filtrar atributos (Confirmed y Tentative siguen compitiendo)
        idx_keep = find(finalDecision ~= -1);
        X_keep = X(:, idx_keep);
        n_keep = length(idx_keep);

        % 2. Crear variables sombra (Shadow Features)
        X_shadow = X_keep;
        
        % REGLA FIEL: Asegurar al menos 5 variables sombra
        while size(X_shadow, 2) < 5
            X_shadow = [X_shadow, X_shadow]; %#ok<AGROW>
        end
        
        % Permutar aleatoriamente cada columna sombra (equivalente a sample en R)
        n_shadow = size(X_shadow, 2);
        for i = 1:n_shadow
            X_shadow(:, i) = X_shadow(randperm(nObs), i); 
        end

        % 3. Entrenar el Random Forest
        X_aug = [X_keep, X_shadow];
        
        % Se usan 500 árboles (default de Boruta) y se extrae la
        % importancia pura
        %CAMBIE A 50 ARBOLES PARA VERIFICAR VELOCIDAD
        Mdl = TreeBagger(50, X_aug, Y, 'Method', 'classification', ...
                         'OOBPredictorImportance', 'on');
                         
        impRaw = Mdl.OOBPermutedPredictorDeltaError;

        % 4. Separar importancias
        imp_real = impRaw(1:n_keep);
        imp_shadow = impRaw(n_keep+1:end);
        
        % La métrica contra la que compiten es la sombra máxima
        max_shadow = max(imp_shadow);

        % 5. Asignar Hits
        hits = imp_real > max_shadow;
        hitReg(idx_keep) = hitReg(idx_keep) + hits;

        % 6. El Juez: Pruebas Binomiales y Ajuste de Bonferroni
        % Probabilidad de obtener los hits actuales o más por azar (lower.tail=FALSE)
        p_accept = 1 - binocdf(hitReg - 1, runs, 0.5);
        
        % Probabilidad de obtener los hits actuales o menos por azar (lower.tail=TRUE)
        p_reject = binocdf(hitReg, runs, 0.5);

        % Ajuste de Bonferroni: p.adjust(..., method='bonferroni') en R
        % Se multiplica el valor P por el número total de atributos originales
        p_accept_adj = min(1, p_accept * nAtt);
        p_reject_adj = min(1, p_reject * nAtt);

        % Tomar decisiones solo sobre las variables Tentative (0)
        to_accept = (finalDecision == 0) & (p_accept_adj < pValue);
        to_reject = (finalDecision == 0) & (p_reject_adj < pValue);

        finalDecision(to_accept) = 1;
        finalDecision(to_reject) = -1;
        
        fprintf('  Confirmadas: %d | Rechazadas: %d | Tentativas: %d\n', ...
                sum(finalDecision == 1), sum(finalDecision == -1), sum(finalDecision == 0));
    end
    
    disp('¡Algoritmo Boruta finalizado!');
end