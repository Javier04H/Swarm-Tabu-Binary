function [xposbest, fvalbest, Curve, Sf, Nf] = CBSFOASig2(Npop, Max_it, lb, ub, nD, fobj)
% Starfish Optimization Algorithm (SFOA) - Versión Binaria Caótica (Piecewise)
% Original Created by Dr. Changting Zhong (zhongct@hainanu.edu.cn)
% Modificado para incluir Mapa Caótico Piecewise (PWLCM) y Binarización

%% Inicialización
GP = 0.5; % Parámetro de control de SFOA
p_map = 0.4; % Parámetro del Mapa Caótico Piecewise (0 < p < 0.5)

if size(ub, 2) == 1
    lb = lb * ones(1, nD);
    ub = ub * ones(1, nD);
end

fvalbest = inf;
Curve = zeros(1, Max_it);
Xpos = zeros(Npop, nD);

% Se utiliza el mapa PWLCM para generar la población inicial en lugar de rand()
chaos = rand; % Semilla inicial aleatoria
for i = 1:Npop
    for j = 1:nD
        chaos = pwlcm(chaos, p_map);
        % Mapeo al rango [lb, ub]
        Xpos(i, j) = lb(j) + chaos * (ub(j) - lb(j));
    end
end
% -----------------------------------------------------------------

Fitness = zeros(1, Npop);
for i = 1:Npop
    % Transformar a probabilidad usando Sigmoide (Transfer Function)
    TF = 1 ./ (1 + exp(-10 * (Xpos(i, :) - 0.5)));
    % Binarizamos comparando con un valor aleatorio
    binX = TF >= 0.5;
    % Evaluamos el Fitness con la versión binarizada
    Fitness(i) = feval(fobj, binX);
end

[fvalbest, order] = min(Fitness);
xposbest = Xpos(order, :); % Mejor posición global (continua)

newX = zeros(Npop, nD);

%% Evolución
T = 1;
while T <= Max_it
    theta = (pi / 2) * (T / Max_it);
    tEO = ((Max_it - T) / Max_it) * cos(theta);
    
    % Decisión de exploración vs explotación
    if rand < GP
        chaos = rand;
        for m = 1:Npop
            chaos = pwlcm(chaos,p_map);
            %para evitar que el caos caiga en un solo valor
            if chaos < 0.0000001 || chaos > 0.9999999
                chaos = rand;
                chaos = pwlcm(chaos, p_map);
            end
            chaos_agente(m) =chaos;
        end

        for i = 1:Npop
            
            if nD > 5
                % Para nD mayor a 5
                jp1 = randperm(nD, 5);
                for j = 1:5
                    pm = (2 * chaos_agente(i) - 1) * pi;
                    if rand < GP
                        newX(i, jp1(j)) = Xpos(i, jp1(j)) + pm * (xposbest(jp1(j)) - Xpos(i, jp1(j))) * cos(theta);
                    else
                        newX(i, jp1(j)) = Xpos(i, jp1(j)) - pm * (xposbest(jp1(j)) - Xpos(i, jp1(j))) * sin(theta);
                    end
                    % Control de límites
                    if newX(i, jp1(j)) > ub(jp1(j)) || newX(i, jp1(j)) < lb(jp1(j))
                        newX(i, jp1(j)) = Xpos(i, jp1(j));
                    end
                end
            else
                % Para nD menor o igual a 5
                jp2 = ceil(nD * rand);
                im = randperm(Npop);
                rand1 = 2 * chaos_agente(i) - 1;
                chaos_agente = pwlcm(chaos_agente(i), p_map);%para dar el siguiente valor caotico
                rand2 = 2 * chaos_agente(i) - 1;
                newX(i, jp2) = tEO * Xpos(i, jp2) + rand1 * (Xpos(im(1), jp2) - Xpos(i, jp2)) + rand2 * (Xpos(im(2), jp2) - Xpos(i, jp2));
                if newX(i, jp2) > ub(jp2) || newX(i, jp2) < lb(jp2)
                    newX(i, jp2) = Xpos(i, jp2);
                end
            end
            newX(i, :) = max(min(newX(i, :), ub), lb); % Boundary check
        end
    else
        % Explotación
        df = randperm(Npop, 5);
        dm = zeros(5, nD);
        dm(1, :) = xposbest - Xpos(df(1), :);
        dm(2, :) = xposbest - Xpos(df(2), :);
        dm(3, :) = xposbest - Xpos(df(3), :);
        dm(4, :) = xposbest - Xpos(df(4), :);
        dm(5, :) = xposbest - Xpos(df(5), :); % Cinco brazos de la estrella de mar
        
        for i = 1:Npop
            r1 = rand;
            r2 = rand;
            kp = randperm(length(df), 2);
            newX(i, :) = Xpos(i, :) + r1 * dm(kp(1), :) + r2 * dm(kp(2), :);
            if i == Npop
                newX(i, :) = exp(-T * Npop / Max_it) .* Xpos(i, :); % Regeneración
            end
            newX(i, :) = max(min(newX(i, :), ub), lb); % Boundary check
        end
    end
    
    % Evaluación de Fitness
    for i = 1:Npop
        % Sigmoide para binarización
        TF = 1 ./ (1 + exp(-10 * (newX(i, :) - 0.5)));
        binX = TF >= 0.5;
        newFit = feval(fobj, binX);
        
        if newFit < Fitness(i)
            Fitness(i) = newFit;
            Xpos(i, :) = newX(i, :);
            if newFit < fvalbest
                fvalbest = Fitness(i);
                xposbest = Xpos(i, :);
            end
        end
    end
    
    Curve(T) = fvalbest;
    T = T + 1;
end

% Resultados finales binarizados
TF_best = 1 ./ (1 + exp(-10 * (xposbest - 0.5)));
bin_best = TF_best >= 0.5;
Sf = find(bin_best == 1); % Índices de características seleccionadas
Nf = length(Sf); % Cantidad de características

end

%% Chaos Piecewise Formula
function c = pwlcm(c, p)
    if c >= 0 && c < p
        c = c / p;
    elseif c >= p && c < 0.5
        c = (c - p) / (0.5 - p);
    elseif c >= 0.5 && c < 1 - p
        c = (1 - p - c) / (0.5 - p);
    elseif c >= 1 - p && c < 1
        c = (1 - c) / p;
    end
end
