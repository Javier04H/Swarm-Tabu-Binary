
clear all;close all;clc

%dataset = 'BiT-Monkey-Species';
%h5disp("BiT-Monkey-Species.h5")

archivo = 'BiT-Monkey-Species.h5';

kernel = h5read(archivo, '/model_weights/dense/dense/kernel:0'); % 10x2048
bias   = h5read(archivo, '/model_weights/dense/dense/bias:0');   % 10x1

% Norma de cada fila del kernel = "fuerza" de cada clase
normas = vecnorm(kernel, 2, 2);

clases = ["Clase_1","Clase_2","Clase_3","Clase_4","Clase_5", ...
          "Clase_6","Clase_7","Clase_8","Clase_9","Clase_10"]';

T = table(clases, normas, bias, 'VariableNames', {'Clase','NormaKernel','Bias'});
disp(T)