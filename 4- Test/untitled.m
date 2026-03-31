% main code 
clear all;close all;clc

% Parameters of algorithm
Npop = 50;              % Number of search agents
Max_it = 2000;          % Maximum number of iterations



lb= 1000;
ub= 2000;
nD= 40;



[xpos,fval,Curve] = SFOA(Npop,Max_it,lb,ub,nD,@sphere); 

disp(xpos)
function y = sphere(x)
    y = sum(x.^2);
    
end


