clear all;
close all;
Folder='F:\Antoine Lagarde\20160726\colloide5 bin2 x10\absorbance1\200hz\particle tracking'; % Folders containing stacks at Frequency1
cd(Folder);
load('trackabsorbance1200hz');

%% Initialisation
Nblink                 = max(objs_link(6,:)); %nombre de trajectoires
Nbobjets             = max(objs_link(4,:)); %nombre de bactéries différentes
longueurminimale     = 1e3; %on ne prend que les trajectoires qui apparaissent dans suffisamment d'images

deltaglobal         = zeros(1,longueurminimale); 
deltamoyenglobal     = zeros(1,longueurminimale);
nbtrajectoiremobile    = 0;
nbtrajectoirelongue = 0;

binning        = 2;
PixelSize     = binning*(6450/10)*10^-3; %Conversion 1 pixel -> um
frequence     = 200;

%% trajectoire
i=1;

   if length(objs_link(1,find(objs_link(6,:)==i)))> longueurminimale; %selection des trajectoires qui apparaissent dans suffisamment d'images

        x1 = objs_link(1,find(objs_link(6,:)==i)).*PixelSize; %abscisse de la trajectoire
        y1 = objs_link(2,find(objs_link(6,:)==i)).*PixelSize; %ordonnée de la trajectoire

        nbtrajectoirelongue = nbtrajectoirelongue+1;             %nombre de trajectoires qu'on garde
        dureetrajectoire     = length(x1);                         %nombre d'images dans lesquelles la trajectoire est présente
    end
