clear all; clc; format long g
% addpath('./MDS/CONTINENTAL')
tic
% Constantes
G = 6.67259*1E-11;   % m3 kg-1 s-2
rho=2670;          % kg m-3

% Importando o MDE
H = double(imread('MDS_MERIT_SRTM15PLUS_900m_fill.tif'));

[lin,col]=size(H);

% Resolução
res=0.008333333333333333868; res_metros=900; res_metros_2=res_metros^2;    % Resolução em °, em metros e em metros ao quadrado, respectivamente
raio_integracao=100000;                          % raio de integração em metros

% Coordenadas geodésicas do canto superior esquerdo do primeiro pixel da imagem no canto NW - superior esquerdo - depois passando para o centro geométrico do pixel

latN=-21.2504166666666663-(res/2);    % latN
lonW=-55.7495833333333337+(res/2);   % longW
latS=-28.7504166666666627+(res/2);    % latS
lonE=-47.2495833333333337-(res/2);   % longE

% Grade de coordenadas geodésicas
gradeMDA_lat=(latN:-res:latS)'.*ones(1,col);
gradeMDA_long=(lonW:res:lonE).*ones(lin,1);

% Coordenadas geodésicas das estações de cálculo
latP=-(25+26/60+54.12695/3600);        % em graus decimais
longP=-(49+13/60+51.43717/3600);       % em graus decimais
HP=918.520870155954;                   % HP extraído do MDS
% HP=interp2(gradeMDA_long,gradeMDA_lat,H,longP,latP,'linear');   % interpolação para obtenção do valor de HN do MDE nas posições de cálculo

cte1=(G*rho/2)*res_metros_2*1E5;  % mGal
cte2=(3*G*rho/8)*res_metros_2*1E5;  % mGal
Ct_total=0;

% Correção do terreno

for i=1:lin
    for j=1:col
        delta_y=(gradeMDA_lat(i,j)-latP)*3600*30;     % m
        delta_x=(gradeMDA_long(i,j)-longP)*3600*30;   % m
        distancia=(delta_x^2+delta_y^2)^0.5;   % distância plana - l0
%         distancia=distancia+(distancia^3)/(3*6371000^2);    % correção de curvatura na distância

        if distancia<=raio_integracao && H(i,j)~=-9999
%         H(i,j)=H(i,j)+(distancia^2)/(2*6371000);    % correção de curvatura
%         Ct=cte1*(((H(i,j)-HP(ii,1))^2)/distancia^3);
        Ct=cte1*(((H(i,j)-HP)^2)/distancia^3)  -  cte2*(((H(i,j)-HP)^4)/distancia^5);
        else
        Ct=0;
        end

Ct_total=Ct_total+Ct;
    end
end

toc
