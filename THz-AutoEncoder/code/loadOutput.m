%%   Distribution code Version 1.0 -- 24/07/2019 by Tak Ming Wong Copyright 2019, University of Siegen
%%
%%   The Code is created based on the method described in the following paper 
%%   [1] "Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction", T.M. Wong, M. Kahl, P. Haring Bolivar, A. Kolb, M. Moeller, 
%%   German Conference on Pattern Recognition (GCPR), 2019.
%%   
%%   If you use this code in your scientific publication, please cite the mentioned paper.
%%   The code and the algorithm are for non-comercial use only.

clear;
clc;
close all;

load('OutputMeasure_final.mat');

mkdir('figure');

diff = curvesIn - curvesOut;
loss = mean(real(diff).^2, 3) + mean( imag(diff).^2, 3);

I = A.^2;
theta = pi * Mu - Phi;

result_cnn_A = A;
result_cnn_mu = Mu;
result_cnn_sigma = Sigma;
result_cnn_phi = Phi;
result_cnn_loss = loss;

save('result_cnn.mat', 'result_cnn_*', '-v7.3');

figure;
imagesc( A );
axis image;
colorbar;
title('A');
print('figure/A.png', '-dpng');
print('figure/A.eps', '-depsc2');

figure;
imagesc( abs(A) );
axis image;
colorbar;
title('abs A');
print('figure/absA.png', '-dpng');
print('figure/absA.eps', '-depsc2');

figure;
imagesc( I );
axis image;
colorbar;
title('I_{thz}');
print('figure/Ithz.png', '-dpng');
print('figure/Ithz.eps', '-depsc2');

figure;
imagesc( Mu );
axis image;
colorbar;
title('mu');
caxis([6301-3 6301+3]);
print('figure/mu.png', '-dpng');
print('figure/mu.eps', '-depsc2');

figure;
imagesc( Sigma );
axis image;
colorbar;
title('sigma');
print('figure/sigma.png', '-dpng');
print('figure/sigma.eps', '-depsc2');

figure;
imagesc( wrapTo2Pi( Phi ) );
axis image;
colorbar;
title('phi');
print('figure/phi.png', '-dpng');
print('figure/phi.eps', '-depsc2');

figure;
imagesc( loss );
axis image;
colorbar;
title('loss');
print('figure/loss.png', '-dpng');
print('figure/loss.eps', '-depsc2');

figure;
imagesc( theta );
axis image;
colorbar;
title('Theta');
print('figure/theta.png', '-dpng');
print('figure/theta.eps', '-depsc2');
