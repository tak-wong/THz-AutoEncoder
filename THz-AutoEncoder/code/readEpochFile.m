%%   Distribution code Version 1.0 -- 24/07/2019 by Tak Ming Wong Copyright 2019, University of Siegen
%%
%%   The Code is created based on the method described in the following paper 
%%   [1] "Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction", T.M. Wong, M. Kahl, P. Haring Bolivar, A. Kolb, M. Moeller, 
%%   German Conference on Pattern Recognition (GCPR), 2019.
%%   
%%   If you use this code in your scientific publication, please cite the mentioned paper.
%%   The code and the algorithm are for non-comercial use only.

close all;
clc;

fid = fopen('epoch.txt'); 
content = textscan(fid, '%d %f %f %f %s %s', 'HeaderLines', 1, 'Delimiter', ','); 
fclose(fid);

epoch_index = content{1};
epoch_lr = content{2};
epoch_loss_train = content{3};
epoch_loss_valid = content{4};
epoch_start_tstr = content{5};
epoch_end_tstr = content{6};

epoch_start_time = cell(length(content{5}), 1);
epoch_end_time = cell(length(content{6}), 1);
epoch_elapsed_time = cell(length(content{1}), 1);
for it = 1:length(epoch_index)
   epoch_start_time{it} = datetime(epoch_start_tstr{it},'InputFormat','yyyy/MM/dd:HH:mm:ss.SSSSSS');
   epoch_end_time{it} = datetime(epoch_end_tstr{it},'InputFormat','yyyy/MM/dd:HH:mm:ss.SSSSSS');
   epoch_elapsed_time{it} = epoch_end_time{it} - epoch_start_time{it};
end


elapsed_time_total = 0;
fprintf('Epoch, LR, Average Loss, Elapsed Sec\n');
for it = 1:length(epoch_index)
    elapsed_time_total = elapsed_time_total + seconds(epoch_elapsed_time{it});
    fprintf('%05d, %.8f, %12.8f, %12.8f, %11.6f\n', epoch_index(it), epoch_lr(it), epoch_loss_train(it), epoch_loss_valid(it), seconds(epoch_elapsed_time{it}));
end
elapsed_time_totalstr = datestr(seconds(elapsed_time_total),'HH:MM:SS');
fprintf('elapsed time: %5.4f sec (%s)\n', elapsed_time_total, elapsed_time_totalstr)

valid_interval = 50;

figure;
plot( epoch_index, epoch_loss_train, 'k-', 'LineWidth', 1.1 );
hold on;
plot( epoch_index(1:valid_interval:end), epoch_loss_valid(1:valid_interval:end), 'b-.', 'LineWidth', 1.5 );
hold off;
grid on;
xlim( minmax(epoch_index') );
legend('training', 'validation');
title('loss');

figure;
plot( epoch_index, 10*log10(epoch_loss_train), 'k-', 'LineWidth', 1.1 );
hold on;
plot( epoch_index(1:valid_interval:end), 10*log10(epoch_loss_valid(1:valid_interval:end)), 'b-.', 'LineWidth', 1.5 );
hold off;
grid on;
xlim( minmax(epoch_index') );
legend('training', 'validation');
title('10*log10(loss)');
