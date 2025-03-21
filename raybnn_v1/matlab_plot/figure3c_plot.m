clear all;
close all;
fontsz=16;


before_delete_WRowIdxCOO = readmatrix("before_delete_WRowIdxCOO.csv");
before_delete_WColIdx = readmatrix("before_delete_WColIdx.csv");
before_delete_WValues = readmatrix("before_delete_WValues.csv");

after_delete_WRowIdxCOO = readmatrix("after_delete_WRowIdxCOO.csv");
after_delete_WColIdx = readmatrix("after_delete_WColIdx.csv");
after_delete_WValues = readmatrix("after_delete_WValues.csv");

neuron_size =  max(before_delete_WRowIdxCOO)+1;

before_global_idx = (before_delete_WRowIdxCOO*(neuron_size)) +  before_delete_WColIdx;

after_global_idx = (after_delete_WRowIdxCOO*(neuron_size)) +  after_delete_WColIdx;

[C,idx] = setdiff(before_global_idx,after_global_idx);

delWeights = before_delete_WValues(idx);



figure(1);clf;
hold on;

histogram(100.0*delWeights/max(delWeights));
set(gca, 'YScale', 'log','box','on');
set(gca,'FontSize',fontsz);
ylabel('Counts','FontSize',fontsz);
xlabel('Deletion Percentile','FontSize',fontsz);
xlim([0 100]);

saveas(gcf,"percentile",'fig');
saveas(gcf,"percentile",'png');


