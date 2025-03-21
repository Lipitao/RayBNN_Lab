

W = readmatrix('W.csv');



neuron_size = 1600;
input_size = 1000;
output_size = 9;
sel = 1;



Wi = W(sel,:);
Wi(Wi ~= 0.0) = 1.0;
Wi = reshape(Wi,[neuron_size ,neuron_size])';

G = digraph(Wi);


TR = shortestpathtree(G,'all',1599);
plot(TR)

