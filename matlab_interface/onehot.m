function data_matrix = onehot(input_series,dim)
N = length(input_series);
T = 1:N;
V = ones(N,1);
data_matrix = full(sparse(T,double(input_series),V,N,dim));
end