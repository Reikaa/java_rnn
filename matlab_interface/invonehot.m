function seq = invonehot(data_matrix)
fu = com.sq.tsingjyujing.rnn.util;
N = size(data_matrix,1);
seq = zeros(N,1);
for i = 1:N
    seq(i) = fu.argmax(data_matrix(i,:))+1;
end
end