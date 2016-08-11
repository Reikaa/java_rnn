classdef lstm
    
    properties
        lstm_model
        dim
    end
    
    methods
        function obj = lstm(dim,cell_size)
            r = java.util.Random(979);
            obj.lstm_model = com.sq.tsingjyujing.rnn.LSTM(r,dim,dim,cell_size);
            obj.dim = dim;
        end
        
        function disp_paras(obj)
            obj.lstm_model.Display();
        end
        
        function obj = train(obj,seqs,iters)
            AccR = zeros(1,iters);
            wb = waitbar(0);
            for iter = 1:iters
                fitted = 0;
                fitall = 0;
                waitbar(0,wb,['Iter:' num2str(iter)]);
                for i = 1:length(seqs)
                    waitbar(i/length(seqs),wb);
                    seq = seqs{i};
                    obj.lstm_model.Reset();
                    dm = onehot(seq,obj.dim);
                    fitall = fitall + length(seq);
                    for j = 1:(length(seq) - 1)
                        rv = obj.lstm_model.Next(dm(j,:),dm(j+1,:));
                        u = invonehot(rv(:)');
                        if u == seq(j+1)
                            fitted = fitted + 1;
                        end
                    end
                end
                AccR(iter) = fitted*100/fitall;
                plot(AccR(1:iter));drawnow;
                fprintf('AccRate:%f%%\n',AccR(iter));
                
                F=obj.lstm_model.weightsF;
                G=obj.lstm_model.weightsG;
                O=obj.lstm_model.weightsOut;
                save('lstm_log','F','G','O')
            end
            close(wb)
        end
        
        
        function pdseq = predict(obj,seqs)
            pdseq = seqs;
            for i = 1:length(seqs)
                seq = seqs{i};
                dm = onehot(seq,obj.dim);
                dom = zeros(length(seq),obj.dim);
                dom(1,:) = dm(1,:);
                for j = 1:(length(seq)-1)
                    dom(j+1,:) = obj.lstm_model.Next(dm(j,:));
                end
                pdseq{i} = invonehot(dom);
            end
        end
    end
    
end

