package com.sq.tsingjyujing.rnn;

public interface IRNN {
    void Reset();
    double[] Next(double[] input, double[] target_output) throws Exception;
    double[] Next(double[] input) throws Exception;
}
