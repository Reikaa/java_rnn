package com.sq.tsingjyujing.rnn.neurons;

public class satur_neuron extends Neuron {
    @Override
    public double Activate(double x) {
        if (x>1.0){
            return 1.0;
        }else if(x<-1.0){
            return -1.0;
        }else{
            return x;
        }
    }

    @Override
    public double Derivative(double x) {
        if (x<=1.0 || x>=-1.0){
            return 1;
        }else{
            return 0;
        }
    }
}