package com.sq.tsingjyujing.rnn.neurons;

public class relu_neuron extends Neuron
{
    @Override
    public double Activate(double x) {
        if(x > 0.0){
            return x;
        }else{
            return 0.0;
        }
    }

    @Override
    public double Derivative(double x) {
        if(x > 0.0){
            return 1.0;
        }else{
            return 0.0;
        }
    }


}
