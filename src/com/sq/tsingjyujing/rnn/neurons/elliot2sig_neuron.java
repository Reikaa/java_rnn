package com.sq.tsingjyujing.rnn.neurons;

public class elliot2sig_neuron extends Neuron {
    @Override
    public double Activate(double x) {
        double x2 = x*x;
        return Math.signum(x)*x2/(1+x2);
    }

    @Override
    public double Derivative(double x) {
        // TODO Auto-generated method stub
        double x2 = x*x;
        double tmpval = 1 + x2;
        return 2*Math.abs(x)/(tmpval*tmpval);
    }
}