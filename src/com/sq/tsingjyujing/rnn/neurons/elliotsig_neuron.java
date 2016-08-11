package com.sq.tsingjyujing.rnn.neurons;

public class elliotsig_neuron extends Neuron {
    @Override
    public double Activate(double x) {
        return x/(1+Math.abs(x));
    }

    @Override
    public double Derivative(double x) {
        // TODO Auto-generated method stub
        double tmpvar = 1 + x*Math.signum(x);
        return (tmpvar*tmpvar);
    }
}