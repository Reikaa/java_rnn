package com.sq.tsingjyujing.rnn.neurons;

public class halfelliotsig_neuron extends Neuron {
    @Override
    public double Activate(double x) {
        return 0.5+0.5*x/(1+Math.abs(x));
    }

    @Override
    public double Derivative(double x) {
        // TODO Auto-generated method stub
        double tmpvar = 1 + x*Math.signum(x);
        return (tmpvar*tmpvar)/2;
    }
}