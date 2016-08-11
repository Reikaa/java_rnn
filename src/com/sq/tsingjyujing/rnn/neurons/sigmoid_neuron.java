package com.sq.tsingjyujing.rnn.neurons;

public class sigmoid_neuron extends Neuron {
  
    @Override
    public double Activate(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double Derivative(double x) {
        double act = Activate(x);
        return act * (1 - act);
    }

}
