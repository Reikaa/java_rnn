/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.sq.tsingjyujing.rnn.neurons;

/**
 *
 * @author yuanyifan
 */
public class fast_sigmoid_neuron extends Neuron{
    private double [] sigmoid_value;
    private double [] dsigmoid_value;
    private int sample_count = 3000;
    private double upbound = 15.0;
    private double downbound = -15.0;
    private boolean is_inited = false;
    
    public void initialize(){
        //Initialization
        sigmoid_value = new double [sample_count];
        dsigmoid_value = new double [sample_count];
        double range = upbound - downbound;
        double drange = range/sample_count;
        for (int i = 0; i<sample_count; i++) {
            double x = drange*i + downbound;
            x = 1/(1+Math.exp(-x));
            sigmoid_value[i] = x;
            dsigmoid_value[i] = x*(1-x);
        }
        this.is_inited = true;
        // Debug Output: System.out.println("Fast Sigmoid Initialized.");
    }
    
    public int val2index(double x){
        double psd_index = sample_count*(x - downbound)/(upbound - downbound);
        psd_index = Math.round(psd_index);
        if (psd_index<0) {
            return 0;
        }else if(psd_index>=sample_count){
            return sample_count-1;
        }else{
            return (int)psd_index;
        }
    }
    
    @Override
    public double Activate(double x) {
        if (!is_inited){
            initialize();
        }
        if (x>upbound){
            return(1.0);
        }else if(x<downbound){
            return(0.0);
        }else{
            return(sigmoid_value[val2index(x)]);
        }
    }
    
    @Override
    public double Derivative(double x) {
        if (!is_inited){
            initialize();
        }
        if (x>upbound || x<downbound){
            return(0.0);
        }else{
            return(dsigmoid_value[val2index(x)]);
        }
    }
    
    
}
