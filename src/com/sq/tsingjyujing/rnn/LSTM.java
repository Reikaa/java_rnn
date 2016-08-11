package com.sq.tsingjyujing.rnn;

import com.sq.tsingjyujing.rnn.neurons.neuron_types;
import com.sq.tsingjyujing.rnn.neurons.Neuron;
import java.util.*;

public class LSTM implements IRNN {
    private double init_weight_range = 0.1;
    private int full_input_dimension;
    private int output_dimension;
    private int cell_blocks;
    private Neuron F;
    private Neuron G;

    private double [] context;

    public double [][] weightsF;
    public double [][] weightsG;
    public double [][] weightsOut;

    public double [][] dSdF;
    public double [][] dSdG;

    private neuron_types neuron_type_F = neuron_types.Fast_sigmoid_neuron;
    private neuron_types neuron_type_G = neuron_types.Fast_sigmoid_neuron;

    private double SCALE_OUTPUT_DELTA = 1.0;

    public static double learningRate = 0.1;
    
    public LSTM(Random r, int input_dimension, int output_dimension, int cell_blocks) {
        this.output_dimension = output_dimension;
        this.cell_blocks = cell_blocks;

        context = new double[cell_blocks];

        full_input_dimension = input_dimension + cell_blocks + 1; //+1 for bias

        F = Neuron.Factory(neuron_type_F);
        G = Neuron.Factory(neuron_type_G);

        weightsF = new double[cell_blocks][full_input_dimension];
        weightsG = new double[cell_blocks][full_input_dimension];

        dSdF = new double[cell_blocks][full_input_dimension];
        dSdG = new double[cell_blocks][full_input_dimension];

        for (int i = 0; i < full_input_dimension; i++) {
            for (int j = 0; j < cell_blocks; j++) {
                weightsF[j][i] = (r.nextDouble() * 2 - 1) * init_weight_range;
                weightsG[j][i] = (r.nextDouble() * 2 - 1) * init_weight_range;
            }
        }

        weightsOut = new double[output_dimension][cell_blocks + 1];

        for (int j = 0; j < cell_blocks + 1; j++) {
            for (int k = 0; k < output_dimension; k++){
                weightsOut[k][j] = (r.nextDouble() * 2 - 1) * init_weight_range;
            }
        }
    }

    public void set_neuron_type(String TypeF, String TypeG){
        F = Neuron.Factory(TypeF);
        G = Neuron.Factory(TypeG);
    }
    
    @Override
    public void Reset() {
        // 把上下文的单元清空
        for (int c = 0; c < context.length; c++){
            context[c] = 0.0;
        }
        // 把偏导数清空
        for (int c = 0; c < cell_blocks; c++) {
            for (int i = 0; i < full_input_dimension; i++) {
                this.dSdG[c][i] = 0;
                this.dSdF[c][i] = 0;
            }
        }
    }

    @Override
    public double[] Next(double[] input){
        return Next(input, null);
    }

    @Override
    public double[] Next(double[] input, double[] target_output) {

        //setup input vector
        double[] full_input = new double[full_input_dimension];
        int loc = 0;
        for (int i = 0; i < input.length; i++) {
                full_input[loc++] = input[i];
        }
        for (int c = 0; c < context.length; c++) {
                full_input[loc++] = context[c];
        }
        full_input[loc++] = 1.0; //bias
        //cell block arrays
        double[] sumF = new double[cell_blocks];
        double[] actF = new double[cell_blocks];
        double[] sumG = new double[cell_blocks];
        double[] actG = new double[cell_blocks];
        double[] actH = new double[cell_blocks];

        //inputs to cell blocks
        //sumF=weightsF*full_input
        //sumG=weightsG*full_input
        for (int i = 0; i < full_input_dimension; i++) {
            for (int j = 0; j < cell_blocks; j++) {
                sumF[j] += weightsF[j][i] * full_input[i];
                sumG[j] += weightsG[j][i] * full_input[i];
            }
        }

        for (int j = 0; j < cell_blocks; j++) {
            actF[j] = F.Activate(sumF[j]);
            actG[j] = G.Activate(sumG[j]);
            actH[j] = actF[j] * context[j] + (1 - actF[j]) * actG[j];
        }

        //prepare hidden layer plus bias
        double [] full_hidden = new double[cell_blocks + 1];
        loc = 0;
        for (int j = 0; j < cell_blocks; j++){
            full_hidden[loc++] = actH[j];
        }
        full_hidden[loc++] = 1.0; //bias

        //calculate output
        double[] output = new double[output_dimension];
        for (int k = 0; k < output_dimension; k++) {
            for (int j = 0; j < cell_blocks + 1; j++) {
                output[k] += weightsOut[k][j] * full_hidden[j];
            }
            //output not squashed
        }

        //*****反馈算法开始*****

        //scale partials
        for (int j = 0; j < cell_blocks; j++) {
            double f = actF[j];
            double df = F.Derivative(sumF[j]);
            double g = actG[j];
            double dg = G.Derivative(sumG[j]);
            double context_j = context[j]; //prev value of h
            for (int i = 0; i < full_input_dimension; i++) {
                double prevdSdF = dSdF[j][i];
                double prevdSdG = dSdG[j][i];
                double in = full_input[i];
                dSdG[j][i] = ((1 - f)*dg*in) + (f*prevdSdG);
                dSdF[j][i] = ((context_j- g)*df*in) + (f*prevdSdF);
            }
        }

        if (target_output != null) {

            //output to hidden
            double [] deltaOutput = new double[output_dimension];
            double [] deltaH = new double[cell_blocks];
            for (int k = 0; k < output_dimension; k++) {
                deltaOutput[k] = (target_output[k] - output[k]) * SCALE_OUTPUT_DELTA;
                for (int j = 0; j < cell_blocks; j++) {
                    deltaH[j] += deltaOutput[k] * weightsOut[k][j];
                    weightsOut[k][j] += deltaOutput[k] * actH[j] * learningRate;
                }
                //bias
                weightsOut[k][cell_blocks] += deltaOutput[k] * 1.0 * learningRate;
            }

            //input to hidden
            for (int j = 0; j < cell_blocks; j++) {
                for (int i = 0; i < full_input_dimension; i++) {
                    weightsF[j][i] += deltaH[j] * dSdF[j][i] * learningRate;
                    weightsG[j][i] += deltaH[j] * dSdG[j][i] * learningRate;
                }
            }
        }

        //*****反馈算法结束*****
        
        //roll-over context to next time step
        System.arraycopy(actH, 0, context, 0, cell_blocks);

        //give results
        return output;
    }
    
    public void Display() {
        //输出维度相关的信息
        System.out.println("Dimensional Information");
        System.out.println("Full input dimension: " + full_input_dimension);
        System.out.println("Output dimension: " + output_dimension);
        System.out.println("Cell dimension: " + cell_blocks);
        //输出神经元类型
        System.out.println("Neural Information");
        System.out.println("Neuron F: " + neuron_type_F);
        System.out.println("Neuron G: " + neuron_type_G);
        //输出F矩阵
        System.out.println("Matrix F:");
        for (int i=0; i<cell_blocks; i++){
            for (int j=0; j<full_input_dimension; j++){
                System.out.print(weightsF[i][j]+",");
            }
            System.out.println();
        }
        //输出G矩阵
        System.out.println("Matrix G:");
        for (int i=0; i<cell_blocks; i++){
            for (int j=0; j<full_input_dimension; j++){
                System.out.print(weightsG[i][j]+",");
            }
            System.out.println();
        }
        //输出O矩阵
        System.out.println("Matrix O:");
        for (int i=0; i<output_dimension; i++){
            for (int j=0; j<(cell_blocks + 1); j++){
                System.out.print(weightsOut[i][j]+",");
            }
            System.out.println();
        }
    }
}


