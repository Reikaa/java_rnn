package com.sq.tsingjyujing.rnn.neurons;

public abstract class Neuron {
    public static Neuron Factory(neuron_types neuron_type) {
        if (null != neuron_type) switch (neuron_type) {
            case Sigmoid:
                return new sigmoid_neuron();
            case Linear:
                return new linear_neuron();
            case Tanh:
                return new tanh_neuron();
            case ReLU:
                return new relu_neuron();
            case Satur:
                return new satur_neuron();
            case Elliotsig:
                return new elliotsig_neuron();
            case Elliot2sig:
                return new elliot2sig_neuron();
            case Halfelliotsig:
                return new halfelliotsig_neuron();
            case Fast_sigmoid_neuron:
                return new fast_sigmoid_neuron();    
            default:
                System.out.println("ERROR: unknown neuron type");
                break;
        }
        return null;
    }
    
    public static Neuron Factory(String neuron_type) {
        neuron_type = neuron_type.toLowerCase();
        if (null != neuron_type) switch (neuron_type) {
            case "sigmoid":
                return new sigmoid_neuron();
            case "linear":
                return new linear_neuron();
            case "tanh":
                return new tanh_neuron();
            case "relu":
                return new relu_neuron();
            case "satur":
                return new satur_neuron();
            case "elliotsig":
                return new elliotsig_neuron();
            case "elliot2sig":
                return new elliot2sig_neuron();
            case "halfelliotsig":
                return new halfelliotsig_neuron();
            case "fast_sigmoid_neuron":
                return new fast_sigmoid_neuron();    
            default:
                //默认使用Sigmoid函数，都使用ReLU会死的，真的
                break;
        }
        return new sigmoid_neuron();
    }
    
    abstract public double Activate(double x); // 基活(GayPlay)函数
    abstract public double Derivative(double x); // 导数
}
