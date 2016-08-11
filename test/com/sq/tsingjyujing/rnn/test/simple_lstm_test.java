/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.sq.tsingjyujing.rnn.test;

import com.sq.tsingjyujing.rnn.LSTM;
import java.util.Random;

public class simple_lstm_test {
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception{
        System.out.println("Test of SimpleLSTM\n");
        Random r = new Random(1234);
        DistractedSequenceRecall task = new DistractedSequenceRecall(r);

        int cell_blocks = 15;
        LSTM slstm = new LSTM(r, task.GetObservationDimension(), task.GetActionDimension(), cell_blocks);

        for (int epoch = 0; epoch < 5000; epoch++) {
            double fit = task.EvaluateFitnessSupervised(slstm);
            if (epoch % 10 == 0)
                System.out.println("["+epoch+"] error = " + (1 - fit)*100+"%");
            if (fit>0.999){
                break;
            }else if(fit<=1e-4){
                System.out.println("Failed.");
                break;
            }
        }
        System.out.println("done.");
        slstm.Display();
    }
    
}
