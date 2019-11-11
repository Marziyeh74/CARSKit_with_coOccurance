/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package carskit.alg.cars.adaptation.independent;

import carskit.data.structure.SparseMatrix;
import static carskit.generic.Recommender.rateDao;
import carskit.generic.TensorRecommender;
import happy.coding.io.Logs;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.TensorEntry;

/**
 * BiasedCPTF with coOccurance tensor
 *
 * @author Marziyeh
 */
public class BiasedCPTF_v2 extends TensorRecommender {

    // dimension-feature matrices
    // M = UVW
    // private DenseMatrix[] M;
    //Z= factor matrix used for coOccuranceTensor
    // U for uer factor matrix,V and Z for Item factor matrix ,W for context factor matrix
    private DenseMatrix U, V, W, Z;

    private DenseVector condBias;

    private final Lock lock = new ReentrantLock();

    private double max_U,max_V,max_W,max_Z,min_U,min_V,min_W,min_Z;
    private double max_pred,min_pred,max_bu,min_bu,max_bi,min_bi,min_bc,max_bc;
    public BiasedCPTF_v2(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) throws Exception {
        super(trainMatrix, testMatrix, fold);
    }

    @Override
    protected void initModel() throws Exception {
        // M = new DenseMatrix[numDimensions];
        U = new DenseMatrix(numUsers, numFactors);
        V = new DenseMatrix(numItems, numFactors);
        W = new DenseMatrix(numConditions, numFactors);
        Z = new DenseMatrix(numItems, numFactors);

        
        U.init(1, 0.1);
        V.init(1, 0.1);
        W.init(1, 0.1);
        Z.init(1, 0.1);

        System.out.println("************BiasedCPTF_v2_initmodel");
        Logs.debug("************BiasedCPTF_v2_initmodel\n");
        //System.out.println("numDimensions=" + numDimensions);
        //System.out.println("numConditions=" + numConditions);
        // System.exit(0);

        userBias = new DenseVector(numUsers);
        userBias.init(initMean, initStd);

        itemBias = new DenseVector(numItems);
        itemBias.init(initMean, initStd);

        condBias = new DenseVector(numConditions);
        condBias.init(initMean, initStd);
        
        max_pred=Double.MIN_VALUE;
        min_pred=Double.MAX_VALUE;
    }

    protected void normalize() {

        System.out.println("*********normalize");
        // column-wise normalization
        for (int f = 0; f < numFactors; f++) {

            double norm_U = 0, norm_V = 0, norm_W = 0, norm_Z = 0;
            //U
            for (int r = 0; r < U.numRows(); r++) {
                norm_U += Math.pow(U.get(r, f), 2);
            }
            norm_U = Math.sqrt(norm_U);

            for (int r = 0; r < U.numRows(); r++) {
                U.set(r, f, U.get(r, f) / norm_U);
            }
            //V
            for (int r = 0; r < V.numRows(); r++) {
                norm_V += Math.pow(V.get(r, f), 2);
            }
            norm_V = Math.sqrt(norm_V);

            for (int r = 0; r < V.numRows(); r++) {
                V.set(r, f, V.get(r, f) / norm_V);
            }
            //W
            for (int r = 0; r < W.numRows(); r++) {
                norm_W += Math.pow(W.get(r, f), 2);
            }
            norm_W = Math.sqrt(norm_W);

            for (int r = 0; r < W.numRows(); r++) {
                W.set(r, f, W.get(r, f) / norm_W);
            }
            //z
            for (int r = 0; r < Z.numRows(); r++) {
                norm_Z += Math.pow(Z.get(r, f), 2);
            }
            norm_Z = Math.sqrt(norm_Z);

            for (int r = 0; r < Z.numRows(); r++) {
                Z.set(r, f, Z.get(r, f) / norm_Z);
            }
        }
    }
    
    protected void normalize_minMax() {

      
        System.out.println("*********normalize");
        // column-wise normalization
        for (int f = 0; f < numFactors; f++) {

            double max_U = Double.MIN_VALUE, max_V = Double.MIN_VALUE, max_W = Double.MIN_VALUE, max_Z = Double.MIN_VALUE;
            double min_U=Double.MAX_VALUE, min_V=Double.MAX_VALUE , min_W=Double.MAX_VALUE , min_Z=Double.MAX_VALUE;
            //U
            for (int r = 0; r < U.numRows(); r++) {
                if(U.get(r, f)>max_U){
                    max_U=U.get(r, f);
                }
                if(U.get(r, f)<min_U){
                    min_U=U.get(r, f);
                }
            }
            
            for (int r = 0; r < U.numRows(); r++) {
                U.set(r, f, (U.get(r, f)-min_U)/(max_U-min_U));
            }
            //V
            for (int r = 0; r < V.numRows(); r++) {
                if(V.get(r, f)>max_V){
                    max_V=V.get(r, f);
                }
                if(V.get(r, f)<min_V){
                    min_V=V.get(r, f);
                }
            }
           

            for (int r = 0; r < V.numRows(); r++) {
                V.set(r, f, (V.get(r, f)-min_V)/(max_V-min_V));
            }
            //W
            for (int r = 0; r < W.numRows(); r++) {
                 if( W.get(r, f)>max_W){
                    max_W=W.get(r, f);
                }
                if( W.get(r, f)<min_W){
                    min_W=W.get(r, f);
                }
            }
           

            for (int r = 0; r < W.numRows(); r++) {
                W.set(r, f, (W.get(r, f)-min_W)/(max_W-min_W));
            }
            //z
            for (int r = 0; r < Z.numRows(); r++) {
                 if( Z.get(r, f)>max_Z){
                    max_Z=Z.get(r, f);
                }
                if( Z.get(r, f)<min_Z){
                    min_Z=Z.get(r, f);
                }
            }
           

            for (int r = 0; r < Z.numRows(); r++) {
                Z.set(r, f, (Z.get(r, f)-min_Z)/(max_Z-min_Z));
            }
        }
    }


    @Override
    protected void buildModel() throws Exception {

        Logs.debug("************BiasedCPTF_v2_buildmodel\n");
        for (int iter = 1; iter <= numIters; iter++) {

            Logs.debug("=> iter={}:\n\n", iter);
            // SGD Optimization
            loss = 0;
            for (TensorEntry te : trainTensor) {
                
           calculateMaxMin();

                Logs.debug("--keys : --\n");
                int[] keys = te.keys();
                for (int i = 0; i < keys.length; i++) {

                   // System.out.print(keys[i]);
                    if (i == 0) {
                        //System.out.print(":" + rateDao.getUserId(keys[i]) + " ,");
                        Logs.debug("user={} ,", rateDao.getUserId(keys[i]));
                    }
                    if (i == 1) {
                        //System.out.print(":" + rateDao.getItemId(keys[i]) + " ,");
                        Logs.debug("item={} ,", rateDao.getUserId(keys[i]));
                    } else {
                        Logs.debug(" context_dimention={},", rateDao.getUserId(keys[i]));
                    }

                }
                double rate = te.get();
              //  System.out.print(", rate=" + rate);

                if (rate <= 0) {
                    continue;
                }

                double pred = predict(keys);
                double e = rate - pred;

                loss += e * e;
                double bu = userBias.get(keys[0]);
                double bi = itemBias.get(keys[1]);

                loss += regB * bu * bu;
                loss += regB * bi * bi;
                HashMap<Integer, ArrayList<Integer>> dimensionConditionsList = rateDao.getDimensionConditionsList();

                //int[] conditions=new int[dimensionConditionsList.size()];
                double bc_sum = 0, sgd, bc = 0.0;
                for (int i = 2; i < keys.length; i++) {
                    int cond = dimensionConditionsList.get(i - 2).get(keys[i]);
                    System.out.print("cond" + ":" + cond + ",");
                    bc = condBias.get(cond);
                    bc_sum += bc * bc;
                    sgd = lRate * (e - regC * bc);
                  //  Logs.debug(", cond={} , sgd_cond={}\n", cond, sgd);
                    condBias.add(cond, sgd);
                }
                loss += regC * bc_sum;

                Logs.debug(", rate={} , pred={}\n", rate, pred);
                Logs.debug(", bu={}, bi={} , bc={} , bc_sum={}\n", bu, bi, bc, bc_sum);
                double co_occurance_sum = 0.0, UVZ = 0.0;

                double[] co_occurance_val_ARRAY = new double[numItems];
                // calculate sgd of co-occurance
                Logs.debug("**calculate sgd of co-occurance**\n");
                for (int j = 0; j < rateDao.numItems(); j++) {

                    if (rateDao.getCoOccuranceTensor()[keys[1]][j][keys[0]] != 0) {
                        for (int f = 0; f < numFactors; f++) {
                           

                                UVZ += U.get(keys[0], f) * V.get(keys[1], f) * Z.get(j, f);

                            
                          //  Logs.debug("- j={},f={},Uf={},Vf={},Zf={},UVZ={}\n", j, f,
                                //  U.get(keys[0], f), V.get(keys[1], f), Z.get(j, f), UVZ);

                            if (UVZ == Double.POSITIVE_INFINITY || UVZ==Double.NEGATIVE_INFINITY) {
                                Logs.debug("user={} , item={},context_dimention={}, rate={},pred={}, bu={},bi={}, bc={} , bc_sum={} ,j={} , f={} , U={},V={},Z={}",
                                        keys[0], keys[1], keys[2], rate, pred, bu, bi, bc, bc_sum, j, f, U.get(keys[0], f), V.get(keys[1], f), Z.get(j, f));
                                System.out.print("infinitiy=" + Double.POSITIVE_INFINITY + " , -infinity=" + Double.NEGATIVE_INFINITY + ", min-value=" + Double.MIN_VALUE
                                        + ", max-value=" + Double.MAX_VALUE+"\n");
                                UVZ = Double.MAX_VALUE;
                                System.exit(0);

                            }
                        }
                        lock.lock();
                        try {

                            co_occurance_val_ARRAY[j] = (rateDao.getCoOccuranceTensor()[keys[1]][j][keys[0]]
                                    - UVZ - bu - itemBias.get(j) - bc);
                          //  Logs.debug("\n - j={},co_occurance_val_j={}\n", j, co_occurance_val_ARRAY[j]);
                            co_occurance_sum += co_occurance_val_ARRAY[j];
                        } finally {
                            lock.unlock();
                        }
                    }

                }
                loss += co_occurance_sum * co_occurance_sum;
               //Logs.debug(" co_occurance_sum={},", co_occurance_sum);
                //userBias

                sgd = (e + co_occurance_sum - regB * bu) * lRate;
                userBias.add(keys[0], sgd);
               // Logs.debug(" sgd_bu={},", sgd);

                sgd = (e + co_occurance_sum - regB * bi) * lRate;
                itemBias.add(keys[1], sgd);

               // Logs.debug(" sgd_bi={},", sgd);

              //  System.out.print(", pred=" + pred);
                System.out.println();

               // Logs.debug("**calculating SGD for U,V,W,Z**\n");
                for (int f = 0; f < numFactors; f++) {

                    double sgd_U = 0.0, sgd_V = 0.0;

                    double Uf = U.get(keys[0], f);
                    double Vf = V.get(keys[1], f);
                    double Wf = W.get(keys[2], f);

                    loss += reg * Uf * Uf;
                    loss += reg * Vf * Vf;
                    loss += reg * Wf * Wf;

                    //sgd for W
                    double sgd_W = (Uf * Vf * e - reg * Wf) * lRate;
                    W.add(keys[2], f, sgd_W);

                    //sgd for U,V is in for-loop of co-occurance-array
                    for (int j = 0; j < co_occurance_val_ARRAY.length; j++) {

                        double Zf = Z.get(j, f);
                        double bj = itemBias.get(j);

                        sgd_U = sgd_U + (Vf * Zf * co_occurance_val_ARRAY[j]);
                        sgd_V = sgd_V + (Uf * Zf * co_occurance_val_ARRAY[j]);

                        double sgd_z = (Uf * Vf * co_occurance_val_ARRAY[j] - reg * Zf) * lRate;
                        double sgd_bj = (co_occurance_val_ARRAY[j] - regB * bj) * lRate;
                        Z.add(j, f, sgd_z);
                        itemBias.add(j, sgd_bj);

                        loss += reg * Zf * Zf;
                        loss += regB * bj * bj;
                    //   Logs.debug("- j={},Zf={},bj={},sgd_Uf={},sgd_Vf={}, sgd_Zf={},sgd_bj={}\n",
                               // j, Zf, bj, sgd_U, sgd_V, sgd_z, sgd_bj);

                    }
                    sgd_U += (e * Vf * Wf - reg * Uf);
                    U.add(keys[0], f, sgd_U * lRate);

                    sgd_V += (e * Uf * Wf - reg * Vf);
                    V.add(keys[1], f, sgd_V * lRate);

                   // Logs.debug("f={},Uf={},Vf={},Wf={},sgd_Uf={},sgd_Vf={},sgd_Wf={}\n", f, Uf, Vf, Wf, sgd_U, sgd_V, sgd_W);

                }

            }

            //   System.exit(0);
            loss *= 0.5;
            if (isConverged(iter)) {
                break;
            }
        }
    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        double pred = 0;
        int[] keys = getKeys(u, j, c);

        for (int f = 0; f < numFactors; f++) {
            pred += (U.get(keys[0], f) * V.get(keys[1], f) * W.get(keys[2], f));
        }

        pred += itemBias.get(keys[1]) + userBias.get(keys[0]);

        HashMap<Integer, ArrayList<Integer>> dimensionConditionsList = rateDao.getDimensionConditionsList();

        double bc_sum = 0.0;
        for (int i = 2; i < keys.length; i++) {
            int cond = dimensionConditionsList.get(i - 2).get(keys[i]);
            // System.out.println("cond" + ":" + cond + ",");
            double bc = condBias.get(cond);
            bc_sum += bc * bc;

        }
        pred += bc_sum + globalMean;
        if (pred > maxRate) {
            pred = maxRate;
        }
        if (pred < minRate) {
            pred = minRate;
        }

        return pred;
    }

    protected double predict(int[] keys) {
        double u, v, w, pred = 0;

        /*  for (int f = 0; f < numFactors; f++) {

         double prod = 1;
         for (int d = 0; d < numDimensions; d++) {
         prod *= M[d].get(keys[d], f);
         }

         pred += prod;
         }*/
        Logs.debug("***calculate Predict");
        normalize_minMax();
        for (int f = 0; f < numFactors; f++) {
            u = U.get(keys[0], f);
            v = V.get(keys[1], f);
            w = W.get(keys[2], f);
            pred = pred + (u * v * w);
            ///System.out.println(", f="+f+",U="+U.get(keys[0], f)+" ,V="+V.get(keys[1], f)+", W="+W.get(keys[2], f));
            //Logs.debug("f={},uf={},vf={},wf={}\n", f, u, v, w);
        }

          pred+=itemBias.get(keys[1])+ userBias.get(keys[0]);
        
         HashMap<Integer, ArrayList<Integer>> dimensionConditionsList = rateDao.getDimensionConditionsList();

         double bc_sum=0.0;
         for (int i = 2; i < keys.length; i++) {
         int cond = dimensionConditionsList.get(i - 2).get(keys[i]);
         //  System.out.println("cond" + ":" + cond + ",");
         double bc = condBias.get(cond);
         bc_sum += bc * bc;
                   
         }
         pred+=bc_sum+globalMean;
         
         if(pred>max_pred){
             max_pred=pred;
         }
         if(pred<min_pred){
             min_pred=pred;
         }
         
         Logs.debug("max_pred={} , min_pred={}\n", max_pred,min_pred);
        return pred;
    }
    
    public double checkInfinityOrNan(double val){
       if(val==Double.POSITIVE_INFINITY){
          val=Double.POSITIVE_INFINITY - 1000; 
          System.out.println("Positive_Infinitiy");
       }
       else if(val==Double.NEGATIVE_INFINITY){
            System.out.println("negetive_infinity");
        val=Double.NEGATIVE_INFINITY + 1000;
       
      /* if(Double.isNaN(val)){
        
           
       }*/
    }
    return val;
    }
    
    public void calculateMaxMin(){
      
        //U
        max_U=Double.MIN_VALUE;
        min_U=Double.MAX_VALUE;
        for (int i = 0; i < numUsers; i++) {
            for (int j = 0; j < numFactors; j++) {
                if(U.get(i, j)>max_U ){
                    max_U=U.get(i, j);
                }
                if(U.get(i, j)<min_U ){
                    min_U=U.get(i, j);
                }
            }
        }
        
        
         max_V=Double.MIN_VALUE;
         min_V=Double.MAX_VALUE;
        for (int i = 0; i < numItems; i++) {
            for (int j = 0; j < numFactors; j++) {
                if(V.get(i, j)>max_V ){
                    max_V=V.get(i, j);
                }
                if(V.get(i, j)<min_V ){
                    min_V=V.get(i, j);
                }
            }
        }
        
        
        
         max_W=Double.MIN_VALUE;
         min_W=Double.MAX_VALUE;
        for (int i = 0; i < numConditions; i++) {
            for (int j = 0; j < numFactors; j++) {
                if(W.get(i, j)>max_W ){
                    max_W=W.get(i, j);
                }
                 if(W.get(i, j)<min_W ){
                    min_W=W.get(i, j);
                }
            }
        }
        
          max_Z=Double.MIN_VALUE;
          min_Z=Double.MAX_VALUE;
        for (int i = 0; i < numItems; i++) {
            for (int j = 0; j < numFactors; j++) {
                if(Z.get(i, j)>max_Z ){
                    max_Z=Z.get(i, j);
                }
            }
        }
        
        max_bu=Double.MIN_VALUE;
        max_bi=Double.MIN_VALUE;
        max_bc=Double.MIN_VALUE;
        
        min_bu=Double.MIN_VALUE;
        min_bi=Double.MIN_VALUE;
        min_bc=Double.MIN_VALUE;
        
        for (int i = 0; i < numItems; i++) {
            if(itemBias.get(i)>max_bi){
                max_bi=itemBias.get(i);
            }
            if(itemBias.get(i)<min_bi){
                min_bi=itemBias.get(i);
            }
        }
        
          for (int i = 0; i < numUsers; i++) {
            if(userBias.get(i)>max_bu){
                max_bu=userBias.get(i);
            }
            if(userBias.get(i)<min_bu){
                min_bu=userBias.get(i);
            }
        }
          
            for (int i = 0; i < numConditions; i++) {
            if(condBias.get(i)>max_bc){
                max_bu=condBias.get(i);
            }
            if(condBias.get(i)<min_bc){
                min_bc=condBias.get(i);
            }
        }
        
                
        
        
        Logs.debug("max_U={} , max_V={} , max_W={} , max_Z={}\n", max_U,max_V,max_W,max_Z);
        Logs.debug("min_U={} , min_V={} , min_W={} , min_Z={}\n", min_U,min_V,min_W,min_Z);
        Logs.debug("max_bu={} , min_bu={} , max_bi={} , min_bi={} , max_bc={} ,min_bc={}\n", max_bu,min_bu,max_bi,min_bi,max_bc,min_bc);
    }

}
