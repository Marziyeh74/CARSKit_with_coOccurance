// Copyright (C) 2015 Yong Zheng
//
// This file is part of CARSKit.
//
// CARSKit is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// CARSKit is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with CARSKit. If not, see <http://www.gnu.org/licenses/>.
//

package carskit.alg.cars.adaptation.independent;

import carskit.generic.TensorRecommender;
import carskit.data.structure.SparseMatrix;
import static carskit.generic.Recommender.rateDao;
import happy.coding.io.Logs;
import java.util.ArrayList;
import java.util.HashMap;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.TensorEntry;

/**
 * CANDECOMP/PARAFAC (CP) Tensor Factorization <br>
 *
 * Shao W., <strong>Tensor Completion</strong> (Section 3.2), Saarland University.
 *
 * Note: This implementation is modified from the algorithm in LibRec
 *
 * @author Yong Zheng
 * 
 *
 */
public class BiasedCPTF extends TensorRecommender {

    // dimension-feature matrices
    private DenseMatrix[] M;

      private DenseVector condBias;
      private double max_pred,min_pred,max_bu,min_bu,max_bi,min_bi,min_bc,max_bc;
      private double max_Md0,max_Md1,max_Md2,min_Md0,min_Md1,min_Md2;
      private DenseMatrix U, V, W;
       private double max_U,max_V,max_W,min_U,min_V,min_W;
    public BiasedCPTF(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) throws Exception {
        super(trainMatrix, testMatrix, fold);
    }

    @Override
    protected void initModel() throws Exception {
        M = new DenseMatrix[numDimensions];

         U = new DenseMatrix(numUsers, numFactors);
        V = new DenseMatrix(numItems, numFactors);
        W = new DenseMatrix(numConditions, numFactors);
       
        
        U.init(1, 0.1);
        V.init(1, 0.1);
        W.init(1, 0.1);
       
        
         System.out.println("************BiasedCPTF");
        System.out.println("numDimensions="+numDimensions);
        System.out.println("numConditions="+numConditions);
       // System.exit(0);
        for (int d = 0; d < numDimensions; d++) {
            M[d] = new DenseMatrix(dimensions[d], numFactors);
            //M[d].init(initMean,initStd);
            //normalize(d);
            M[d].init(1, 0.1); // randomly initialization
        }
         userBias = new DenseVector(numUsers);
        userBias.init(initMean, initStd);

        itemBias = new DenseVector(numItems);
        itemBias.init(initMean, initStd);

        condBias = new DenseVector(numConditions);
        condBias.init(initMean, initStd);
    }

    protected void normalize(int d) {

        // column-wise normalization
        for (int f = 0; f < numFactors; f++) {

            double norm = 0;
            for (int r = 0; r < M[d].numRows(); r++) {
                norm += Math.pow(M[d].get(r, f), 2);
            }
            norm = Math.sqrt(norm);

            for (int r = 0; r < M[d].numRows(); r++) {
                M[d].set(r, f, M[d].get(r, f) / norm);
            }
        }
    }

    @Override
    protected void buildModel() throws Exception {
        for (int iter = 1; iter <=numIters; iter++) {

            // SGD Optimization

            loss = 0;
            for (TensorEntry te : trainTensor) {
                
                calculateMaxMin();
                int[] keys = te.keys();
                 for (int i = 0; i < keys.length; i++) {
                    
                     System.out.print(keys[i]);
                    if (i == 0) {
                       // System.out.print(":" + rateDao.getUserId(keys[i]) + " ,");
                        Logs.debug("user={} ,", rateDao.getUserId(keys[i]));
                    }
                    if (i == 1) {
                      //  System.out.print(":" + rateDao.getItemId(keys[i]) + " ,");
                        Logs.debug("item={} ,", rateDao.getUserId(keys[i]));
                    } else {
                        Logs.debug(" context_dimention={},", rateDao.getUserId(keys[i]));
                    }
                    
                    
                }
                  double rate = te.get();
               
               
                if (rate <= 0)
                    continue;

                double pred = predict(keys);
                double e = rate - pred;

                loss += e * e;

                //userBias
                double bu = userBias.get(keys[0]);
                
                double sgd = e - regB * bu;
                userBias.add(keys[0], lRate * sgd);

                loss += regB * bu * bu;
                
                
               double bj = itemBias.get(keys[1]);
                sgd = e - regB * bj;
                itemBias.add(keys[1], lRate * sgd);

                loss += regB * bj * bj;
                
                HashMap<Integer, ArrayList<Integer>> dimensionConditionsList =rateDao.getDimensionConditionsList();
       
                //int[] conditions=new int[dimensionConditionsList.size()];
                double bc_sum=0;
                  double bc=0;
                for(int i=2; i<keys.length; i++){
                   int cond=dimensionConditionsList.get(i-2).get(keys[i]);
                  // System.out.print("cond"+":"+cond+",");
                     bc = condBias.get(cond);
                    bc_sum+=bc;
                    sgd = e - regC * bc;
                    condBias.add(cond, lRate * sgd);
                }
                 loss += regC * bc_sum;
                
                
                Logs.debug(", rate={} , pred={},max_pred={} , min_pred={}\n", rate, pred,max_pred,min_pred);
                Logs.debug(", bu={}, bi={} , bc={} , bc_sum={}\n", bu, bj, bc, bc_sum);
                
                for (int f = 0; f < numFactors; f++) {

                    
                     
                    double Uf = U.get(keys[0], f);
                    double Vf = V.get(keys[1], f);
                    double Wf = W.get(keys[2], f);
                    
                     sgd = (e * Vf * Wf - reg * Uf);
                    U.add(keys[0], f, sgd * lRate);

                    sgd = (e * Uf * Wf - reg * Vf);
                    V.add(keys[1], f, sgd * lRate);

                      sgd = (Uf * Vf * e - reg * Wf) * lRate;
                    W.add(keys[2], f, sgd);
                    
                     loss += reg * Uf * Uf;
                    loss += reg * Vf * Vf;
                    loss += reg * Wf * Wf;

                    /*
                     for (int dd = 0; dd < numDimensions; dd++) {
                        sgd *= M[dd].get(keys[dd], f);
                    }

                    for (int d = 0; d < numDimensions; d++) {
                        double df = M[d].get(keys[d], f);

                        double gdf = sgd / df * e;
                        M[d].add(keys[d], f, lRate * (gdf - reg * df));

                        loss += reg * df * df;
                    }*/
                }
            }

         //   System.exit(0);
            loss *= 0.5;
            if (isConverged(iter))
                break;
        }
    }


    @Override
    protected double predict(int u, int j, int c) throws Exception {
        double pred = 0;
        int[] keys = getKeys(u,j,c);

        for (int f = 0; f < numFactors; f++) {

            double prod = 1;
            for (int d = 0; d < numDimensions; d++) {
                prod *= M[d].get(keys[d], f);
            }

            pred += prod;
        }

        if (pred > maxRate)
            pred = maxRate;
        if (pred < minRate)
            pred = minRate;

        return pred;
    }

    protected double predict(int[] keys) {
          double u, v, w, pred = 0;

         System.out.println("into predict:");
         for (int f = 0; f < numFactors; f++) {
            u = U.get(keys[0], f);
            v = V.get(keys[1], f);
            w = W.get(keys[2], f);
            pred = pred + (u * v * w);
            ///System.out.println(", f="+f+",U="+U.get(keys[0], f)+" ,V="+V.get(keys[1], f)+", W="+W.get(keys[2], f));
            //Logs.debug("f={},uf={},vf={},wf={}\n", f, u, v, w);
        }
       /* for (int f = 0; f < numFactors; f++) {

            double prod = 1;
            for (int d = 0; d < numDimensions; d++) {
                prod *= M[d].get(keys[d], f);
                //System.out.print(", d="+d+",f="+f+" , val="+M[d].get(keys[d], f));
            }
            System.out.println();

            pred += prod;
        }
        */
        if(pred>max_pred){
             max_pred=pred;
         }
         if(pred<min_pred){
             min_pred=pred;
         }

        return pred;
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
        
         
        
        /*
         max_Md0=Double.MIN_VALUE;
        min_Md0=Double.MAX_VALUE;
        for (int i = 0; i < numUsers; i++) {
            for (int j = 0; j < numFactors; j++) {
                if(M[0].get(i, j)>max_Md0 ){
                    max_Md0=M[0].get(i, j);
                }
                if(M[0].get(i, j)<min_Md0 ){
                    min_Md0=M[0].get(i, j);
                }
            }
        }
         max_Md1=Double.MIN_VALUE;
        min_Md1=Double.MAX_VALUE;
        for (int i = 0; i < numItems; i++) {
            for (int j = 0; j < numFactors; j++) {
                if(M[1].get(i, j)>max_Md1){
                    max_Md1=M[1].get(i, j);
                }
                if(M[1].get(i, j)<min_Md1 ){
                    min_Md1=M[1].get(i, j);
                }
            }
        }
        
           max_Md2=Double.MIN_VALUE;
        min_Md2=Double.MAX_VALUE;
        for (int i = 0; i < numConditions; i++) {
            for (int j = 0; j < numFactors; j++) {
                if(M[2].get(i, j)>max_Md2){
                    max_Md2=M[2].get(i, j);
                }
                if(M[2].get(i, j)<min_Md2 ){
                    min_Md2=M[2].get(i, j);
                }
            }
        }
                */
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
        
                
         Logs.debug("max_U={} , max_V={} , max_W={}\n", max_U,max_V,max_W);
        Logs.debug("min_U={} , min_V={} , min_W={}\n", min_U,min_V,min_W);
        
       // Logs.debug("max_Md0={} , max_Md2={} , max_Md2={}\n", max_Md0,max_Md1,max_Md2);
       // Logs.debug("min_Md0={} , min_Md1={} , min_Md2={}\n", min_Md0,min_Md1,min_Md2);
        Logs.debug("max_bu={} , min_bu={} , max_bi={} , min_bi={} , max_bc={} ,min_bc={}\n", max_bu,min_bu,max_bi,min_bi,max_bc,min_bc);
    }
}
