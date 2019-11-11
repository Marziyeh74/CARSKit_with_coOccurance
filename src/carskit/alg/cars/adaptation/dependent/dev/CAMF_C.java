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

package carskit.alg.cars.adaptation.dependent.dev;

import carskit.alg.cars.adaptation.dependent.CAMF;
import carskit.data.setting.Configuration;
import carskit.data.structure.SparseMatrix;
import carskit.generic.ContextRecommender;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import happy.coding.math.Randoms;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;

/**
 * CAMF_C: Baltrunas, Linas, Bernd Ludwig, and Francesco Ricci. "Matrix factorization techniques for context aware recommendation." Proceedings of the fifth ACM conference on Recommender systems. ACM, 2011.
 * <p></p>
 * Note: in this algorithm, there is a rating deviation for each context condition
 *
 * @author Yong Zheng
 *
 */

public class CAMF_C extends CAMF{



    public CAMF_C(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);
        this.algoName = "CAMF_C";
    }

    protected void initModel() throws Exception {

        super.initModel();

        userBias = new DenseVector(numUsers);
        userBias.init(initMean, initStd);

        itemBias = new DenseVector(numItems);
        itemBias.init(initMean, initStd);

        condBias = new DenseVector(numConditions);
        condBias.init(initMean, initStd);

    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        double pred=globalMean + userBias.get(u) + itemBias.get(j) +DenseMatrix.rowMult(P, u, Q, j);
        for(int cond:getConditions(c)){
            pred+=condBias.get(cond);
        }
        return pred;
    }

    @Override
    protected void buildModel() throws Exception {

        double MF_w=0.3;
        double IE_w=0.7;
        for (int iter = 1; iter <= numIters; iter++) {

           //  System.out.println("iter"+iter);
            loss = 0;
            loss_MF=0;
            loss_itemEmbedding=0;
            for (MatrixEntry me : trainMatrix) {

                int ui = me.row(); // user-item
                int u= rateDao.getUserIdFromUI(ui);
                //System.out.print(u+":"+rateDao.getUserId(u)+",");
                int j= rateDao.getItemIdFromUI(ui);
              //  System.out.print(j+":"+rateDao.getItemId(j)+",");
                int ctx = me.column(); // context
                double rujc = me.get();

                double pred = predict(u, j, ctx, false);
                double euj = rujc - pred;

                
                loss += euj * euj;
               // loss_MF += euj * euj;

                // update factors
                double bu = userBias.get(u);
                
                //positon1
                itemEmbedding(j, u, bu);   // bad reslut
                double sgd = euj - regB * bu;
                userBias.add(u, lRate * sgd);

                loss += regB * bu * bu;
                //loss_MF += regB * bu * bu;


                double bj = itemBias.get(j);
                sgd = euj - regB * bj;
                itemBias.add(j, lRate * sgd);

                loss += regB * bj * bj;
                //loss_MF += regB * bj * bj;

                
                 String context=rateDao.getContextId(ctx);
                 
                 // System.out.print("ctx="+ctx+",context="+context+",");
                double bc_sum=0;
                for(int cond:getConditions(ctx)) {
                    double bc = condBias.get(cond);
                    bc_sum+=bc;
                    sgd = euj - regC * bc;
                    condBias.add(cond, lRate * sgd);
                   // System.out.print(cond+":"+rateDao.getContextConditionId(cond)+",");
                }
               // System.out.println();

                loss += regB * bc_sum;
               // loss_MF += regB * bc_sum;
                
                  

                for (int f = 0; f < numFactors; f++) {
                    double puf = P.get(u, f);
                    double qjf = Q.get(j, f);

                    double delta_u = euj * qjf - regU * puf;
                    double delta_j = euj * puf - regI * qjf;

                    P.add(u, f, lRate * delta_u);
                    Q.add(j, f, lRate * delta_j);

                    loss += regU * puf * puf + regI * qjf * qjf;
                  //  loss_MF += regU * puf * puf + regI * qjf * qjf;
                }
                
                //itemEmbedding(j, u, userBias.get(u));
               
                //loss= MF_w* loss_MF + IE_w* loss_itemEmbedding;
                loss = MF_w* loss + IE_w* loss_itemEmbedding;
                //System.out.println("loss="+loss+" loss_MF"+loss_MF+"loss embedding"+loss_itemEmbedding);
                

            }
          //  System.exit(0);
            loss *= 0.5;

            if (isConverged(iter))
                break;

        }// end of training

    }
    
    protected void itemEmbedding(int i,int u,double bu){
        
      
        double mij;
        // double  loss=0.0;
        //double bu = userBias.get(u);
                
              //positon1  itemEmbedding(j, u, bu);   // bad reslut
               // double sgd = euj - regB * bu;
               
         Collection values = (Collection) rateDao.getCoOccuranceItemsList().get((Integer)i);
        Iterator valuesIterator = values.iterator( );
       
        while( valuesIterator.hasNext( ) ) {
            Integer j=(Integer)valuesIterator.next();
           
            mij= rateDao.getCoOccuranceItems()[i][j];
           
              
          double   eij=mij- loss_itemEmbedding;
             for (int f = 0; f < numFactors; f++) {
                    double beta_i = Q.get(i, f);
                    double landa_j = Q.get(j, f);

                     double delta_i = eij * beta_i - regI * landa_j;
                    double delta_j = eij * landa_j - regI * beta_i;

                   // Q.add(i, f, lRate * delta_i);
                    //Q.add(j, f, lRate * delta_j);
                    
                    loss_itemEmbedding +=(beta_i *landa_j) ;
                    loss_itemEmbedding += regI * beta_i * beta_i + regI * landa_j * landa_j;
                }
             
            loss_itemEmbedding += ( eij - bu);
           
            //double sgd = mij- loss_itemEmbedding - regB * bu;
               // userBias.add(u, lRate * sgd);
                
           
            
        }
        
    }
    
}
