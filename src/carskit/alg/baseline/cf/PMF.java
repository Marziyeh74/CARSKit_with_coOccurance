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
package carskit.alg.baseline.cf;

import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import carskit.data.structure.SparseMatrix;
import carskit.generic.IterativeRecommender;
import static carskit.generic.Recommender.rateDao;
import java.util.ArrayList;
import java.util.Collection;

import java.util.Iterator;

/**
 * PMF: Mnih, Andriy, and Ruslan Salakhutdinov. "Probabilistic matrix
 * factorization." Advances in neural information processing systems. 2007.
 * <p>
 * </p>
 * Note: This implementation is modified from the algorithm in LibRec-v1.3
 *
 * @author Yong Zheng
 *
 */
public class PMF extends IterativeRecommender {

    public PMF(SparseMatrix rm, SparseMatrix tm, int fold) {

        super(rm, tm, fold);
        this.algoName = "PMF";
    }

    protected void initModel() throws Exception {

        super.initModel();

        DenseVector itemBias_Wi = new DenseVector(numItems);
        DenseVector itemBias_Cj = new DenseVector(numItems);

        itemBias_Wi.init(initMean, initStd);
        //  itemBias_Wi.setAll(0);
        itemBias_Cj.init(initMean, initStd);
    }

    @Override
    protected void buildModel() throws Exception {

        float cui=(float) 0.1;
        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;
            for (MatrixEntry me : train) {

                int u = me.row(); // user
                int i = me.column(); // item
                double rui = me.get();

                double pui = predict(u, i, -1, false);
                double eui = rui - pui;

                loss += cui*eui * eui;
                
                //updaing biases wi , cj;
                 ArrayList j_index_list = getJListForI(i);
                  double sgd_w = 0.0;
                for (int j = 1; j < j_index_list.size(); j++) {
                        double mij = rateDao.getCoOccuranceItems()[i][j];
                        
                        double cj = itemBias.get(j);
                        double prod_q_y = mult_item_latent_factors(i, j);
                        double item_embedding=mij - prod_q_y - itemBias.get(i) - itemBias.get(j);
                       

                        sgd_w = sgd_w + (item_embedding);
                        
                          //updating cj
                           itemBias.add(j, lRate * item_embedding);
                           loss+=item_embedding;
                        
                    }
                
                    //updating wi
                    itemBias.add(i, lRate * sgd_w);


                // update factors
                for (int f = 0; f < numFactors; f++) {
                    double puf = P.get(u, f);
                    double qif = Q.get(i, f);
                    double wi = itemBias.get(i);

                    double sgd_P = lRate * (eui * qif - regU * puf);
                    P.add(u, f, sgd_P);
                   // Q.add(j, f, lRate * (euj * puf - regI * qjf));

                    //updating qif,yjf:
                   
                    double sgd_q = eui * puf - regI * qif;
                   
                    for (int j = 1; j < j_index_list.size(); j++) {
                        double mij = rateDao.getCoOccuranceItems()[i][j];
                        double yjf = Q.get(j, f);
                        double cj = itemBias.get(j);
                        double prod_q_y = mult_item_latent_factors(i, j);
                        double item_embedding=mij - prod_q_y - itemBias.get(i) - itemBias.get(j);
                        sgd_q = sgd_q + yjf * item_embedding;

                        double sgd_y=lRate * (qif*item_embedding-reg_lam_gamma*yjf);
                        sgd_w = sgd_w + (item_embedding);
                        //updaing yjf
                          Q.add(j, f, sgd_y);
                          //updating cj
                           itemBias.add(j, lRate * item_embedding);
                           loss+= reg_lam_gamma*yjf*yjf;
                        
                    }
                    
                    //updating qif
                    Q.add(i, f, lRate * sgd_q);
                   
                    loss += regU * puf * puf + regI * qif * qif;
                }

            }

            loss *= 0.5;

            if (isConverged(iter)) {
                break;
            }

        }// end of training

    }

    public double mult_item_latent_factors(int i, int j) {

        double mult = 0.0;
        for (int f = 0; f < numFactors; f++) {

            double qif = Q.get(i, f);
            double yjf = Q.get(j, f);
            mult = mult + qif * yjf;
        }
        return mult;
    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        if (isUserSplitting) {
            u = userIdMapper.contains(u, c) ? userIdMapper.get(u, c) : u;
        }
        if (isItemSplitting) {
            j = itemIdMapper.contains(j, c) ? itemIdMapper.get(j, c) : j;
        }
        return predict(u, j);
    }

    @Override
    protected double predict(int u, int j) throws Exception {

        return DenseMatrix.rowMult(P, u, Q, j);
    }

    //get row i  from rateDao.getCoOccuranceItems() that mij!=0
    protected ArrayList getJListForI(int i) {

        ArrayList j_list = new ArrayList();

        Collection values = (Collection) rateDao.getCoOccuranceItemsList().get((Integer) i);
        Iterator valuesIterator = values.iterator();

        while (valuesIterator.hasNext()) {
            Integer j = (Integer) valuesIterator.next();

            j_list.add(j);

        }
        return j_list;

    }
}
