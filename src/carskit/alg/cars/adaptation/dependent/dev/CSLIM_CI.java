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

import carskit.data.setting.Configuration;
import carskit.data.structure.DenseVector;
import carskit.data.structure.SparseMatrix;
import carskit.generic.IterativeRecommender;
import carskit.alg.cars.adaptation.dependent.CSLIM;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import happy.coding.io.Lists;
import happy.coding.io.Logs;
import happy.coding.io.Strings;
import librec.data.*;

import java.util.*;
import java.util.Map.Entry;

/**
 * CSLIM_CI: Zheng, Yong, Bamshad Mobasher, and Robin Burke. "CSLIM: Contextual slim recommendation algorithms." Proceedings of the 8th ACM Conference on Recommender Systems. ACM, 2014.
 * <p></p>
 * Note: in this algorithm, there is a rating deviation for each pair of item and context condition; and it is built upon SLIM-I algorithm
 *
 * @author Yong Zheng
 *
 */

@Configuration("binThold, knn, regLw2, regLw1, regLc2, regLc1, similarity, iters, rc")
public class CSLIM_CI extends CSLIM {
    private DenseMatrix W;

    // item's nearest neighbors for kNN > 0
    private Multimap<Integer, Integer> itemNNs;

    // item's nearest neighbors for kNN <=0, i.e., all other items
    private List<Integer> allItems;

    public CSLIM_CI(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isRankingPred = true;
        
        isCARSRecommender=false; // this option is used to allow the algorithm to call 2D rating matrix "train"
        this.algoName = "CSLIM_CI";

        regLw1 = algoOptions.getFloat("-lw1");
        regLw2 = algoOptions.getFloat("-lw2");
        regLc1 = algoOptions.getFloat("-lc1");
        regLc2 = algoOptions.getFloat("-lc2");

        knn = algoOptions.getInt("-k");
        als = algoOptions.getInt("-als");
        
        //print();
    }

    @Override
    protected void initModel() throws Exception {
        super.initModel();

        /*ciDev = new DenseMatrix(numItems, numConditions);
       ciDev.init();*/

         ciDev = new DenseMatrix(rateDao.getCoOccuranceContextItems(), numConditions, numConditions);

        
        
      //  W = new DenseMatrix(numItems, numItems);
        //W.init(); // initial guesses: make smaller guesses (e.g., W.init(0.01)) to speed up training

               
        W= new DenseMatrix(rateDao.getCoOccuranceItems(), numItems, numItems);


       // printM_and_C();
        // find knn based on 2D rating matrix, train
        userCache = train.rowCache(cacheSpec);

        if (knn > 0) {
            // find the nearest neighbors for each item based on item similarity
            SymmMatrix itemCorrs = buildCorrs(false); // this is based on transformed 2D rating matrix, this.train
            itemNNs = HashMultimap.create();

            for (int j = 0; j < numItems; j++) {
                // set diagonal entries to 0
                W.set(j, j, 0);

                // find the k-nearest neighbors for each item
                Map<Integer, Double> nns = itemCorrs.row(j).toMap();

                // sort by values to retriev topN similar items
                if (knn > 0 && knn < nns.size()) {
                    List<Map.Entry<Integer, Double>> sorted = Lists.sortMap(nns, true);
                    List<Map.Entry<Integer, Double>> subset = sorted.subList(0, knn);
                    nns.clear();
                    for (Map.Entry<Integer, Double> kv : subset)
                        nns.put(kv.getKey(), kv.getValue());
                }

                // put into the nns multimap
                for (Map.Entry<Integer, Double> en : nns.entrySet())
                    itemNNs.put(j, en.getKey());
            }
        } else {
            // all items are used
            allItems = train.columns();

            for (int j = 0; j < numItems; j++)
                W.set(j, j, 0.0);
        }
    }

    @Override
    protected void buildModel() throws Exception {


        // number of iteration cycles
        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;

            for (MatrixEntry me : trainMatrix) {

                int ui = me.row(); // user-item
                int u= rateDao.getUserIdFromUI(ui);
                int j= rateDao.getItemIdFromUI(ui);
                int c = me.column(); // context
                double rujc = me.get();

                double pred = predict(u, j, c, true, j);
                System.out.println("pred="+pred+"at u="+u+", j="+j);
                double eujc = rujc - pred;
                loss += eujc * eujc;

                // find k-nearest neighbors
                Collection<Integer> nns = knn > 0 ? itemNNs.get(j) : allItems;

                // update factors


               /* Collection<Integer> conditions=rateDao.getContextConditionsList().get(c);
                double dev_c=0;
                
                for(Integer cond:conditions)
                {
                    dev_c+=ciDev.get(j,cond);
                }*/
                
                Collection<Integer> conditions = rateDao.getContextConditionsList().get(c);
                Integer[] conditions_array =new Integer[conditions.size()];
                
                int counter=0;
               Iterator condIterator=conditions.iterator();
                while(condIterator.hasNext()){
                    conditions_array[counter]=(Integer) condIterator.next();
                    counter++;
                }
                double dev_c = 0;

                // sum over CoOccuranceContextItems and update CoOccuranceContextItems 
                for (int cond1 = 0; cond1 < conditions.size(); cond1++) {
                    for (int cond2 = cond1 + 1; cond2 < conditions.size(); cond2++) {

                        dev_c += ciDev.get(conditions_array[cond1], conditions_array[cond2]);
                    }

                }
                SparseVector Ru = userCache.get(u);

                // start updating W
                double sum_w=0;
                for (int k=0; k<numItems; k++) {
                    if(k!=j){
                    double update=W.get(k, j);
                    sum_w += update;

                    loss += regLw2*update*update + regLw1*update;

                    double delta_w = eujc*(Ru.get(k) + dev_c) - regLw2*update - regLw1;
                    update += lRate*delta_w;
                    W.set(k,j,update);
                }
                }

                // start updating ciDev
                   for (int cond1 = 0; cond1 < conditions.size(); cond1++) {
                    for (int cond2 = cond1 + 1; cond2 < conditions.size(); cond2++) {

                        double update = ciDev.get(conditions_array[cond1], conditions_array[cond2]);

                       // dev_c += update;
                        loss += regLc2 * update * update + regLc1 * update;

                        double delta_c = eujc * sum_w - regLc2 * update - regLc1;

                        update += lRate * delta_c;
                        ciDev.set(cond1, cond2, update);
                    }

                }

            }



        }
         Logs.debug("finish build model in ItemEmbedding_v4 in fold:{}",fold);
    }

    protected double predict(int u, int j, int c, boolean exclude, int excluded_item) throws Exception {


       SparseVector Ru = userCache.get(u);

        Collection<Integer> conditions = rateDao.getContextConditionsList().get(c);
      Integer[] conditions_array =new Integer[conditions.size()];
                
                int counter=0;
                Iterator condIterator=conditions.iterator();
                while(condIterator.hasNext()){
                    conditions_array[counter]=(Integer) condIterator.next();
                    counter++;
                }
        double dev_c;
        dev_c = 0.0;

        // sum over CoOccuranceContextItems and update CoOccuranceContextItems 
        for (int cond1 = 0; cond1 < conditions.size(); cond1++) {
            for (int cond2 = cond1 + 1; cond2 < conditions.size(); cond2++) {

                double val=ciDev.get(conditions_array[cond1], conditions_array[cond2]);
                System.out.println("fold:"+fold+","+"cond1="+conditions_array[cond1]+", cond2="+conditions_array[cond2]+",val="+val);
                dev_c = dev_c + val;
            }

        }

        double pred = 0;
        for (int k=0; k<numItems;k++) {
            if (Ru.contains(k)) {
                if (exclude == true && k == excluded_item) {
                    continue;
                } else {
                    double ruk = Ru.get(k);
                    pred += (ruk + dev_c) * W.get(k, j);
                }
            }
        }
        //double pred=1;

        return pred ;

    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        return predict(u,j,c,true,j);
    }

    @Override
    protected boolean isConverged(int iter) {
        double delta_loss = last_loss - loss;
        last_loss = loss;

        if (verbose)
            Logs.debug("{}{} iter {}: loss = {}, delta_loss = {}", algoName, foldInfo, iter, loss, delta_loss);

        return iter > 1 ? delta_loss < 1e-5 : false;
    }
    
    public void printM_and_C(){
        
        int sw=1;
        Logs.debug("\n M-row:{} and M-col:{} , fold:{}",W.numRows(),W.numColumns(),fold);
        for(int i=0; i<W.numRows(); i++)
        {
            System.out.println("\n i="+i);
            for(int j=0; j<W.numColumns(); j++){
            /*    if(new Double(W.get(i, j)).isNaN()){
                    System.out.println("i="+i+",j="+j+": "+W.get(i, j)+",");
                    sw=0;
                }*/
                System.out.print(W.get(i, j)+",");
            }
        }
        
      /*  if(sw==1){
           System.out.println("**********Not NAN value in W");
       }
       sw=1;*/
        
        Logs.debug("\n C-row:{} and C-col:{} , fold:{}",ciDev.numRows(),ciDev.numColumns(),fold);
        for(int i=0; i<ciDev.numRows(); i++)
        {
           System.out.println("\n i="+i);
            for(int j=0; j<ciDev.numColumns(); j++){
                System.out.print(ciDev.get(i, j)+",");
                 /*if(new Double(ciDev.get(i, j)).isNaN()){
                    System.out.println("i="+i+",j="+j+": "+ciDev.get(i, j)+",");
                    sw=0;
                }*/
            }
        }
      /*  if(sw==1){
           System.out.println("**********Not NAN value in ciDev");
       }*/
       
    }
    
    
    public void print(){
        
        int sw=1;
         Logs.debug("\n coOccurance-Items-row:{} and -col:{} , fold:{}",rateDao.getCoOccuranceItems().length,rateDao.getCoOccuranceItems().length,fold);
       for (int i = 0; i < rateDao.getCoOccuranceItems().length; i++) {
           
             System.out.println("\n i="+i);
            for (int j = 0; j < rateDao.getCoOccuranceItems().length; j++) {
              
                
                /*if(new Double(rateDao.getCoOccuranceItems()[i][j]).isNaN()){
                    System.out.println("i="+i+",j="+j+": "+rateDao.getCoOccuranceItems()[i][j]+",");
                    sw=0;
                }*/
               // rateDao.getCoOccuranceItems()[j][i]=rateDao.getCoOccuranceItems()[i][j];
                 System.out.print(rateDao.getCoOccuranceItems()[i][j]+",");
                
            }
       }
       
      /* if(sw==1){
           System.out.println("**********Not NAN value in coOccurance-Items");
       }
       sw=1;*/
        
        Logs.debug("\n ContextCoOccuranceuItems-row:{} and col:{} , fold:{}",rateDao.getCoOccuranceContextItems().length,rateDao.getCoOccuranceContextItems().length,fold);
        
        for (int i = 0; i < rateDao.getCoOccuranceContextItems().length; i++) {
           
          System.out.println("\n i="+i);
            for (int j = 0; j < rateDao.getCoOccuranceContextItems().length; j++) {
               
                /*if(new Double(this.rateDao.getCoOccuranceContextItems()[i][j]).isNaN()){
                    System.out.println("i="+i+",j="+j+": "+rateDao.getCoOccuranceContextItems()[i][j]+",");
                    sw=0;
                }*/
               
                System.out.print( this.rateDao.getCoOccuranceContextItems()[i][j]+",");
        
                
            }
    }
        
         /*if(sw==1){
           System.out.println("**********Not NAN value in ContextCoOccuranceuItems");
       }
       sw=1;
        */
        
        

}


}

