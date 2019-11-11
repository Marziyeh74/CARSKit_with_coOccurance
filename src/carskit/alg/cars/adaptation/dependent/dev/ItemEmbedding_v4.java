/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package carskit.alg.cars.adaptation.dependent.dev;

import carskit.alg.cars.adaptation.dependent.CSLIM;
import carskit.data.processor.DataDAO;
import carskit.data.setting.Configuration;
import carskit.data.structure.SparseMatrix;
import static carskit.generic.Recommender.rateDao;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import happy.coding.io.Lists;
import happy.coding.io.Logs;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import librec.data.DenseMatrix;
import librec.data.MatrixEntry;
import librec.data.SparseVector;
import librec.data.SymmMatrix;

/**
 *
 * @author Marziyeh
 */
@Configuration("binThold, knn, regLw2, regLw1, regLc2, regLc1, similarity, iters, rc")
public class ItemEmbedding_v4 extends CSLIM {

    // coOccurance_matrix items , items
    private DenseMatrix M;
    // coOccurance_matrix context_cond , conrext_conditions
    private DenseMatrix C;

    // item's nearest neighbors for kNN > 0
    private Multimap<Integer, Integer> itemNNs;

    // item's nearest neighbors for kNN <=0, i.e., all other items
    private List<Integer> allItems;

    public ItemEmbedding_v4(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold,double[][] M , double[][] C) {
        super(trainMatrix, testMatrix, fold);

        isRankingPred = true;
        isCARSRecommender = false; // this option is used to allow the algorithm to call 2D rating matrix "train"
        this.algoName = "ItemEmbedding_version4";

        regLw1 = algoOptions.getFloat("-lw1");
        regLw2 = algoOptions.getFloat("-lw2");
        regLc1 = algoOptions.getFloat("-lc1");
        regLc2 = algoOptions.getFloat("-lc2");

        knn = algoOptions.getInt("-k");
        als = algoOptions.getInt("-als");
        
        print();
       // this.M = new DenseMatrix(M, numItems, numItems);

        //this.C = new DenseMatrix(C, numConditions, numConditions);

    }

    @Override
    protected void initModel() throws Exception {
        super.initModel();

       // ciDev = new DenseMatrix(numItems, numConditions);
        //ciDev.init();
        
        M = new DenseMatrix(rateDao.getCoOccuranceItems(), numItems, numItems);

        C = new DenseMatrix(rateDao.getCoOccuranceContextItems(), numConditions, numConditions);

        
        printM_and_C();
        // find knn based on 2D rating matrix, train
        userCache = train.rowCache(cacheSpec);
        /*if (knn > 0) {
            // find the nearest neighbors for each item based on item similarity
            SymmMatrix itemCorrs = buildCorrs(false); // this is based on transformed 2D rating matrix, this.train
            itemNNs = HashMultimap.create();

            for (int j = 0; j < numItems; j++) {
                // set diagonal entries to 0
                M.set(j, j, 0);

                // find the k-nearest neighbors for each item
                Map<Integer, Double> nns = itemCorrs.row(j).toMap();

                // sort by values to retriev topN similar items
                if (knn > 0 && knn < nns.size()) {
                    List<Map.Entry<Integer, Double>> sorted = Lists.sortMap(nns, true);
                    List<Map.Entry<Integer, Double>> subset = sorted.subList(0, knn);
                    nns.clear();
                    for (Map.Entry<Integer, Double> kv : subset) {
                        nns.put(kv.getKey(), kv.getValue());
                    }
                }

                // put into the nns multimap
                for (Map.Entry<Integer, Double> en : nns.entrySet()) {
                    itemNNs.put(j, en.getKey());
                }
            }
        } else {
            // all items are used
            allItems = train.columns();

            for (int j = 0; j < numItems; j++) {
                M.set(j, j, 0.0);
            }
        }*/
        
        Logs.debug("finish init model in ItemEmbedding_v4 .. numItems:{} ..M-numrows:{} , fold={}", numItems, M.numRows(),fold);
        

        

    }

    @Override
    protected void buildModel() throws Exception {

        // number of iteration cycles
        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;

            
            Logs.debug(" start build model in ItemEmbedding_v4 in  iteration:{}, fold:{} ",iter,fold);
            for (MatrixEntry me : trainMatrix) {

                int ui = me.row(); // user-item
                int u = rateDao.getUserIdFromUI(ui);
                int i = rateDao.getItemIdFromUI(ui);
                int c = me.column(); // context
                double ruic = me.get();

                double pred = predict(u, i, c, true, i);
                double euic = ruic - pred;
                loss += euic * euic;

                // find k-nearest neighbors
               // Collection<Integer> nns = knn > 0 ? itemNNs.get(i) : allItems;

                // update factors
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

                        dev_c += C.get(conditions_array[cond1], conditions_array[cond2]);
                    }

                }
                SparseVector Ru = userCache.get(u);

                // start updating W
                double sum_w = 0;
                for (int k=0; k<numItems; k++) {
                    if(k!=i){
                    double update = M.get(k, i);
                    sum_w += update;

                    loss += regLw2 * update * update + regLw1 * update;

                    double delta_w = euic * (Ru.get(k) + dev_c) - regLw2 * update - regLw1;
                    update += lRate * delta_w;
                    M.set(k, i, update);
                    }
                }

                for (int cond1 = 0; cond1 < conditions.size(); cond1++) {
                    for (int cond2 = cond1 + 1; cond2 < conditions.size(); cond2++) {

                        double update = C.get(conditions_array[cond1], conditions_array[cond2]);

                       // dev_c += update;
                        loss += regLc2 * update * update + regLc1 * update;

                        double delta_c = euic * sum_w - regLc2 * update - regLc1;

                        update += lRate * delta_c;
                        C.set(cond1, cond2, update);
                    }

                }

            }
            
             //System.out.println(" build model in ItemEmbedding_v4 in iteration "+ iter);

        }
         Logs.debug("finish build model in ItemEmbedding_v4 in fold:{}",fold);
    }

    protected double predict(int u, int j, int c, boolean exclude, int excluded_item) throws Exception {

       // Collection<Integer> nns = knn > 0 ? itemNNs.get(j) : allItems;
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

                double val=C.get(conditions_array[cond1], conditions_array[cond2]);
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
                    pred += (ruk + dev_c) * M.get(k, j);
                }
            }
        }

        return pred;

    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        return predict(u, j, c, true, j);
    }

    @Override
    protected boolean isConverged(int iter) {
        double delta_loss = last_loss - loss;
        last_loss = loss;

        if (verbose) {
            Logs.debug("{}{} iter {}: loss = {}, delta_loss = {}", algoName, foldInfo, iter, loss, delta_loss);
        }

        return iter > 1 ? delta_loss < 1e-5 : false;
    }
    
    public void printM_and_C(){
        
        Logs.debug("\n M-row:{} and M-col:{} , fold:{}",M.numRows(),M.numColumns(),fold);
        for(int i=0; i<M.numRows(); i++)
        {
            System.out.println("\n i="+i);
            for(int j=0; j<M.numColumns(); j++){
                System.out.print(M.get(i, j)+",");
            }
        }
        
        Logs.debug("\n C-row:{} and C-col:{} , fold:{}",C.numRows(),C.numColumns(),fold);
        for(int i=0; i<C.numRows(); i++)
        {
            System.out.println("\n i="+i);
            for(int j=0; j<C.numColumns(); j++){
                System.out.print(C.get(i, j)+",");
            }
        }
    }
    
    
    public void print(){
        
         Logs.debug("\n coOccurance-Items-row:{} and -col:{} , fold:{}",rateDao.getCoOccuranceItems().length,rateDao.getCoOccuranceItems().length,fold);
       for (int i = 0; i < rateDao.getCoOccuranceItems().length; i++) {
           
             System.out.println("\n i="+i);
            for (int j = 0; j < rateDao.getCoOccuranceItems().length; j++) {
              
               // rateDao.getCoOccuranceItems()[j][i]=rateDao.getCoOccuranceItems()[i][j];
                 System.out.print(rateDao.getCoOccuranceItems()[i][j]+",");
                
            }
       }
        
        Logs.debug("\n ContextCoOccuranceuItems-row:{} and col:{} , fold:{}",rateDao.getCoOccuranceContextItems().length,rateDao.getCoOccuranceContextItems().length,fold);
        
        for (int i = 0; i < rateDao.getCoOccuranceContextItems().length; i++) {
           
           System.out.println("\n i="+i);
            for (int j = 0; j < rateDao.getCoOccuranceContextItems().length; j++) {
               
               
                System.out.print( this.rateDao.getCoOccuranceContextItems()[i][j]+",");
        
                
            }
    }

}
}
