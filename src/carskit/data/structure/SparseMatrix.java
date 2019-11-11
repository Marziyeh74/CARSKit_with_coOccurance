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

package carskit.data.structure;

import com.google.common.collect.*;
import happy.coding.io.Logs;
import happy.coding.math.Stats;
import librec.data.MatrixEntry;
import librec.data.SparseVector;

import java.util.*;

/**
 * Created by yzheng on 7/31/15.
 */
public class SparseMatrix extends librec.data.SparseMatrix {

    double rating_global=0;

    public SparseMatrix(int rows, int cols, Table<Integer, Integer, Double> dataTable, Multimap<Integer, Integer> colMap) {
        super(rows, cols, dataTable, colMap);
        System.out.println("size="+super.size());// num of record
        System.out.println("row="+super.numRows());
        System.out.println("column="+super.numColumns());
        
        double[] getdata = super.getData();
         System.out.println("getDate size="+getdata.length);// equal of super.size()
        
         System.out.println("****getDate:"+getdata.length);
        
        /*
        for (int i = 0; i < getdata.length; i++) {
            double data = getdata[i];
             System.out.println("data:"+data);
        
            
        }
         */
        printTable(dataTable);
        
       // System.exit(1);
    }

    public SparseMatrix(int rows, int cols, Table<Integer, Integer, Double> dataTable) {
        super(rows, cols, dataTable);
    }

    public SparseMatrix(librec.data.SparseMatrix mat) {
        super(mat);
    }


    public double getGlobalAvg()
    {
        if(rating_global==0)
        {
            rating_global=this.sum()/this.size();
        }
        return rating_global;
    }

    // get average rating by a list of UserItem as query
    public double getAverage(Collection<Integer> uiList)
    {
            double rate = 0.0;
            double counter = 0.0;
            for(int uiid:uiList){
                SparseVector sv=this.row(uiid);
                if(sv.size()>0){
                    rate+=sv.sum();
                    counter+=sv.size();
                }
            }
        if (counter > 0)
            rate /= counter;
        else
            rate = 0;

        return rate;

    }
    
    public void printTable(Table<Integer, Integer, Double> dataTable){
       
      /*  for (Iterator i = dataTable.rowKeySet().iterator(); i.hasNext();) {
            double[] cols = super.colData;
            System.out.println("i="+i+" : ");
            
            for (int j = 0; j < cols.length; j++) {
                double col = cols[j];
              //  System.out.println(col[j]+" ,");
            
            }
             System.out.println();
              */
     
    }
        
    
}
