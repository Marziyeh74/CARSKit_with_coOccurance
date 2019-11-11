/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package carskit.data.structure;

import java.util.ArrayList;

/**
 *
 * @author Marziyeh
 */
public class Triple {
    
   public  int user;
    public int item;
   public  ArrayList<Integer> condlist;
    
    public Triple(int user,int item,ArrayList<Integer>  condList){
        this.user=user;
        this.item=item;
        this.condlist=condlist;
    }
}
