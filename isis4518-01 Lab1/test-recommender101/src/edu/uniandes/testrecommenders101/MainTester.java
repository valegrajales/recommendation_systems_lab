package edu.uniandes.testrecommenders101;

import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;

import org.recommender101.data.DataModel;
import org.recommender101.data.DefaultDataLoader;

import edu.uniandes.testrecommenders101.util.ItemInfoLoader;
import edu.uniandes.testrecommenders101.util.ItemInformation;



public class MainTester {
	
	public static void main(String[] args) {
		
		try {  
      DataModel model= new DataModel();
      DefaultDataLoader loader= new DefaultDataLoader();
      loader.setFilename("data/ml-100k/u.data");
      loader.loadData(model);
      
      model.addRating(1000, 10, 5); // El usuario con id 1000, califica la película con id 10 (Ricardo III) con 5
      
      ItemInfoLoader itemInfo = new ItemInfoLoader();
      String itemFileName = "data/ml-100k/u.item";
      ItemInformation[] array= itemInfo.getItemsSortedByPopularity(model,itemFileName);
      System.out.println("Most popular items sorted by number of ratings are: ");
      int count = 30;
      for(int i =0; i< count ;i++){
        System.out.println(array[i]);
      }
      System.out.println("Least popular items sorted by number of ratings are: ");
      for(int i =array.length-1; i> array.length-1-count;i--){
        System.out.println(array[i]);
      }
      
      FirstRecommender recomendador1 = new FirstRecommender();
      
      
    } catch (FileNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
	}

}
