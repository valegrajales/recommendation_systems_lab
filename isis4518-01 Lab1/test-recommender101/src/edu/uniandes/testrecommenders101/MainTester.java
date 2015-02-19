package edu.uniandes.testrecommenders101;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

import org.recommender101.data.DataModel;
import org.recommender101.data.DefaultDataLoader;
import org.recommender101.recommender.baseline.NearestNeighbors;

import edu.uniandes.testrecommenders101.util.ItemInfoLoader;
import edu.uniandes.testrecommenders101.util.ItemInformation;



public class MainTester {
	
	public static void main(String[] args) {
		try{
		    DataModel model= new DataModel();
		    DefaultDataLoader loader= new DefaultDataLoader();
		    loader.setFilename("data/ml-100k/u.data");
			loader.loadData(model);
			System.out.println(model.getRating(1000, 10));
			model.addRating(1000, 10, 5); // El usuario con id 1000, califica la película con id 10 (Ricardo III) con 5
			System.out.println(model.getRating(1000, 10));
			
		}catch (FileNotFoundException e) {
		  // TODO Auto-generated catch block
		  e.printStackTrace();
		} catch (Exception e) {
		  // TODO Auto-generated catch block
		  e.printStackTrace();
		}
		
		System.out.println("*****************************************************************************************");
		 
		AvgRecommender recomend = new AvgRecommender();
	    System.out.println("Most popular items sorted by average rating are: ");
		System.out.println( recomend.recommendItems(123) );

		System.out.println("*****************************************************************************************");
		
		FirstRecommender recomend2 = new FirstRecommender();
	    System.out.println("Most popular items sorted by number of ratings are: ");
		System.out.println(recomend2.recommendItems(123));
		
		try{
			System.out.println("*****************************************************************************************");

			DataModel dm = new DataModel();
			   DefaultDataLoader loader = new DefaultDataLoader();
			   loader.setFilename("data/ml-100k/u.data");
			   loader.setMinNumberOfRatingsPerUser("250");
			   loader.loadData(dm);
			   NearestNeighbors rec = new NearestNeighbors();
			   rec.setDataModel(dm);
			   rec.setCosineSimilarity("true");
			   
			   rec.init();
			   
			   List<Integer> list = rec.recommendItems(1);
			   System.out.println("Recomendation USER/USER");
			   System.out.println(list);
			
		}catch (FileNotFoundException e) {
		  // TODO Auto-generated catch block
		  e.printStackTrace();
		} catch (Exception e) {
		  // TODO Auto-generated catch block
		  e.printStackTrace();
		}

		try{
			System.out.println("*****************************************************************************************");
			DataModel dm = new DataModel();
			   DefaultDataLoader loader = new DefaultDataLoader();
			   loader.setFilename("data/ml-100k/u.data");
			   loader.setMinNumberOfRatingsPerUser("250");
			   loader.loadData(dm);
			   NearestNeighbors rec = new NearestNeighbors();
			   rec.setDataModel(dm);
			   rec.setCosineSimilarity("true");
			   rec.setItemBased("true");
			   
			   rec.init();
			   
			   List<Integer> list = rec.recommendItems(1);
			   System.out.println("Recomendation ITEM/ITEM");
			   System.out.println(list);
			
		}catch (FileNotFoundException e) {
		  // TODO Auto-generated catch block
		  e.printStackTrace();
		} catch (Exception e) {
		  // TODO Auto-generated catch block
		  e.printStackTrace();
		}
		
	}

}
