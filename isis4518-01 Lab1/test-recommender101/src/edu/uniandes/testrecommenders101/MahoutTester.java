package edu.uniandes.testrecommenders101;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.SpearmanCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;


public class MahoutTester {

  public static void main(String[] args) {
    try {
      DataModel model = new FileDataModel(new File("data/ml-100k/umahout.data")); // importar datos del modelo user id | item id | rating | timestamp
      UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
      System.out.println(similarity.userSimilarity(196, 210));
      UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);
      UserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
      
      List<RecommendedItem> recommendations = recommender.recommend(2, 5);
      for (RecommendedItem recommendation : recommendations) {
        System.out.println(recommendation);
      }
      
      similarity = new SpearmanCorrelationSimilarity(model);
      System.out.println(similarity.userSimilarity(196, 210));
      neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);
      recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
      
      recommendations = recommender.recommend(2, 5);
      for (RecommendedItem recommendation : recommendations) {
        System.out.println(recommendation);
      }
      
      similarity = new TanimotoCoefficientSimilarity(model);
      System.out.println(similarity.userSimilarity(196, 210));
      neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);
      recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
      
      recommendations = recommender.recommend(2, 5);
      for (RecommendedItem recommendation : recommendations) {
        System.out.println(recommendation);
      }
      
      // Aplicando mecanismo de formación de vecindario NarestNserNeightborhood
      similarity = new PearsonCorrelationSimilarity(model);
      neighborhood = new NearestNUserNeighborhood(10, similarity, model);
      recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
      
      recommendations = recommender.recommend(2, 5);
      for (RecommendedItem recommendation : recommendations) {
        System.out.println(recommendation);
      }
      
      similarity = new SpearmanCorrelationSimilarity(model);
      System.out.println(similarity.userSimilarity(196, 210));
      neighborhood = new NearestNUserNeighborhood(10, similarity, model);
      recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
      
      recommendations = recommender.recommend(2, 5);
      for (RecommendedItem recommendation : recommendations) {
        System.out.println(recommendation);
      }
      
      similarity = new TanimotoCoefficientSimilarity(model);
      System.out.println(similarity.userSimilarity(196, 210));
      neighborhood = new NearestNUserNeighborhood(10, similarity, model);
      recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
      
      recommendations = recommender.recommend(2, 5);
      for (RecommendedItem recommendation : recommendations) {
        System.out.println(recommendation);
      }
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } catch (TasteException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }

  }

}
