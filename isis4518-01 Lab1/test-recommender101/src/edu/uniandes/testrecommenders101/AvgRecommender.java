package edu.uniandes.testrecommenders101;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

import org.recommender101.data.DataModel;
import org.recommender101.data.DefaultDataLoader;
import org.recommender101.recommender.AbstractRecommender;
import org.recommender101.recommender.extensions.bprmf.BPRMFRecommender;
import org.recommender101.recommender.extensions.contentbased.ContentBasedRecommender;
import org.recommender101.recommender.extensions.rfrec.RfRecRecommender;

import edu.uniandes.testrecommenders101.util.ItemInfoLoader;
import edu.uniandes.testrecommenders101.util.ItemInformation;


public class AvgRecommender extends AbstractRecommender {

  @Override
  public float predictRating(int user, int item) {
	  return 0;
  }

  @Override
  public List<Integer> recommendItems(int user) {
    try {  
	  DataModel model= new DataModel();
	  DefaultDataLoader loader= new DefaultDataLoader();
	  loader.setFilename("data/ml-100k/u.data");
	  loader.loadData(model);
	  
	  ItemInfoLoader itemInfo = new ItemInfoLoader();
	  String itemFileName = "data/ml-100k/u.item";
	  ItemInformation[] array= itemInfo.getItemsSortedByAverage(model,itemFileName);
	  
	  List<Integer> ids = new ArrayList<Integer>(); 
	  for(int i = 0; ids.size() <= 30; i++){
	    Long id = array[i].getItemId();
	    Integer iid = (int) (long) id;
		if( model.getRating(user, iid) == -1)
		{
			ids.add( iid );
//		    System.out.println(array[i]);
	    }
	  }
	  return ids;
	  
	} catch (FileNotFoundException e) {
	  // TODO Auto-generated catch block
	  e.printStackTrace();
	} catch (Exception e) {
	  // TODO Auto-generated catch block
	  e.printStackTrace();
	}
    return null;
  }

  @Override
  public void init() throws Exception {
    // TODO Auto-generated method stub

  }

}
