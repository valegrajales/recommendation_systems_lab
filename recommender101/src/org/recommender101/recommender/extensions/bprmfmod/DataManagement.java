package org.recommender101.recommender.extensions.bprmfmod;

import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;

import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.recommender101.tools.matrix.SparseByteMatrix;
import org.recommender101.tools.matrix.SparseIntMatrix;
import org.recommender101.tools.matrix.SparseMatrix;

import org.recommender101.data.DataModel;
import org.recommender101.recommender.extensions.funksvd.impl.RandomUtils;

/**
 * Manages the data-objects of BPR-MF
 * 
 * Modified by LL
 * 
 */
public class DataManagement {
	// HashMaps to convert from real ids to mapped ones
	protected HashMap<Integer, Integer> userMap;
	protected HashMap<Integer, Integer> itemMap;

	// HashMaps to convert from mapped ids to real ones
	protected HashMap<Integer, Integer> userIndices;
	protected HashMap<Integer, Integer> itemIndices;

	// Itembias Array
	public double[] item_bias;

	// start values for mapping
	public int userid;
	public int itemid;

	// values for initialization of the latent matrices
	public static double sqrt_e_div_2_pi = Math.sqrt(Math.E / (2 * Math.PI));
	public static final Random random = RandomUtils.getRandom();

	// for initilizations of latent vectors
	private double initMean = 0;
	private double initStDev = 0.1;

	// latent vectors
	public double[][] latentUserVector;
	public double[][] latentItemVector;

	// HashMap containing for each user a list with seen item
	public HashMap<Integer, ArrayList<Integer>> userMatrix;

	// matrix containing for each user/item - combination a bool value,
	// indicating whether the user has seen the item
	// was in original: public boolean[][] boolMatrix;
	SparseByteMatrix boolMatrix;
	int boolMatrix_numUsers;
	int boolMatrix_numItems;
	
	// number of positive entries in then boolmatrix
	public int numPosentries;
	
	// additional matrices for multi-criteria BPR
	public SparseIntMatrix numberMatrixP;
	public SparseIntMatrix timeMatrixP;
	public SparseIntMatrix numberMatrixV;
	public SparseIntMatrix timeMatrixV;
	public SparseIntMatrix numberMatrixC;
	public SparseIntMatrix timeMatrixC;
	public SparseIntMatrix numberMatrixW;
	public SparseIntMatrix timeMatrixW;
	public SparseIntMatrix numberMatrixVR;
	public SparseIntMatrix timeMatrixVR;
	public SparseIntMatrix numberMatrixDC;
	public SparseIntMatrix timeMatrixDC;
	public SparseIntMatrix numberMatrixDW;
	public SparseIntMatrix timeMatrixDW;
	
	public SparseMatrix[] featureMatrixArray;
	
	// a parameter that enables the multi-criterie BPR
	// it is set from within the recommenders main class
	public boolean useUIMaps = false;
	
	public String selectIIMaps = "";
	public boolean useIIMaps = false;
	
	// use zalando or ML dataset
	boolean zalandoMode = false;
	
	// the datamodel
	public DataModel dm;

	// How to interpret rating data as binary data
	// Default: No (as done in original implementation)
	// If set to yes, the global item relevance threshold is applied
	// TODO UNIMPLEMENTED
	public boolean useRatingThreshold = false;
	

	/**
	 * initializes all the needed objects
	 * 
	 * @param dataModel
	 *            DataModel - the dataModel of the current training data
	 * @param numUser
	 *            Number - number of users
	 * @param numItem
	 *            Number - number of items
	 * @param numFeatures
	 *            Number - number of columns in the latent matrices
	 */
	public void init(DataModel dataModel, int numUsers, int numItems,
			int numFeatures) {
		dm = dataModel;
		
		userMap = new HashMap<Integer, Integer>();
		itemMap = new HashMap<Integer, Integer>();
		userIndices = new HashMap<Integer, Integer>();
		itemIndices = new HashMap<Integer, Integer>();
		userid = 0;
		itemid = 0;
		numPosentries = 0;
		userMatrix = new HashMap<Integer, ArrayList<Integer>>();

		for (Integer user : dataModel.getUsers()) {

			this.addUser(user);
		}

		for (Integer item : dataModel.getItems()) {

			this.addItem(item);
		}

		// init the latent vectors
		latentUserVector = new double[numUsers][numFeatures];
		latentItemVector = new double[numItems][numFeatures];

		initLatentmatrix(latentUserVector);
		initLatentmatrix(latentItemVector);
		
		// init the boolean matrix (used for implicit 0/1 ratings)
		item_bias = new double[numItems];
		boolMatrix = new SparseByteMatrix(numUsers,numItems);
		boolMatrix_numUsers = numUsers;
		boolMatrix_numItems = numItems;
		
		// init the more specific UI matrices that will carry number or time info
		if (useUIMaps) {
			numberMatrixP = new SparseIntMatrix(numUsers,numItems);
			timeMatrixP = new SparseIntMatrix(numUsers,numItems);
			if(zalandoMode){
				numberMatrixV = new SparseIntMatrix(numUsers,numItems);
				timeMatrixV = new SparseIntMatrix(numUsers,numItems);
				numberMatrixC = new SparseIntMatrix(numUsers,numItems);
				timeMatrixC = new SparseIntMatrix(numUsers,numItems);
				numberMatrixW = new SparseIntMatrix(numUsers,numItems);
				timeMatrixW = new SparseIntMatrix(numUsers,numItems);
				
				numberMatrixVR = new SparseIntMatrix(numUsers,numItems);
				timeMatrixVR = new SparseIntMatrix(numUsers,numItems);
				numberMatrixDC = new SparseIntMatrix(numUsers,numItems);
				timeMatrixDC = new SparseIntMatrix(numUsers,numItems);
				numberMatrixDW = new SparseIntMatrix(numUsers,numItems);
				timeMatrixDW = new SparseIntMatrix(numUsers,numItems);
			}
		}
		
		// fill those matrices
		this.booleanRatings();
		
		// then init and fill the II matrices
		if(useIIMaps){
			if(zalandoMode){
				this.createFeatureMatrix();
			}
			else{
				this.createFeatureMatrixML();
			}
		}
		System.out.println("Dataloader finished, start learning BPR++");
		
	}

	/**
	 * initiates the user/item-matrix with booleans instead of ratings
	 */
	public void booleanRatings() {
		
		// these maps are the ones that will be filled with the files from the splitter
		// they have a different indexing, therefore this second set is needed
		SparseIntMatrix loadedNumberMatrixP = null;
		SparseIntMatrix loadedTimeMatrixP = null;
		SparseIntMatrix loadedNumberMatrixV = null;
		SparseIntMatrix loadedTimeMatrixV = null;
		SparseIntMatrix loadedNumberMatrixC = null;
		SparseIntMatrix loadedTimeMatrixC = null;
		SparseIntMatrix loadedNumberMatrixW = null;
		SparseIntMatrix loadedTimeMatrixW = null;
		
		SparseIntMatrix loadedNumberMatrixVR = null;
		SparseIntMatrix loadedTimeMatrixVR = null;
		SparseIntMatrix loadedNumberMatrixDC = null;
		SparseIntMatrix loadedTimeMatrixDC = null;
		SparseIntMatrix loadedNumberMatrixDW = null;
		SparseIntMatrix loadedTimeMatrixDW = null;
		
		loadedNumberMatrixP = new SparseIntMatrix();
		loadedTimeMatrixP = new SparseIntMatrix();
		if(zalandoMode){
			loadedNumberMatrixV = new SparseIntMatrix();
			loadedTimeMatrixV = new SparseIntMatrix();
			loadedNumberMatrixC = new SparseIntMatrix();
			loadedTimeMatrixC = new SparseIntMatrix();
			loadedNumberMatrixW = new SparseIntMatrix();
			loadedTimeMatrixW = new SparseIntMatrix();
			
			loadedNumberMatrixVR = new SparseIntMatrix();
			loadedTimeMatrixVR = new SparseIntMatrix();
			loadedNumberMatrixDC = new SparseIntMatrix();
			loadedTimeMatrixDC = new SparseIntMatrix();
			loadedNumberMatrixDW = new SparseIntMatrix();
			loadedTimeMatrixDW = new SparseIntMatrix();
		}
		
		// load UI matrices from disk (up to 4x2 = 8)
		if (useUIMaps) {
			String persistanceFilename = "test_dm_exta_info";
			String persistanceFilenameEx = ".bin";
			System.out.print("Loading extra information...");
			ObjectInputStream ois;
			try {
				ois = new ObjectInputStream(new FileInputStream(
						persistanceFilename + persistanceFilenameEx));
				loadedNumberMatrixP = (SparseIntMatrix) ois.readObject();
				ois.close();
				ois = new ObjectInputStream(new FileInputStream(
						persistanceFilename + "_pt" + persistanceFilenameEx));
				loadedTimeMatrixP = (SparseIntMatrix) ois.readObject();
				ois.close();
				
				if(zalandoMode){
				ois = new ObjectInputStream(new FileInputStream(
						persistanceFilename + "_vn" + persistanceFilenameEx));
				loadedNumberMatrixV = (SparseIntMatrix) ois.readObject();
				ois.close();
				ois = new ObjectInputStream(new FileInputStream(
						persistanceFilename + "_vt" + persistanceFilenameEx));
				loadedTimeMatrixV = (SparseIntMatrix) ois.readObject();
				ois.close();
				
				ois = new ObjectInputStream(new FileInputStream(
						persistanceFilename + "_cn" + persistanceFilenameEx));
				loadedNumberMatrixC = (SparseIntMatrix) ois.readObject();
				ois.close();
				ois = new ObjectInputStream(new FileInputStream(
						persistanceFilename + "_ct" + persistanceFilenameEx));
				loadedTimeMatrixC = (SparseIntMatrix) ois.readObject();
				ois.close();
				
				ois = new ObjectInputStream(new FileInputStream(
						persistanceFilename + "_wn" + persistanceFilenameEx));
				loadedNumberMatrixW = (SparseIntMatrix) ois.readObject();
				ois.close();
				ois = new ObjectInputStream(new FileInputStream(
						persistanceFilename + "_wt" + persistanceFilenameEx));
				loadedTimeMatrixW = (SparseIntMatrix) ois.readObject();
				ois.close();
				
				ois = new ObjectInputStream(new FileInputStream(
						persistanceFilename + "_vrn" + persistanceFilenameEx));
				loadedNumberMatrixVR = (SparseIntMatrix) ois.readObject();
				ois.close();
				ois = new ObjectInputStream(new FileInputStream(
						persistanceFilename + "_vrt" + persistanceFilenameEx));
				loadedTimeMatrixVR = (SparseIntMatrix) ois.readObject();
				ois.close();
				
				ois = new ObjectInputStream(new FileInputStream(
						persistanceFilename + "_dcn" + persistanceFilenameEx));
				loadedNumberMatrixDC = (SparseIntMatrix) ois.readObject();
				ois.close();
				ois = new ObjectInputStream(new FileInputStream(
						persistanceFilename + "_dct" + persistanceFilenameEx));
				loadedTimeMatrixDC = (SparseIntMatrix) ois.readObject();
				ois.close();
				
				ois = new ObjectInputStream(new FileInputStream(
						persistanceFilename + "_dwn" + persistanceFilenameEx));
				loadedNumberMatrixDW = (SparseIntMatrix) ois.readObject();
				ois.close();
				ois = new ObjectInputStream(new FileInputStream(
						persistanceFilename + "_dwt" + persistanceFilenameEx));
				loadedTimeMatrixDW = (SparseIntMatrix) ois.readObject();
				ois.close();
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
			System.out.println("DONE!");
			/*System.out.println("P " + numberTimeMap.size());
			System.out.println("V " + VnumberTimeMap.size());
			System.out.println("C " + CnumberTimeMap.size());
			System.out.println("W " + WnumberTimeMap.size());*/
		}
		
		// go over all users ...
		for (int k = 0; k < boolMatrix_numUsers; k++) {
			ArrayList<Integer> userItems = new ArrayList<Integer>();
			// ... and items
			for (int l = 0; l < boolMatrix_numItems; l++) {
				// and get the internal IDs (e.g. the internal ML or Zalando IDs)
				int user = userMap.get(k);
				int item = itemMap.get(l);
				
				// if the item is interacted (e.g. purchased, rated)
				if (dm.getRating(user, item) > 0) {
					
					// TODO UNIMPLEMENTED: Should we only consider relevant items here?
					// e.g. for the ML data set, items which hace r > threshold
					if (this.useRatingThreshold) {
					}
					
					// check this binary in the bool matrix
					boolMatrix.setBool(k, l, true);
					
					// also if UI maps should be created, do this for the P (= purchase) maps
					if (useUIMaps){
						numberMatrixP.set(k,l,loadedNumberMatrixP.get(user,item));
						timeMatrixP.set(k,l,loadedTimeMatrixP.get(user,item));
						if (numberMatrixP.get(k, l) == 0 || timeMatrixP.get(k,l) == 0)
							System.err.println("Inconsistent Matrices! 1");
					}
					userItems.add(l);
					numPosentries++;
				} else { // if not interacted, do the opposite
					boolMatrix.setBool(k, l, false);
					
					if (useUIMaps){
						numberMatrixP.set(k,l,0);
						timeMatrixP.set(k,l,0);
						if (numberMatrixP.get(k, l) != 0 || timeMatrixP.get(k,l) != 0)
							System.err.println("Inconsistent Matrices! 2");
					}
				}
				
				
				// for Zalando, the other 3 UI matrices can be set
				if (useUIMaps){
					if(zalandoMode){
					numberMatrixV.set(k,l,loadedNumberMatrixV.get(user,item));
					timeMatrixV.set(k,l,loadedTimeMatrixV.get(user,item));
					numberMatrixC.set(k,l,loadedNumberMatrixC.get(user,item));
					timeMatrixC.set(k,l,loadedTimeMatrixC.get(user,item));
					numberMatrixW.set(k,l,loadedNumberMatrixW.get(user,item));
					timeMatrixW.set(k,l,loadedTimeMatrixW.get(user,item));
					
					numberMatrixVR.set(k,l,loadedNumberMatrixVR.get(user,item));
					timeMatrixVR.set(k,l,loadedTimeMatrixVR.get(user,item));
					numberMatrixDC.set(k,l,loadedNumberMatrixDC.get(user,item));
					timeMatrixDC.set(k,l,loadedTimeMatrixDC.get(user,item));
					numberMatrixDW.set(k,l,loadedNumberMatrixDW.get(user,item));
					timeMatrixDW.set(k,l,loadedTimeMatrixDW.get(user,item));
					}
				}
			}
			// finally add all interacted items (but not V,C,W) in a map
			userMatrix.put(k, userItems);
		}
	}

	/**
	 * This method fills the II (feature) matrices from disk for the current training set.
	 * Zalando-Edition
	 */
	private void createFeatureMatrix() {
		// an array containing all the feature matrices
		// each feature is associated with the index of the matrix
		//TIntIntMap[] featureMapArray = null;
		Map<String,TIntIntMap> featureMapArray = null;
		
		// load from disk
		String persistanceFilename = "test_dm_feature_info";
		String persistanceFilenameEx = ".bin";
		System.out.print("Loading extra information...");
		ObjectInputStream ois;
			try {
				ois = new ObjectInputStream(
						new FileInputStream(persistanceFilename + persistanceFilenameEx));
				//featureMapArray = (TIntIntMap[]) ois.readObject();
				featureMapArray = (Map<String,TIntIntMap>) ois.readObject();
				ois.close();
			} catch (Exception e) {
				e.printStackTrace();
			}
		System.out.println("DONE loading feature maps!");
		
		//int numberOfFeatures = 0;
		Set<String> matchingKeys = new HashSet<String>();
		
		Set<String> keys = featureMapArray.keySet();
		for (String key : keys){
			if(selectIIMaps.contains(key)) 
			{
				//numberOfFeatures++;
				matchingKeys.add(key);
			}
		}
		
		
		
		
		//int numberOfFeatures = featureMapArray.length;
		featureMatrixArray = new SparseByteMatrix[matchingKeys.size()];
		
		// until now, we have the features of items
		// each feature has exactly one value
		// what we want are matrices that indicate when 2 items share the same value of a feature

		// go through all features
		//for (int i = 0; i < numberOfFeatures; i++) {
		int i = 0;
		for (String key : matchingKeys) {
			featureMatrixArray[i] = new SparseByteMatrix(boolMatrix_numItems,boolMatrix_numItems);
			// go through all items
			for (int l = 0; l < boolMatrix_numItems; l++) {
				int item_l = itemMap.get(l);
				//get its features value
				int feature_l = featureMapArray.get(key).get(item_l);
				
				// go through all items again
				for (int k = 0; k < boolMatrix_numItems; k++) {
					int item_k = itemMap.get(k);
					// and check if the same feature for that other item has
					int feature_k = featureMapArray.get(key).get(item_k);
					if (feature_l == feature_k) {
						featureMatrixArray[i].set(l, k, (byte) 1);
					}
				}
			}
			i++;
		}
		System.out.println("DONE creating feature matrices!");
	}
	
	/**
	 * This method fills the II (feature) matrices from disk for the current training set.
	 * MovieLens-Edition
	 */
	private void createFeatureMatrixML() {
				// an array containing all the feature matrices
				// each feature is assiciated with the index of the matrix
				//TIntObjectMap[] featureMapArray = null;
				Map<String,TIntObjectMap> featureMapArray = null;
				
				// load from disk
				String persistanceFilename = "test_dm_feature_info";
				String persistanceFilenameEx = ".bin";
				System.out.print("Loading extra information...");
				ObjectInputStream ois;
					try {
						ois = new ObjectInputStream(
								new FileInputStream(persistanceFilename + persistanceFilenameEx));
						featureMapArray = (Map<String,TIntObjectMap>) ois.readObject();
						ois.close();
					} catch (Exception e) {
						e.printStackTrace();
					}
				System.out.println("DONE loading feature maps!");
				
				//int numberOfFeatures = 0;
				Set<String> matchingKeys = new HashSet<String>();
				
				Set<String> keys = featureMapArray.keySet();
				for (String key : keys){
					if(selectIIMaps.contains(key)) 
					{
						//numberOfFeatures++;
						matchingKeys.add(key);
					}
				}
				
				//int numberOfFeatures = featureMapArray.length;
				featureMatrixArray = new SparseIntMatrix[matchingKeys.size()];
				
				// until now, we have the features of items
				// each feature has a list of values
				// what we want are matrices that indicate percentage of value overlapping for 2 items
				
				// go through all features
				//for (int i = 0; i < numberOfFeatures; i++) {
				int i = 0;
				for (String key : matchingKeys) {
					featureMatrixArray[i] = new SparseIntMatrix(boolMatrix_numItems,boolMatrix_numItems);
					// go through all items
					for (int l = 0; l < boolMatrix_numItems; l++) {
						int item_l = itemMap.get(l);
						//get its set of values for the feature
						Set<Integer> featureset_l = (Set<Integer>)featureMapArray.get(key).get(item_l);
						// go again over all items
						for (int k = 0; k < boolMatrix_numItems; k++) {
							int item_k = itemMap.get(k);
							Set<Integer> featureset_k = (Set<Integer>)featureMapArray.get(key).get(item_k);
							// and calculate the feature intersection
							Set<Integer> intersection = intersect(featureset_k,featureset_l);
							featureMatrixArray[i].set(l, k, intersection.size());
						}
					}
					i++;
				}
				System.out.println("DONE creating feature matrices!");
	}
	
	/**
	 * initiates the given latent matrix with random values
	 * 
	 * @param matix
	 *            double[][] - the given latent matrix
	 */
	private void initLatentmatrix(double[][] matrix) {
		for (int k = 0; k < matrix.length; k++) {
			for (int l = 0; l < matrix[k].length; l++) {
				matrix[k][l] = this.nextNormal(initMean, initStDev);;
			}
		}
	}

	/**
	 * calculates the scalarproduct with rowdifference for the given parameters
	 * 
	 * @param user
	 *            Number - the mapped userID
	 * @param items1
	 *            Number - the mapped itemID of a viewed item
	 * @param item2
	 *            Number - the mapped itemID of an unviewed item
	 * @return result Number - the scalarproduct with rowdifference
	 */
	public double rowScalarProductWithRowDifference(int user, int item1,
			int item2) {

		if (user >= latentUserVector.length)
			throw new IllegalArgumentException("user too big: " + user
					+ ", dim1 is " + latentUserVector.length);
		if (item1 >= latentItemVector.length)
			throw new IllegalArgumentException("item1 too big: " + item1
					+ ", dim1 is " + latentItemVector.length);
		if (item2 >= latentItemVector.length)
			throw new IllegalArgumentException("item2 too big: " + item2
					+ ", dim1 is " + latentItemVector.length);
		if (latentUserVector[user].length != latentItemVector[item1].length)
			throw new IllegalArgumentException("wrong row size: "
					+ latentUserVector[user].length + " vs. "
					+ latentItemVector[item1].length);
		if (latentUserVector[user].length != latentItemVector[item2].length)
			throw new IllegalArgumentException("wrong row size: "
					+ latentUserVector[user].length + " vs. "
					+ latentItemVector[item2].length);

		double result = 0.0;
		for (int c = 0; c < latentUserVector[user].length; c++)
			result += (Double) latentUserVector[user][c]
					* ((Double) latentItemVector[item1][c] - (Double) latentItemVector[item2][c]);
		return result;
	}
	
	public double rowScalarProductWithRowDifferenceItem(int leftItem, int item1,
			int item2) {

		if (leftItem >= latentItemVector.length)
			throw new IllegalArgumentException("i too big: " + leftItem
					+ ", dim1 is " + latentItemVector.length);
		if (item1 >= latentItemVector.length)
			throw new IllegalArgumentException("item1 too big: " + item1
					+ ", dim1 is " + latentItemVector.length);
		if (item2 >= latentItemVector.length)
			throw new IllegalArgumentException("item2 too big: " + item2
					+ ", dim1 is " + latentItemVector.length);
		if (latentItemVector[leftItem].length != latentItemVector[item1].length)
			throw new IllegalArgumentException("wrong row size: "
					+ latentItemVector[leftItem].length + " vs. "
					+ latentItemVector[item1].length);
		if (latentItemVector[leftItem].length != latentItemVector[item2].length)
			throw new IllegalArgumentException("wrong row size: "
					+ latentItemVector[leftItem].length + " vs. "
					+ latentItemVector[item2].length);

		double result = 0.0;
		for (int c = 0; c < latentItemVector[leftItem].length; c++)
			result += (Double) latentItemVector[leftItem][c]
					* ((Double) latentItemVector[item1][c] - (Double) latentItemVector[item2][c]);
		return result;
	}

	/**
	 * calculates the scalarproduct for the given parameters
	 * 
	 * @param user
	 *            Number - the mapped userID
	 * @param items
	 *            Number - the mapped itemID of a viewed item
	 * @return result Number - the scalarproduct
	 */
	public double rowScalarProduct(int user, int item) {
		if (user >= latentUserVector.length)
			throw new IllegalArgumentException("i too big: " + user
					+ ", dim1 is " + latentUserVector.length);
		if (item >= latentItemVector.length)
			throw new IllegalArgumentException("j too big: " + item
					+ ", dim1 is " + latentItemVector.length);
		if (latentUserVector[user].length != latentItemVector[item].length)
			throw new IllegalArgumentException("wrong row size: "
					+ latentUserVector[user].length + " vs. "
					+ latentItemVector[item].length);

		Double result = 0.0;
		for (int c = 0; c < latentUserVector[user].length; c++)
			result += (Double) latentUserVector[user][c]
					* ((Double) latentItemVector[item][c]);
		return result;
	}

	/**
	 * adds the given user to the userMap and the userIndices
	 * 
	 * @param user
	 *            Number - unmapped userID
	 */
	public void addUser(int user) {
		userMap.put(userid, user);
		userIndices.put(user, userid);
		userid++;
	}

	/**
	 * adds the given item to the itemMap and the itemIndices
	 * 
	 * @param item
	 *            Number - unmapped itemID
	 */
	public void addItem(int item) {
		itemMap.put(itemid, item);
		itemIndices.put(item, itemid);
		itemid++;
	}

	public double nextNormal(double mean, double stdev) {
		return mean + stdev * nextNormal();
	}

	public double nextNormal() {
		double y;
		double x;
		do {
			double u = random.nextDouble();
			x = nextExp(1);
			y = 2 * u * sqrt_e_div_2_pi * Math.exp(-x);
		} while (y < (2 / (2 * Math.PI)) * Math.exp(-0.5 * x * x));
		if (random.nextDouble() < 0.5) {
			return x;
		} else {
			return -x;
		}
	}

	public double nextExp(double lambda) {
		double u = random.nextDouble();
		return -(1 / lambda) * Math.log(1 - u);
	}
	
	/**
	 * intersection of thw sets
	 * @param x
	 * @param y
	 * @return
	 */
	public Set<Integer> intersect(Set<Integer> x, Set<Integer> y){
		Set<Integer> result = new HashSet<Integer>();
		if(x!= null && y!= null){
			for (int val_x : x){
				for(int val_y : y){
					if (val_x == val_y) result.add(val_x);
				}
			}
		}
		
		return result;
	}
	
}
