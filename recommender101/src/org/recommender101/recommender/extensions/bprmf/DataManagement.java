package org.recommender101.recommender.extensions.bprmf;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import org.recommender101.data.DataModel;
import org.recommender101.data.Rating;
import org.recommender101.recommender.extensions.funksvd.impl.RandomUtils;
import org.recommender101.tools.matrix.SparseByteMatrix;

/**
 * Manages the data-objects of BPR-MF
 * 
 */
public class DataManagement  implements Serializable {
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

	private double initMean = 0;
	private double initStDev = 0.1;

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

	public DataModel dm;

	// How to interpret rating data as binary data
	// Default: No (as done in original implementation)
	// If set to yes, the global item relevance threshold is applied
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

		
		
		latentUserVector = new double[numUsers][numFeatures];
		latentItemVector = new double[numItems][numFeatures];

//		System.out.println("0");
		
		initLatentmatrix(latentUserVector);
		initLatentmatrix(latentItemVector);

//		System.out.println("1");

		item_bias = new double[numItems];
		// was before:
		// boolMatrix = new boolean[numUsers][numItems];
		boolMatrix = new SparseByteMatrix(numUsers,numItems);
		boolMatrix_numUsers = numUsers;
		boolMatrix_numItems = numItems;

//		System.out.println("2");

		this.booleanRatings();
		
//		System.out.println("3");

		
	}

	/**
	 * initializes the user/item-matrix with booleans instead of ratings
	 */
	public void booleanRatings() {
		
		// DJ: new version .. no need to iterate over everything
		for (Rating r : dm.getRatings()) {
			Integer user = userIndices.get(r.user);
			Integer item = itemIndices.get(r.item);
			boolMatrix.setBool(user, item, true);
			
			ArrayList<Integer> ratingsOfUser = userMatrix.get(user);
			if (ratingsOfUser == null) {
				ratingsOfUser = new ArrayList<Integer>();
				userMatrix.put(user, ratingsOfUser);
			}
			ratingsOfUser.add(item);
			numPosentries++;
			
		}
		
/*		
		for (int k = 0; k < boolMatrix_numUsers; k++) { // was before: boolMatrix.length
			ArrayList<Integer> userItems = new ArrayList<Integer>();
			for (int l = 0; l < boolMatrix_numItems; l++) { // was before: boolMatrix[k].length
				int user = userMap.get(k);
				int item = itemMap.get(l);
				
				// the binary case
				if (Recommender101Impl.MAX_RATING == 1) {
					boolMatrix.setBool(k, l, true);
					userItems.add(l);
					numPosentries++;
					
				}
				else {
					
					if (dm.getRating(user, item) > 0) {
	//					// Should we only consider relevant items here?
	//					if (this.useRatingThreshold) {
	//					}
						
						// was before: boolMatrix[k][l] = true;
						boolMatrix.setBool(k, l, true);
						userItems.add(l);
						numPosentries++;
					} 
//					else {
//						// was before: boolMatrix[k][l] = false;
//						boolMatrix.setBool(k, l, false);
//					}
				}

			}
			userMatrix.put(k, userItems);
		}
		*/
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
			throw new IllegalArgumentException("i too big: " + user
					+ ", dim1 is " + latentUserVector.length);
		if (item1 >= latentItemVector.length)
			throw new IllegalArgumentException("item1 too big: " + item1
					+ ", dim1 is " + latentItemVector.length);
		if (item2 >= latentItemVector.length)
			throw new IllegalArgumentException("j too big: " + item2
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

	
	
}
