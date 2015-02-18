package org.recommender101.recommender.extensions.bprmfmod;

import gnu.trove.map.TIntByteMap;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntByteHashMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.recommender101.tools.matrix.SparseIntMatrix;
import org.recommender101.tools.matrix.SparseMatrix;
import org.recommender101.data.Rating;
import org.recommender101.recommender.AbstractRecommender;
import org.recommender101.recommender.extensions.funksvd.impl.RandomUtils;
import org.recommender101.tools.Debug;
import org.recommender101.tools.Utilities101;

/**
 * Bayesian Personalized Ranking - Ranking by pairwise classification
 * Literature: Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars
 * Schmidt-Thieme: BPR: Bayesian Personalized Ranking from Implicit Feedback.
 * UAI 2009. http://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle_et_al2009-
 * Bayesian_Personalized_Ranking.pdf
 * 
 * Code based on MyMediaLite http://www.ismll.uni-hildesheim.de/mymedialite/
 * 
 * Modified by LL
 * 
 * @author MW
 * 
 */
public class BPRMFMODRecommender extends AbstractRecommender {
	
	
	// Sample uniformly from users, for the strategy of the original paper set this to true
	public boolean UniformUserSampling = false;
	
	// in Uniform user sampling, use sequential learning?
	boolean SequentialLearning = false;
	
	// Defines the criterion for sorted triples x_uij
	// unseen, number or time possible for Zalando data
	// unseen and rating is possible for ML data, also need useUIMaps = true;
	// number and time need useMaps = true
	public String tripleCriterion = "unseen";
	
	// enabled detailed UI comparision based in number or time of access
	// use aux item-to-item realtions in bpr (Zalando and Movielens)
	// this features uses the maps from the ZalandoImplicitDataSplitter
	// therefore you have to enable the createMaps for that class in the settings
	public boolean useUIMaps = false;
	
	// dataset mode. enables loading and usage of V, C and W in zalando dataset
	// use aux user-to-item relations in bpr (ZalandoMode, V,C,W)
	// needs useUIMaps = true;
	public boolean zalandoMode = false;
	
	// enabled detailed II comparision based in number or time of access
	// this features uses the maps from the ZalandoImplicitDataSplitter
	// therefore you have to enable the createMaps for that class in the settings
	//public String selectIIMaps = "";	//TODO Select II MAPS, also in data
	public boolean useIIMaps =  false;
		
	// use the difference between x_ui and x_uj as a multiplier for 1/1+x_uij
	// This accelerates the gradient for a strong "likeness"
	// makes only sense for non-binary tripleCriteria
	public boolean useXscale = false;
	
	
	private static final Random random = RandomUtils.getRandom();

	// Regularization parameter for the bias term
	public double biasReg = 0;

	// number of columns in the latent matrices
	public int numFeatures = 100;

	// number of iterations over the data
	public int initialSteps = 100;

	// number of users
	protected int numUsers;

	// number of items
	public int numItems;

	// datamanagement-object
	public DataManagement data = new DataManagement();

	// Learning rate alpha
	public double learnRate = 0.05;

	// Regularization parameter for user factors
	public double regU = 0.0025;

	// Regularization parameter for positive item factors
	public double regI = 0.0025;

	// Regularization parameter for negative item factors</summary>
	public double regJ = 0.00025;

	// If set (default), update factors for negative sampled items during
	// learning
	public boolean updateJ = true;
	
	// debug parameter for testing
	public String debug = "";

	@Override
	/**
	 * Not implemented
	 * Returns rating of item by user
	 * @param user Number  - the user ID
	 * @param item Number  - the item ID
	 * @returns Float rating of the given item by the given user
	 */
	public float predictRating(int user, int item) {
		return Float.NaN;
	}

	/**
	 * Returns rating of item by user -> not usable for rating prediction
	 * @param user Number  - the user ID
	 * @param item Number  - the item ID
	 * @returns Float rating of the given item by the given user
	 */
	public float predictRatingBPR(int user, int item) {
		
		// Note: Predictions are only helpful for ranking and not for prediction
		// convert IDs in mapped values
		int itemidx = data.itemIndices.get(item);
		Integer useridx = data.userIndices.get(user);
		
		if (useridx != null) {
			return (float) (data.item_bias[itemidx] + data.rowScalarProduct( useridx, itemidx));
		}
		else {
			// This might happen during training test splits for super-sparse (test) data
//			System.out.println("-- No entry for user: " + user);
			return Float.NaN;
		}
	}
	// =====================================================================================

	/**
	 * This is similar to AbstractRecommender.recommendItemsByRatingPrediction but uses the internal function
	 */
	public List<Integer> recommendByPrediction(int user) {
		List<Integer> result = new ArrayList<Integer>();

		// If there are no ratings for the user in the training set,
		// there is no point of making a recommendation.
		Set<Rating> ratings = getDataModel().getRatingsOfUser(user);
		// If we have no ratings...
		if (ratings == null || ratings.size() == 0) {
			return Collections.emptyList();
		}
//		System.out.println("I have " + ratings.size() + " total ratings of user " + user);
		
		// Calculate rating predictions for all items we know
		Map<Integer, Float> predictions = new HashMap<Integer, Float>();
		float pred = Float.NaN;
		
		// Go through all the items
		for (Integer item : dataModel.getItems()) {
			boolean userHasAlreadyRatedItem = false;
			
			// Look if there is a shadow copy of original sales
			// We will not recommend items repeatedly here
			if (getDataModel().originalTrainingPerUser.keySet().size() > 0) {
				Set<Rating> originalRatings = getDataModel().originalTrainingPerUser.get(user);
				// Let's look there
//				System.out.println("Looking up the user in the shadow copy");
				if (originalRatings != null && Utilities101.ratingExists(user, item, originalRatings)) {
					userHasAlreadyRatedItem = true;
				}
			}
			else {
				// There were no implicit ratings
				// Do the standard procedure
				byte rating = dataModel.getRating(user, item);
				if (rating != -1) {
					userHasAlreadyRatedItem = true;
				}
			}
			
			if (!userHasAlreadyRatedItem) {
				// make a prediction and remember it in case the recommender
				// could make one
				pred = predictRatingBPR(user, item);
				if (!Float.isNaN(pred)) {
					predictions.put(item, pred);
				}
			}
		}
		
		predictions = filterElementsByRelevanceThreshold(predictions, user);
		predictions = Utilities101.sortByValueDescending(predictions);
		
		for (Integer item : predictions.keySet()) {
			result.add(item);
		}
		return result;
	}
	
	
	/**
	 * This method recommends items.
	 */
	@Override
	public List<Integer> recommendItems(int user) {
		return recommendByPrediction(user);
	}

	// =====================================================================================

	@Override
	/**
	 * Initialization of the needed objects and variables
	 * 
	 */
	public void init() {
		// ascertain number of users and items
		numItems = dataModel.getItems().size();
		numUsers = dataModel.getUsers().size();
//		System.out.println("Users, items: " + numUsers + " " + numItems + " ratings " + dataModel.getRatings().size());
		
		// Initialization of datamanagement-object
		data.init(dataModel, numUsers, numItems, numFeatures);


		// train the recommender
		train();

		System.out.println("BPR++ training DONE");
		
		Debug.log("BPRMF:init: Initial training done");

	}


	// =====================================================================================
	/**
	 * Training of the given data
	 * The variable "initialSteps" controls the number of training steps
	 * 
	 */
	public void train() {
		for (int i = 0; i < initialSteps; i++) {
//			System.out.println("Iterating BPR: " + i);
			if (i%10 == 0) System.out.println("Processed "+ i +" of "+initialSteps+ " BPR++ training steps.");
			iterate();
		}
	}

	// =====================================================================================
	// THE MAIN RUNNER METHOD:
	// =====================================================================================
	
	
	/**
	 * Perform one iteration of stochastic gradient ascent over the training data.
	 * 
	 * This method is the main runner. Here happens all the logic of this recommender.
	 * Basically, there are 2 different modes how the training can be done:
	 * 
	 * Mode 1: Uniform user sampling
	 * Mode 2: Complete sampling
	 * 
	 * Mode 1 is what is used in the BPR paper by Rendle. It picks randomly purchases and updates the
	 * MF parameters with them. The number of picks is the number of purchases but there is no check for
	 * doublets or missing ones.
	 * 
	 * The modification here is that there can be additional MF updates by auxiliary relations.
	 * The first method to integrate them is sequential learning: Before training with the target relataion
	 * (here: Purchases), there is training done with the auxiliary item 2 item relations, one at a time, then
	 * with the auxiliary user 2 item relations (here View, Cart, Wish). Similar to the target relation, there
	 * are random picks equal to the number of relation entries.
	 * The second method is simultaneous learning: For every update step of the target relation, there is also
	 * a random pick and update from all of the auxiliary relations. Therefore the auxiliary relations are only
	 * accessed as often as the target and they are accessed alternating with the target relation instead of
	 * before.
	 * 
	 * Mode 2 -- instead of picking random -- runs over all users and items and performs MF updates for
	 * every found purchase.
	 * 
	 * It is modified to run over the complete auxiliary relations, too. First, all the auxiliary item 2
	 * item relations are processed completely one after another. After that the target relation and the
	 * user 2 item auxiliary relations are are completely but alternatingly processed.
	 * 
	 * There are two more addtions:
	 * - First, the interpretation of x_uij and x_kij was extended to be not only boolean, but rather integer.
	 *   Therefore in the user 2 items auxiliary and the target relation not only a "has accessed" > "has not"
	 *   is possible but also "has accessed many time" > "has accessed few times". Also time comparision can be
	 *   made.
	 * - Second, there is a new scaling parameter, xscale, that smoothens the BPR. It is taken from the
	 *   the above mentions integer interpretation of x_uij, therefore the difference between x_ui and x_uj.
	 * 
	 */
	public void iterate() {
		// number of all positive ratings
		int num_pos_events = data.numPosentries;

		int user_id, pos_item_id, neg_item_id, initial_item_id, xscale;

		
		/*
		if (!UniformUserSampling){
		// this is a backlog of all the concrete triples. use all of them to learn, maybe emphasize on them
		// in the original paper, rendle did not run ober all the triples because of runtime
		// but using only the concrete triples (which are hard to reach when randomizing) might be a good idea
		List<int[]> backlog = new ArrayList<int[]>();
		if (tripleCriterion.equals("rating")){	// for now use this just for rating
		// A new idea

		Random rng = new Random();
			
		// for all users
		//for (int k = 0; k < data.boolMatrix_numUsers; k++) {
		for (int theUser : data.userMatrix.keySet()) {
		
			// and their rated items
			//for (int l = 0; l < data.boolMatrix_numItems; l++) {
			List<Integer> user_items = data.userMatrix.get(theUser);
			for (int thePosItem : user_items) {
					
					//List<Integer> user_items = data.userMatrix.get(user_id);
				
					// try 5 times for one +- element for each ++ element
					for(int i = 0; i < 5; i++) {
				
					if (user_items.size() > 1){
						int possibleLowerID = rng.nextInt(user_items.size());
						int possibleLowerItem = user_items.get(possibleLowerID);
						if (possibleLowerItem != thePosItem){
							int rating_pos = data.dm.getRating(data.userMap.get(theUser),data.itemMap.get(thePosItem));
							// rating of possible +-
							int rating_neg = data.dm.getRating(data.userMap.get(theUser),data.itemMap.get(possibleLowerItem));
							//check if it is really a +- item:
							if (rating_pos > rating_neg){
								//if yes, then backlog the triple
								int[] sampleTriple = {theUser,thePosItem,possibleLowerItem}; // TODO temp value for xscale
								backlog.add(sampleTriple);
								break;
							}
						}
						
					}
					else{break;}
					}
				
					/*
					// go again over the users interacted items
					for (int possibleLowerItem : user_items){
						// if it is not the same as item ++
						if (possibleLowerItem != thePosItem){
							//rating of ++ item
							int rating_pos = data.dm.getRating(data.userMap.get(theUser),data.itemMap.get(thePosItem));
							// rating of possible +-
							int rating_neg = data.dm.getRating(data.userMap.get(theUser),data.itemMap.get(possibleLowerItem));
							//check if it is really a +- item:
							if (rating_pos > rating_neg){
								//if yes, then backlog the triple
								int[] sampleTriple = {theUser,thePosItem,possibleLowerItem}; // TODO temp value for xscale
								backlog.add(sampleTriple);
							}
						}
					}
					 *//*
			}
		}
		}
		System.out.println(backlog.size());
		Collections.shuffle(backlog);
		//finally perform a gradient descent update for user-to-item relations // TODO where?
		int i = 0;
		for(int[] bTriple : backlog){
			updateFactors(bTriple[0], bTriple[1], bTriple[2], true, true, updateJ, 1,learnRate, false);
			//i++;
			//if (i % 100 == 0) System.out.println("done with " + i);
		}
		// new idea END
		}
		else*/
		
		// Perform Uniform User Sampling: Take a random number of x_uij from target relation to learn
		if (UniformUserSampling) {

			// Sequential Learning for auxiliary relations: Learn from the auxiliary relation BEFORE
			// learning from target relation
			if(SequentialLearning){
				if(useIIMaps){
					// Learn from auxiliary item-to-item relations
					for(int f = 0; f < data.featureMatrixArray.length; f++){
						SparseMatrix matrix = data.featureMatrixArray[f];
						// Learning happens by Uniform Sampling
						for (int entries = 0; entries < matrix.getNumberOfEntries(); entries ++){
							auxiliaryStep(matrix);
						}
					}
				}
				
				// Learn from auxiliary user-to-item relations
				if (zalandoMode){
					SparseIntMatrix[] mcArray = initMCArray();
					for (int f = 0; f < mcArray.length; f++){
						SparseIntMatrix matrix = mcArray[f];
						// Learning happens by Uniform Sampling
						for (int entries = 0; entries < matrix.getNumberOfEntries(); entries ++){
							auxiliaryStep(matrix);
						}
					}
				}
			}
			
			//performing convergence-heuristic of LearnBPR
			for (int i = 0; i < num_pos_events; i++) {

				// Parallel Learning for auxiliary relations: Learn from the auxiliary relation WHILE
				// learning from target relation
				if(!SequentialLearning){
					if(useIIMaps){
						// Learn from auxiliary item-to-item relations
						for(int f = 0; f < data.featureMatrixArray.length; f++){
							SparseMatrix matrix = data.featureMatrixArray[f];
							// There is just one learning step here, because they are tied to the steps of the
							// target relation
							auxiliaryStep(matrix);
						}
					}
						
					// Learn from auxiliary user-to-item relations
					if (zalandoMode){
						SparseIntMatrix[] mcArray = initMCArray();
						for (int f = 0; f < mcArray.length; f++){
							SparseIntMatrix matrix = mcArray[f];
							// There is just one learning step here, because they are tied to the steps of the
							// target relation
							auxiliaryStep(matrix);
						}
					}
				}

				// Learn from the target relation (user-to-item):
				
				// sampling a triple, consisting of a user, a viewed item and an
				// unseen one; also have a 4th component aka xscale smoothing factor
				int[] triple = new int[4];
				
				//initial xscale
				xscale = 1;
				
				// based on the type of input (by number, by time, by unseen),
				// first acquire an user, then a pair of BPR-items
				if (tripleCriterion.equals("number")){
					triple[0] = sampleU(data.numberMatrixP);
					triple = sampleIJ(triple, data.numberMatrixP, false);
				}
				else if (tripleCriterion.equals("time")){
					triple[0] = sampleU(data.timeMatrixP);
					triple = sampleIJ(triple, data.timeMatrixP, false);
				}
				else if (tripleCriterion.equals("rating")){
					triple[0] = sampleU();
					triple = sampleIJrating(triple);
				}
				else{
					// case "unseen"
					triple[0] = sampleU();
					triple = sampleIJ(triple);
				}
				
				user_id = triple[0];
				pos_item_id = triple[1];
				neg_item_id = triple[2];
				if(useXscale) xscale = triple[3];

				// finally perform a gradient descent update
				updateFactors(user_id, pos_item_id, neg_item_id, true, true, updateJ, xscale,learnRate, false);
			}

		// ------------------------------------------------------------------------------------------------------
			
		// !UniformUserSampling -> Complete Sampling
		} else {
			
			if(useIIMaps){
			// Sequential Learning for auxiliary relations: Learn from the auxiliary relation BEFORE
			// learning from target relation
			// Here only the item-to-item relations are learned initally
			for (int f = 0; f < data.featureMatrixArray.length; f++) {

				
				// Run over all possible item-to-item-combinations
				for (int k = 0; k < data.boolMatrix_numItems; k++) {
					for (int l = 0; l < data.boolMatrix_numItems; l++) {

						// set initial values
						initial_item_id = k;
						pos_item_id = l;
						neg_item_id = -1;
						xscale = 1;
						
						// Only use an "initial_item", if it's BPR compliant:
						// if the item pos_item_id in fact has the same value for a featre as initial_item_id;
						if (data.featureMatrixArray[f].getBool(initial_item_id,pos_item_id)) {
							int[] sampleTriple = null;

							// Sample a random negative item
							sampleTriple = sampleJ(initial_item_id,
									pos_item_id, neg_item_id,
									data.featureMatrixArray[f]);

							initial_item_id = sampleTriple[0];
							pos_item_id = sampleTriple[1];
							neg_item_id = sampleTriple[2];
							
							// update xscale if used
							if (useXscale)
								xscale = sampleTriple[3];

							// finally perform a gradient descent update for item-to-item relations
							updateFactors(initial_item_id, pos_item_id,
									neg_item_id, true, true, updateJ, xscale,
									learnRate, true);
						}
					}
				}
			}
			}
			
			// Now do roughly the same for the target-relation (user-to-item)
			// runs over all possible user-item-combinations
			for (int k = 0; k < data.boolMatrix_numUsers; k++) {
				for (int l = 0; l < data.boolMatrix_numItems; l++) {

					// set initial values
					user_id = k;
					pos_item_id = l;
					neg_item_id = -1;
					xscale = 1;
					
					// only use a pair of user and pos_item if it's BPR compliant
					// that is, if it's been interacted with by the user
					if (data.boolMatrix.getBool(user_id,pos_item_id)) {

						// Sample a random negative item in addition to the user and pos_item
						int[] sampleTriple = null;
						if (tripleCriterion.equals("number")) {
							sampleTriple = sampleJ(user_id, pos_item_id, neg_item_id,data.numberMatrixP, false);
						} else if (tripleCriterion.equals("time")) {
							sampleTriple = sampleJ(user_id, pos_item_id, neg_item_id,data.timeMatrixP, false);
						} else if (tripleCriterion.equals("rating")){
							float useravg = data.dm.getUserAverageRating(data.userMap.get(user_id));
							int rating = data.dm.getRating(data.userMap.get(user_id),data.itemMap.get(pos_item_id));
							// only consider items as positive if they have at least avg user rating
							if (rating < useravg) continue;
							else
								sampleTriple = sampleJrating(user_id,pos_item_id, neg_item_id);
						} else {
							// case "unseen"
							sampleTriple = sampleJ(user_id,pos_item_id, neg_item_id);
						}
						
						user_id = sampleTriple[0];
						pos_item_id = sampleTriple[1];
						neg_item_id = sampleTriple[2];
						
						// update xscale if used
						if (useXscale)
							xscale = sampleTriple[3];
						
						// finally perform a gradient descent update for user-to-item relations
						updateFactors(user_id, pos_item_id, neg_item_id, true, true, updateJ, xscale,learnRate, false);
					}
					
					// Parallel Learning for auxiliary relations: Learn from the auxiliary relation WHILE
					// learning from target relation; here only the aux user-to-item relations
					if (zalandoMode){
						
						// Get the user-to-item aux relations based on type
						SparseIntMatrix[] mcArray = initMCArray();
						// for all of them
						for (int f = 0; f < mcArray.length; f++){
							SparseIntMatrix matrix = mcArray[f];
							
							// set initial values
							user_id = k;
							pos_item_id = l;
							neg_item_id = -1;
							xscale = 1;
							
							// if they are BPR compliant, then...
							if (matrix.get(user_id,pos_item_id) > 0){
								//updateWithMatrix(user_id, pos_item_id, neg_item_id,matrix, false, 0.05);
								int[] sampleTriple = null;
								
								// get a random untinteracted item
								if (tripleCriterion.equals("unseen")) sampleTriple = sampleJ(user_id,pos_item_id, neg_item_id,matrix, true);
								else sampleTriple = sampleJ(user_id,pos_item_id, neg_item_id,matrix, false);
								
								user_id = sampleTriple[0];
								pos_item_id = sampleTriple[1];
								neg_item_id = sampleTriple[2];
								// set xscale if enabled
								if (useXscale)
									xscale = sampleTriple[3];
								
								// and finally perform a gradient descent step
								updateFactors(user_id, pos_item_id, neg_item_id, true,
										true, updateJ, xscale, 0.05, false);
							}
							
						}
						/*
						if(tripleCriterion.equals("number")){
							if (data.numberMatrixV.get(user_id,pos_item_id) > 0)
								updateWithMatrix(user_id, pos_item_id, neg_item_id,data.numberMatrixV, false, 0.05);
							if (data.numberMatrixC.get(user_id,pos_item_id) > 0)
								updateWithMatrix(user_id, pos_item_id, neg_item_id,data.numberMatrixC, false, 0.05);
							if (data.numberMatrixW.get(user_id,pos_item_id) > 0)
								updateWithMatrix(user_id, pos_item_id, neg_item_id,data.numberMatrixW, false, 0.05);
						}
						else if (tripleCriterion.equals("time")){
							if (data.numberMatrixV.get(user_id,pos_item_id) > 0)
								updateWithMatrix(user_id, pos_item_id, neg_item_id,data.timeMatrixV, false, 0.05);
							if (data.numberMatrixC.get(user_id,pos_item_id) > 0)
								updateWithMatrix(user_id, pos_item_id, neg_item_id,data.timeMatrixC, false, 0.05);
							if (data.numberMatrixW.get(user_id,pos_item_id) > 0)
								updateWithMatrix(user_id, pos_item_id, neg_item_id,data.timeMatrixW, false, 0.05);
						} else{
							//case "unseen"
							if (data.numberMatrixV.get(user_id,pos_item_id) > 0)
								updateWithMatrix(user_id, pos_item_id, neg_item_id,data.numberMatrixV, true, 0.05);
							if (data.numberMatrixC.get(user_id,pos_item_id) > 0)
								updateWithMatrix(user_id, pos_item_id, neg_item_id,data.numberMatrixC, true, 0.05);
							if (data.numberMatrixW.get(user_id,pos_item_id) > 0)
								updateWithMatrix(user_id, pos_item_id, neg_item_id,data.numberMatrixW, true, 0.05);
						}*/
					}
					
				}
			}
		}
	}

	// =====================================================================================
	// SOME MINOR HELPER METHODS FOR THE METHOD ABOVE:
	// =====================================================================================
	
	/**
	 * Helper method to populate the item 2 item auxiliary relations depending on the evaluation scheme
	 * @return a populated mcArray
	 */
	private SparseIntMatrix[] initMCArray(){
		SparseIntMatrix[] mcArray = new SparseIntMatrix[6];
		if (tripleCriterion.equals("unseen")){
			mcArray[0]= data.numberMatrixV;
			mcArray[1]= data.numberMatrixC;
			mcArray[2]= data.numberMatrixW;
			
			mcArray[3]= data.numberMatrixVR;
			mcArray[4]= data.numberMatrixDC;
			mcArray[5]= data.numberMatrixDW;
		}
		
		if (tripleCriterion.equals("number")){
			mcArray[0]= data.numberMatrixV;
			mcArray[1]= data.numberMatrixC;
			mcArray[2]= data.numberMatrixW;
			
			mcArray[3]= data.numberMatrixVR;
			mcArray[4]= data.numberMatrixDC;
			mcArray[5]= data.numberMatrixDW;
		}
		if (tripleCriterion.equals("time")){
			mcArray[0]= data.timeMatrixV;
			mcArray[1]= data.timeMatrixC;
			mcArray[2]= data.timeMatrixW;
			
			mcArray[3]= data.timeMatrixVR;
			mcArray[4]= data.timeMatrixDC;
			mcArray[5]= data.timeMatrixDW;
		}
		return mcArray;
	}
	
	/**
	 * Helper method to update the MF factors for auxiliary byte relations
	 * 
	 * @param matrix A SparseByteMatrix that represents a relation
	 */
	private void auxiliaryStep(SparseMatrix matrix){
		int[] triple = new int[4];
		triple[0] = sampleK(matrix);
			triple = sampleIJ(triple, matrix);
		updateFactors(triple[0], triple[1], triple[2], true, true, updateJ, triple[3],learnRate, true);
	}
	
	/**
	 * Helper method to update the MF factors for auxiliary u2i (integer) relations
	 * 
	 * @param matrix A SparseIntMatrix that represents a relation
	 */
	private void auxiliaryStep(SparseIntMatrix matrix){
		int[] triple = new int[4];
		triple[0] = sampleU(matrix);
		if (tripleCriterion.equals("unseen")){
			triple = sampleIJ(triple, matrix, true);
		}
		else{
			triple = sampleIJ(triple, matrix, false);
		}
		int xscale = 1;
		if(useXscale) xscale = triple[3];
		
		updateFactors(triple[0], triple[1], triple[2], true, true, updateJ, xscale,learnRate, false);
	}
	
	/**
	 * A helper method for u2i aux relations in Complete Sampling
	 * 
	 * @param user_id
	 * @param pos_item_id
	 * @param neg_item_id
	 * @param NTmatrix
	 * @param unseen
	 * @param newLearnrate
	 */
	/*@Deprecated
	public void updateWithMatrix(int user_id, int pos_item_id, int neg_item_id, SparseIntMatrix NTmatrix, boolean unseen, double newLearnrate){
		int[] sampleTriple = null;
		
		
		if (unseen) sampleTriple = sampleJ(user_id,pos_item_id, neg_item_id,NTmatrix, true);
		else sampleTriple = sampleJ(user_id,pos_item_id, neg_item_id,NTmatrix, false);
		
		int xscale = 1;
		
		user_id = sampleTriple[0];
		pos_item_id = sampleTriple[1];
		neg_item_id = sampleTriple[2];
		if (useXscale) xscale = sampleTriple[3];
		
		updateFactors(user_id, pos_item_id, neg_item_id, true,
				true, updateJ, xscale, newLearnrate);
	}*/
	
	
	// =====================================================================================
	// THE ACTUAL GRADIENT DESCENT STEPS CALLED IN THE RUNNER METHOD OF THIS CLASS:
	// =====================================================================================

	/**
	 * latent matrices and item_bias are updated according to the stochastic
	 * gradient descent update rule
	 * 
	 * this method is for updates based on user 2 item or item 2 item relations
	 * 
	 * @param u
	 *            Number - the mapped userID or initial_itemID
	 * @param i
	 *            Number - the mapped itemID of a interacted item
	 * @param j
	 *            Number - the mapped itemID of an uninteracted item
	 * @param update_u
	 *            Boolean - should u be updated
	 * @param update_i
	 *            Boolean - should i be updated
	 * @param update_j
	 *            Boolean - should j be updated
	 * @param xscale A new scaling factor
	 * @param newLearnrate In-depth-modification of the learn-rate
	 * @param item2item Is this an item2item gradient update?
	 */
	public void updateFactors(int u, int i, int j, boolean update_u,
			boolean update_i, boolean update_j, int xscale, double newLearnrate, boolean item2item) {

		double x_uij;		
		if(!item2item){
		// calculating the estimator
		x_uij = data.item_bias[i] - data.item_bias[j]
				+ data.rowScalarProductWithRowDifference(u, i, j);}
		else{ // item2item
			x_uij = data.item_bias[i] - data.item_bias[j]
					+ data.rowScalarProductWithRowDifferenceItem(u, i, j); // Item rewrite
		}

		// TODO Main use of xscale
		x_uij *= xscale;
		
		double one_over_one_plus_ex = 1 / (1 + Math.exp(x_uij));
		
		// adjust bias terms for seen item
		if (update_i) {
			double update = one_over_one_plus_ex - biasReg * data.item_bias[i];
			data.item_bias[i] += (newLearnrate * update);
		}

		// adjust bias terms for unseen item
		if (update_j) {
			double update = -one_over_one_plus_ex - biasReg * data.item_bias[j];
			data.item_bias[j] += (newLearnrate * update);
		}
		
		// adjust factors
		for (int f = 0; f < numFeatures; f++) {
			double w_uf;
			if(!item2item){
				w_uf = data.latentUserVector[u][f];
			}
			else{ // item2item
				w_uf = data.latentItemVector[u][f];
			}
			double h_if = data.latentItemVector[i][f];
			double h_jf = data.latentItemVector[j][f];

			//adjust component of user-vector / the initial_item-vector
			if (update_u) {
				double update = (h_if - h_jf) * one_over_one_plus_ex - regU * w_uf;
				if(!item2item){
					//data.latentUserVector[u][f] = (w_uf + newLearnrate *(2-(1/xscale))* update);
					data.latentUserVector[u][f] = (w_uf + newLearnrate * update);
				}
				else{ // item2item
					//data.latentItemVector[u][f] = (w_uf + newLearnrate* (2 - (1 / xscale)) * update); // REWRITTEN for i b
					data.latentItemVector[u][f] = (w_uf + newLearnrate * update);
				}
			}// TODO secondary use of xscale... does not work so good

			//adjust component of seen item-vector
			if (update_i) {
				double update = w_uf * one_over_one_plus_ex - regI * h_if;
				//data.latentItemVector[i][f] = (float) (h_if + newLearnrate*(2-(1/xscale)) * update);
				data.latentItemVector[i][f] = (float) (h_if + newLearnrate * update);
			}
			//adjust component of unseen item-vector	
			if (update_j) {
				double update = -w_uf * one_over_one_plus_ex - regJ * h_jf;
				//data.latentItemVector[j][f] = (float) (h_jf + newLearnrate*(2-(1/xscale)) * update);
				data.latentItemVector[j][f] = (float) (h_jf + newLearnrate * update);
			}
		}
	}

	// =====================================================================================
	
	/**
	 * latent matrices and item_bias are updated according to the stochastic
	 * gradient descent update rule
	 * 
	 * this method is for updates based on item 2 item relations
	 * 
	 * @param k Number - the mapped itemID of a specific item
	 * @param i Number - the mapped itemID of an item that shares a characteristic with k
	 * @param j Number - the mapped itemID of an item that does not share that characteristic with k
	 * @param update_k Boolean - should k be updated
	 * @param update_i Boolean - should i be updated
	 * @param update_j Boolean - should j be updated
	 * @param xscale A new scaling factor
	 * @param newLearnrate In-depth-modification of the learn-rate
	 */
	/*@Deprecated
	public void updateItemFactors(int k, int i, int j, boolean update_k,
			boolean update_i, boolean update_j, int xscale, double newLearnrate) {

		
	
		
		// calculating the estimator
		double x_kij = data.item_bias[i] - data.item_bias[j]
				+ data.rowScalarProductWithRowDifferenceItem(k, i, j); // Item rewrite

		x_kij *= xscale;
		
		//OK
		double one_over_one_plus_ex = 1 / (1 + Math.exp(x_kij));

		// adjust bias terms for seen item
		if (update_i) {
			double update = one_over_one_plus_ex - biasReg * data.item_bias[i];
			data.item_bias[i] += (newLearnrate * update);
		}

		// adjust bias terms for unseen item
		if (update_j) {
			double update = -one_over_one_plus_ex - biasReg * data.item_bias[j];
			data.item_bias[j] += (newLearnrate * update);
		}

		// adjust factors
		for (int f = 0; f < numFeatures; f++) {
			double w_kf = data.latentItemVector[k][f];	// rewritten for item-based
			double h_if = data.latentItemVector[i][f];
			double h_jf = data.latentItemVector[j][f];

			//adjust component of initial item-vector
			if (update_k) {
				double update = (h_if - h_jf) * one_over_one_plus_ex - regU
						* w_kf;
				data.latentItemVector[k][f] = (w_kf + newLearnrate *(2-(1/xscale))* update);	// REWRITTEN for i b
			}

			//adjust component of pos item-vector
			if (update_i) {
				double update = w_kf * one_over_one_plus_ex - regI * h_if;
				data.latentItemVector[i][f] = (float) (h_if + newLearnrate*(2-(1/xscale))
						* update);
			}
			//adjust component of neg item-vector	
			if (update_j) {
				double update = -w_kf * one_over_one_plus_ex - regJ * h_jf;
				data.latentItemVector[j][f] = (float) (h_jf + newLearnrate*(2-(1/xscale))
						* update);
			}
		}
	}*/
	
	
	// =====================================================================================
	// METHODS FOR PICKING ITEMS FROM RELATIONS (COMPLETE SAMPLING)
	// =====================================================================================
	

	//used for: Complete Sampling, target relation (P), mode unseen
	/**
	 * finds another unseen item
	 * 
	 * @param u
	 *            Number - the mapped userID
	 * @param i
	 *            Number - the mapped itemID of a viewed item
	 * @param j
	 *            Number - the mapped itemID of an unviewed item
	 * @return sampleTriple Array - an array containing the mapped userID, the
	 *         mapped view itemId and the mapped unviewed itemID
	 */
	public int[] sampleJ(int u, int i, int j) {
		int[] sampleTriple = new int[4];
		sampleTriple[0] = u;
		sampleTriple[1] = i;
		sampleTriple[2] = j;
		boolean item_is_positive = data.boolMatrix.getBool(u,i);
		if (item_is_positive == false) System.err.println("This error should never happen!");
		do
			// get another random as long as it was at least once bought
			sampleTriple[2] = random.nextInt(numItems);
		while (data.boolMatrix.getBool(u,sampleTriple[2]) == item_is_positive);

		// set xscale
		sampleTriple[3] = 1;
		
		return sampleTriple;
	}

	//used for: Complete Sampling, target relation, mode rating
		/**
		 * finds an unrated or less than avg rated item
		 * 
		 * @param u
		 *            Number - the mapped userID
		 * @param i
		 *            Number - the mapped itemID of a viewed item
		 * @param j
		 *            Number - the mapped itemID of an unviewed item
		 * @return sampleTriple Array - an array containing the mapped userID, the
		 *         mapped view itemId and the mapped unviewed itemID
		 */
		public int[] sampleJrating(int u, int i, int j) {
			int[] sampleTriple = new int[4];
			sampleTriple[0] = u;
			sampleTriple[1] = i;
			sampleTriple[2] = j;
	
			// set xscale to default when negative item is unknown
			sampleTriple[3] = 1;
			
			int positive_rating = data.dm.getRating(data.userMap.get(u),data.itemMap.get(i));
			List<Integer> user_items = data.userMatrix.get(u);
			float useravg = data.dm.getUserAverageRating(data.userMap.get(u));
			boolean breaker;
			do{
				// get another random as long as it was at least once bought
				sampleTriple[2] = random.nextInt(numItems);
				if (user_items.contains(sampleTriple[2])){
					int rating = data.dm.getRating(data.userMap.get(u),data.itemMap.get(sampleTriple[2]));
					if (rating < useravg){
						breaker = false;
						// set xscale to the difference of the items ratings, but at least 1
						sampleTriple[3] = Math.max(1, positive_rating - rating);
					}
						
					else
						breaker = true;
				}
				else
					breaker = false;
			}
			while (breaker);
			
			return sampleTriple;
		}
	
	//used for: Complete Sampling, target relation (P), mode number and mode time
	//used for: Complete Sampling, u2i aux relations (V,C,W), all three modes
	/**
	 * finds another unseen or less interacted item by given matrix
	 * more general method of that one above
	 * 
	 * @param u
	 *            Number - the mapped userID
	 * @param i
	 *            Number - the mapped itemID of a viewed item
	 * @param j
	 *            Number - the mapped itemID of an unviewed item
	 * @return sampleTriple Array - an array containing the mapped userID, the
	 *         mapped view itemId and the mapped unviewed itemID
	 */
	public int[] sampleJ(int u, int i, int j, SparseIntMatrix matrix, boolean justUnseen) {
		int[] sampleTriple = new int[4];
		sampleTriple[0] = u;
		sampleTriple[1] = i;
		sampleTriple[2] = j;
		int value_ui = matrix.get(u,i);
		
		do{
			// try another random item, as long ...
			sampleTriple[2] = random.nextInt(numItems);
			//System.out.println(" sampled randim j "+ sampleTriple[2] );
		}
		// justUnseen: ... as the random was already interacted (time / #interactions) with
		// !justUnseen: ... as the random was the same or more often (time: later) interacted with
		while ((justUnseen?1:value_ui) <= matrix.get(u,sampleTriple[2]));

		// set xscale, only for !justUnseen
		sampleTriple[3] = justUnseen?1:(value_ui - matrix.get(u,sampleTriple[2]));
		// set xscale
		//sampleTriple[3] = value_ui - matrix.get(u,sampleTriple[2]);
		// ^ that version of the code would set xscale to value_ui in unseen case. But only for V, C and W.
		//   that is not consistent
		return sampleTriple;
	}
	
	
	//used for: Complete Sampling, i2i aux relations
	/**
	 * finds another item j that does not share the same feature that k and i share
	 * @param k
	 * @param i
	 * @param j
	 * @param matrix
	 * @return
	 */
	public int[] sampleJ(int k, int i, int j, SparseMatrix matrix) {
		int[] sampleTriple = new int[4];
		sampleTriple[0] = k;
		sampleTriple[1] = i;
		sampleTriple[2] = j;
		
		boolean breaker = false;
		int xscale = 1;
		
		do{
			// try another random item, as long ...
			sampleTriple[2] = random.nextInt(numItems);
			
			if(matrix.get(k,sampleTriple[2]) instanceof Byte){
				byte jValue = (byte)matrix.get(k,sampleTriple[2]);
				byte iValue = (byte)matrix.get(k,sampleTriple[1]);
				if(iValue > jValue) breaker = true;
			}
			else if (matrix.get(k,sampleTriple[2]) instanceof Integer){
				int jValue = (int)matrix.get(k,sampleTriple[2]);
				int iValue = (int)matrix.get(k,sampleTriple[1]);
				if(iValue > jValue) breaker = true;
				xscale = iValue - jValue;
			}
			else {System.err.println("WRONG TYPE! 1");return null;}
			
		}
		// ... as the random still has that specific feature
		while (!breaker);

		// set xscale
		sampleTriple[3] = xscale;
		return sampleTriple;
	}
	
	
	
	
	// =====================================================================================
	// METHODS FOR PICKING ITEMS FROM RELATIONS (UNIFORM USER SAMPLING)
	// =====================================================================================
	// (Methods for picking the left side of triples, that is u or k)
	
	//used for: Uniform User Sampling, target relation (P), mode unseen
	/**
	 * finds an user who has viewed at least one item but not all
	 * 
	 * This is basically an old (more specific) version of the method below
	 * 
	 * @return u Number - the mapped userID
	 */
	public int sampleU() {
		while (true) {
			// get a random user
			int u = random.nextInt(numUsers);
			// take a new one if he is not in the user matrix (list with users -> interacted items)
			if (!data.userMatrix.containsKey(u))
				continue;
			List<Integer> viewedItemsList = data.userMatrix.get(u);

			// choose that user if he has min 1, max n-1 interacted items
			if (viewedItemsList == null || viewedItemsList.size() == 0
					|| viewedItemsList.size() == numItems)
				continue;
			return u;
		}
	}

	//used for: Uniform User Sampling, target relation (P), mode number and mode time
	//used for: Uniform User Sampling, u2i aux relations (V,C,W), all three modes
	/**
	 * finds an i that has a value != null for at least one item but not all items at the same value
	 * this is the generalized version of the method above
	 * 
	 * @return u Number - the mapped userID
	 */
	public int sampleU(SparseIntMatrix matrix) {
		
		while (true) {
			// choose a random user
			int i = random.nextInt(matrix.getM());
//			System.out.println(" - Sampled user "+i);
			// choose again if he has not interacted with items
			if (matrix.getRow(i) == null)
				continue;
			
			TIntIntMap ithRow = matrix.getRow(i);

			// these two conditions need to be checked for the users list of items
			boolean oneIsNotZero = false;
			boolean oneIsDifferent = false;
			
			int temp = ithRow.get(0);
			
			// the check is done in this loop
			for (int j = 0; j < matrix.getN(); j++){
				int ij = ithRow.get(j);
				if(ij != 0) {
					oneIsNotZero = true;
//					System.out.println(" - - Item "+j+" was not zero");
					}
				if(ij != temp) {
					oneIsDifferent = true;
//					System.out.println(" - - Item "+j+" was different from j-1");
					}
				temp = ij;
			}
			
			// choose the user, if both conditions apply
			if (oneIsDifferent && oneIsNotZero)
				return i;
			continue;
		}

	}
	
	//used for: Uniform User Sampling, i2i aux relations
	/**
	 * finds an i that has a value != null for at least one item but not all items at the same value
	 * this is the generalized version of the method above
	 * 
	 * @return u Number - the mapped userID
	 */
	public int sampleK(SparseMatrix matrix) {
		
		while (true) {
			// choose a random item
			int i = random.nextInt(matrix.getM());
//			System.out.println(" - Sampled item "+i);
			// choose again if he has not interacted with items
			if (matrix.getRow(i) == null)
				continue;
			
			Object ithRow = matrix.getRow(i);
			boolean breaker;
			
			// these two conditions need to be checked for the items list of items
			boolean oneIsNotZero = false;
			boolean oneIsDifferent = false;
			
			if(ithRow instanceof TIntByteMap || ithRow instanceof TIntByteHashMap){// ???
				TIntByteMap ithRowByte = (TIntByteMap)ithRow;
				byte temp = ithRowByte.get(0);
				
				// the check is done in this loop
				for (int j = 0; j < matrix.getN(); j++){
					byte ij = ithRowByte.get(j);
					if(ij != 0) {
						oneIsNotZero = true;
//						System.out.println(" - - Item "+j+" was not zero");
						}
					if(ij != temp) {
						oneIsDifferent = true;
//						System.out.println(" - - Item "+j+" was different from j-1");
						}
					temp = ij;
				}
			}
			else if(ithRow instanceof TIntIntMap || ithRow instanceof TIntIntHashMap){
				TIntIntMap ithRowInt = (TIntIntMap)ithRow;
				int temp = ithRowInt.get(0);
				
				// the check is done in this loop
				for (int j = 0; j < matrix.getN(); j++){
					int ij = ithRowInt.get(j);
					if(ij != 0) {
						oneIsNotZero = true;
//						System.out.println(" - - Item "+j+" was not zero");
						}
					if(ij != temp) {
						oneIsDifferent = true;
//						System.out.println(" - - Item "+j+" was different from j-1");
						}
					temp = ij;
				}
			}
			else{
				System.err.println(ithRow.getClass());
				System.err.println("WRONG TYPE! 2");
				return -1;
			}
			
			
			// choose the item, if both conditions apply
			if (oneIsDifferent && oneIsNotZero)
				return i;
			continue;
		}

	}

	// =====================================================================================
	// (Methods for picking the right side of triples, that is i and j)
	
	//used for: Uniform User Sampling, target relation (P), mode unseen
	/**
	 * finds a seen item and an unseen item
	 * 
	 * This method is basically an old (more specific) version of sampleRightPair(...,...,true)
	 * 
	 * @return sampleTriple Array - an array containing the mapped userID, the
	 *         mapped view itemId and the mapped unviewed itemID
	 */
	public int[] sampleIJ(int[] triple) {
		int u = triple[0];

		List<Integer> user_items = data.userMatrix.get(u);
		triple[1] = user_items.get((random.nextInt(user_items.size())));
		do
			triple[2] = random.nextInt(numItems);
		while (user_items.contains(triple[2]));

		return triple;
	}
	
	//used for: Uniform User Sampling, target relation, mode rating
		/**
		 * finds a positve and a negative item
		 * 
		 * @return sampleTriple Array - an array containing the mapped userID, the
		 *         mapped view itemId and the mapped unviewed itemID
		 */
		public int[] sampleIJrating(int[] triple) {
			int u = triple[0];
			int neg_rating;
			float useravg = data.dm.getUserAverageRating(data.userMap.get(u));
			int pos_rating;
			
			// default xscale value
			triple[3] = 1;
			
			
			
			// get an item that has at least user avg rating
			List<Integer> user_items = data.userMatrix.get(u);
			do{
				triple[1] = user_items.get((random.nextInt(user_items.size())));
				
				pos_rating = data.dm.getRating(data.userMap.get(u),data.itemMap.get(triple[1]));
				
				if(debug.contains("$anyrated$")){
					break; // take the first one, rating doesnt matter
				}
			}
			while(pos_rating < useravg); //there should be at least one item for each user that breaks this loop
			
			// get an item that has no rating or a rating smaller than user avg
			// TODO other idea: change it so it must just be smaller that item 1
			boolean breaker;
			int debugcount = 0;
			boolean debugswitch = false;
			
			do{
				if(debug.contains("$force50percent$")){
					debugswitch = random.nextBoolean();
				}
				if(debug.contains("$force25percent$")){
					if(random.nextInt(4) == 0)
						debugswitch = true;
					else debugswitch = false;
				}
				if(debug.contains("$force75percent$")){
					if(random.nextInt(4) == 0)
						debugswitch = false;
					else debugswitch = true;
				}
				if ((debug.contains("$forcerated$") && debugcount < 5) || debugswitch){
					triple[2] = user_items.get((random.nextInt(user_items.size())));
					debugcount++; // otherwise we can get wake-locked
				}
				else{
				triple[2] = random.nextInt(numItems);
				}
				if (user_items.contains(triple[2])){
					neg_rating = data.dm.getRating(data.userMap.get(u),data.itemMap.get(triple[2]));
					
					if(debug.contains("$anyrated$") || debug.contains("$smallerthani$")){
						if (neg_rating < pos_rating){
							breaker = false;
							triple[3] = Math.max(1, pos_rating - neg_rating);
						}
							
						else
							breaker = true;
					}
					else{
					if (neg_rating < useravg){
						breaker = false;
						triple[3] = Math.max(1, pos_rating - neg_rating);
					}
						
					else
						breaker = true;
					}
				}
				else
					breaker = false;
				
			}
			while (breaker);
			return triple;
		}
	
	//used for: Uniform User Sampling, target relation (P), mode number and mode time
	//used for: Uniform User Sampling, u2i aux relations (V,C,W), all three modes
	/**
	 * finds an interacted and a less interacted item
	 * 
	 * @return sampleTriple Array - an array containing the mapped userID, the
	 *         mapped view itemId and the mapped unviewed itemID
	 */
	public int[] sampleIJ(int[] triple, SparseIntMatrix matrix, boolean justUnseen) {
		int i = triple[0];
		int j1; //I
		int j2; //J
//		System.out.println(" - Looking for items for user "+i);
//		System.out.println(" - N is "+matrix.getN());
		TIntIntMap ithRow = matrix.getRow(i);
		do{
			j1 = random.nextInt(matrix.getN());
			//System.out.println(" - - rand: " + randomnumber);
			//j1 = ithRow.get(random.nextInt(matrix.getN()));
//			System.out.println(" - - guessing item 1 as "+j1+ " with value "+ithRow.get(j1));
		}
		while (ithRow.get(j1) == 0);
//		System.out.println(" - Chose item 1 as "+j1+ " with value "+ithRow.get(j1));
		do{
			j2 = random.nextInt(matrix.getN());
			//j2 = ithRow.get((random.nextInt(matrix.getN())));
//			System.out.println(" - - guessing item 2 as"+j2+ " with value "+ithRow.get(j2));
		}
		while ((justUnseen?1:ithRow.get(j1)) <= ithRow.get(j2));
		
//		System.out.println(" - Chose item 2 as "+j2+ " with value "+ithRow.get(j2));
		triple[1] = j1;
		triple[2] = j2;
//		System.out.println(" - Triple is "+i+ " "+j1+" "+j2);
		
		// Set xscale
		if (useXscale)
			triple[3] = justUnseen?1:(ithRow.get(j1) - ithRow.get(j2));
		return triple;
	}
	
	//used for: Uniform User Sampling, i2i aux relations
	/**
	 * finds an interacted and a less interacted item
	 * 
	 * @return sampleTriple Array - an array containing the mapped userID, the
	 *         mapped view itemId and the mapped unviewed itemID
	 */
	public int[] sampleIJ(int[] triple, SparseMatrix matrix) {
		int i = triple[0];
		int j1;
		int j2;
		int xscale = 1;
//		System.out.println(" - Looking for items for user "+i);
//		System.out.println(" - N is "+matrix.getN());
		Object ithRow = matrix.getRow(i);
		boolean breaker;
		do{
			j1 = random.nextInt(matrix.getN());
			//System.out.println(" - - rand: " + randomnumber);
			//j1 = ithRow.get(random.nextInt(matrix.getN()));
//			System.out.println(" - - guessing item 1 as "+j1+ " with value "+ithRow.get(j1));
			if(ithRow instanceof TIntByteMap){breaker = (((TIntByteMap)ithRow).get(j1) == 0);}
			else if(ithRow instanceof TIntIntMap){breaker = (((TIntIntMap)ithRow).get(j1) == 0);}
			else{System.err.println("WRONG TYPE! 3"); return null;}
		}
		while (breaker);
//		System.out.println(" - Chose item 1 as "+j1+ " with value "+ithRow.get(j1));
		do{
			j2 = random.nextInt(matrix.getN());
			//j2 = ithRow.get((random.nextInt(matrix.getN())));
//			System.out.println(" - - guessing item 2 as"+j2+ " with value "+ithRow.get(j2));
			if(ithRow instanceof TIntByteMap){breaker = (((TIntByteMap)ithRow).get(j2) != 0);}
			else if(ithRow instanceof TIntIntMap){
				breaker = (((TIntIntMap)ithRow).get(j2) >= ((TIntIntMap)ithRow).get(j1));
				xscale = ((TIntIntMap)ithRow).get(j1) - ((TIntIntMap)ithRow).get(j2);
				}
			else{System.err.println("WRONG TYPE! 4"); return null;}
		}
		while (breaker);
//		System.out.println(" - Chose item 2 as "+j2+ " with value "+ithRow.get(j2));
		triple[1] = j1;
		triple[2] = j2;
//		System.out.println(" - Triple is "+i+ " "+j1+" "+j2);
		
		triple[3] = xscale;
		return triple;
	}
	
	// =====================================================================================
	// SETTERS and GETTERS:
	// =====================================================================================

	/**
	 * Setter for factory
	 * 
	 * @param n
	 */
	public void setNumFeatures(String n) {
		this.numFeatures = Integer.parseInt(n);
	}

	/**
	 * Setter for regI
	 * 
	 * @param n
	 */
	public void setRegI(String n) {
		this.regI = Double.parseDouble(n);
	}
	
	/**
	 * Setter for regJ
	 * 
	 * @param n
	 */
	public void setRegJ(String n) {
		this.regJ = Double.parseDouble(n);
	}
	
	/**
	 * Setter for regU
	 * 
	 * @param n
	 */
	public void setRegU(String n) {
		this.regU = Double.parseDouble(n);
	}
	
	/**
	 * Setter for UpdateJ
	 * 
	 * @param n
	 */
	public void setUpdateJ(String n) {
		this.updateJ = Boolean.parseBoolean(n);
	}
	
	
	/**
	 * Setter for BiasReg
	 * 
	 * @param n
	 */
	public void setBiasReg(String n) {
		this.biasReg = Double.parseDouble(n);
	}
	
	/**
	 * Setter for LearnRate
	 * 
	 * @param n
	 */
	public void setLearnRate(String n) {
		this.learnRate = Double.parseDouble(n);
	}
	

	/**
	 * Setter for the initial steps
	 * 
	 * @param n
	 */
	public void setInitialSteps(String n) {
		this.initialSteps = Integer.parseInt(n);
	}
	
	/**
	 * Setter for the uniform Sampling
	 * 
	 * @param n
	 */
	public void setUniformSampling(String n) {
		this.UniformUserSampling = Boolean.parseBoolean(n);
	}
	
	@Override
	public int getDurationEstimate() {
		return 3;
	}
	
	/**
	 * Should the global relevance threshold be chosen or not
	 * @param u
	 */
	public void setUseRelevanceThreshold(String u)  throws Exception {
		data.useRatingThreshold = Boolean.parseBoolean(u);
	}
	
	/**
	 * Whats the criterion for triples x_uij
	 * @param u
	 */
	public void setTripleCriterion(String u){
		this.tripleCriterion = u;
	}
	
	/**
	 * Sets the useMaps parameter
	 * @param b
	 */
	public void setUseUIMaps(String b){
		useUIMaps = Boolean.parseBoolean(b);
		// also set this for the datamanagement instance
		data.useUIMaps = useUIMaps;
	}
	
	/**
	 * Sets the useMaps parameter
	 * @param b
	 */
	public void setSelectIIMaps(String b){
		if(b != null && !b.equals("")) useIIMaps = true;
		// also set this for the datamanagement instance
		data.selectIIMaps = b;
		data.useIIMaps = useIIMaps;
	}
	
	/**
	 * Sets the useXscale parameter
	 * @param b
	 */
	public void setUseXscale(String b){
		useXscale = Boolean.parseBoolean(b);
		
	}
	
	/**
	 * Sets the useXscale parameter
	 * @param b
	 */
	public void setSequentialLearning(String b){
		SequentialLearning = Boolean.parseBoolean(b);
		
	}
	
	public void setZalandoMode(String b){
		zalandoMode = Boolean.parseBoolean(b);
		data.zalandoMode = zalandoMode;
	}
	
	public void setDebug(String b){
		debug = b;
	}

}