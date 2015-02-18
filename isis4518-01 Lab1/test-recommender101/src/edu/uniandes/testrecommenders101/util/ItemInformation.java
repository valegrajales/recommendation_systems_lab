package edu.uniandes.testrecommenders101.util;

import java.util.LinkedList;

public class ItemInformation implements Comparable<ItemInformation>{

	private long itemId;
	private String itemName;
	private String itemURL;
	private LinkedList<String> itemGenres;
	private long numRatings;
	private double itemAverage;

	

	public ItemInformation(long itemId, String itemName, String itemURL,
			LinkedList<String> itemGenres, int numRatings, double itemAverage) {
		
		this.itemId=itemId;
		this.itemName=itemName;
		this.itemURL=itemURL;
		this.itemGenres=itemGenres;
		this.numRatings=numRatings;
		this.itemAverage=itemAverage;
	}

	@Override
	/**
	 * Default comparator compares based on numRatings for item
	 */
	public int compareTo(ItemInformation o) {
	
		return Long.compare(this.numRatings, o.numRatings);
	}

	public long getItemId() {
		return itemId;
	}

	public String getItemName() {
		return itemName;
	}

	public double getItemAverage() {
		return itemAverage;
	}

	public String getItemURL() {
		return itemURL;
	}

	public LinkedList<String> getItemGenres() {
		return itemGenres;
	}

	public long getNumRatings() {
		return numRatings;
	}
	
	public void setNumRatings(long numRatings) {
		this.numRatings = numRatings;
	}
	
	@Override
	public String toString() {
		String ret="id:"+this.itemId+" "+this.itemName+'\t'+itemURL+'\t'+numRatings+'\t'+itemAverage+'\t'+"Genres:";
		for (String genreName : itemGenres) {
			ret+=genreName+' ';
		}
		return ret.trim();
	}

}
