package assignment4.ttlsda;



import java.util.ArrayList;


import java.util.List;



import org.apache.spark.SparkConf;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import scala.Tuple2;

public class Assignment4KMeans {

	public static void main(String[] args)	{
        System.setProperty("hadoop.home.dir", "C:/winutils");
        //Initialising spark
        JavaSparkContext jsc = new JavaSparkContext( new SparkConf()
        		.setAppName("JavaKMeansExample")
                .setMaster("local[2]")
                .set("spark.executor.memory","1g")
                .set("spark.driver.memory", "1g"));
        //Taking the location of the data into 'path'
        String path = "C:\\#####\\twitter2D.txt";
        //Creating JavaRDD<String> from the 'twitter2D.txt'
        JavaRDD<String> data = jsc.textFile(path);
        //Parsing the loaded data.
		JavaRDD<Tuple2<Vector, String>> parsedData = data.map((String s) -> {
			//Splitting the data on the basis of comma delimiter
			String[] sarray = s.split(",");
			//creating an array to store the two world coordinates
			double[] values = new double[2];			
			//parsing the string as double and storing it in 'values'
			for (int i = 0; i < 2; i++){
				values[i] = Double.parseDouble(sarray[i]);
			}
			//storing the string ,i.e. the tweet as a String 
			String m=sarray[sarray.length-1];
			//Returning the world coordinates and tweet as Tuple
			return new Tuple2<>(Vectors.dense(values),m);
		});
		
		// creating an arraylist of vectors
		ArrayList<Vector> listOfVectors = new ArrayList<Vector>();
		//extracting the vectors from the list of parsedData.collect() and 
		//storing it in a listOfVectors
		for(int i = 0; i < parsedData.collect().size(); i++)
			listOfVectors.add(parsedData.collect().get(i)._1);
		//further operations on the RDDs would be cached
		jsc.parallelize(listOfVectors).rdd().cache();
		// Cluster the data into 4 classes using KMeans
		int numClusters = 4;
		int numIterations = 20;
		//using KMeansModel to train the vectors
		KMeansModel clusters = KMeans.train(jsc.parallelize(listOfVectors).rdd(), numClusters, numIterations);
        //Return the K-means cost (sum of squared distances of points to their nearest center) 
		//for this model on the given data.
		double WSSSE = clusters.computeCost(jsc.parallelize(listOfVectors).rdd());
		
		System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
	    
		//converting the parsedData as a List
		List<Tuple2<Vector, String>> output = parsedData.collect();
     
		//creating a new list which would store
		//as tuple2 data , containing tweets and their respective clusters.
		List<Tuple2<Integer,String>> clustered = new ArrayList<Tuple2<Integer,String>>();
		//adding cluster and tweet
		for(Tuple2<?,?> value : output)
			clustered.add(new Tuple2<>(clusters.predict((Vector)value._1),(String)value._2));
		//storing the sorted data in sorted (JavaRDD<Tuple2<Integer,String>>)
		JavaRDD<Tuple2<Integer,String>> sorted = jsc.parallelize(clustered).sortBy(p -> p._1, true, jsc.parallelize(clustered).partitions().size());
		//storing the sorted RDDs as list 
		List<Tuple2<Integer,String>> finalSortedList =  sorted.collect();
		//Printing out the finalSortedList in the desired form.
		for(int i=0; i<finalSortedList.size();i++ )
			System.out.println("Tweet "+"\""+finalSortedList.get(i)._2+"\""+" is in cluster "+finalSortedList.get(i)._1);
		
		
		//closing the java spark context
		jsc.close();
    
	}
}