package com.inf2006.team6;

import java.io.IOException;
import java.util.Comparator;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * Reducer class for Airline complaints and Country of origin.
 */
public class ComplaintsReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

    /**
     * Sorts a map by values in descending order
     * 
     * @param map Map to be sorted in descending order
     * @return SortedSet of the map entries sorted by values in descending order
     */
    private static <K, V extends Comparable<? super V>> SortedSet<Map.Entry<K, V>> entriesSortedByValuesDescending(
            Map<K, V> map) {
        SortedSet<Map.Entry<K, V>> sortedEntries =
                new TreeSet<Map.Entry<K, V>>(new Comparator<Map.Entry<K, V>>() {
                    @Override
                    public int compare(Map.Entry<K, V> e1, Map.Entry<K, V> e2) {
                        int res = e2.getValue().compareTo(e1.getValue());
                        return res != 0 ? res : 1;
                    }
                });
        sortedEntries.addAll(map.entrySet());
        return sortedEntries;
    }

    private int n; // top N of output
    private TreeMap<String, Integer> countryComplaintsMap; // Map of country to number of complaints

    /**
     * Setup method to initialize the reducer.
     * 
     * @param context Context of the job.
     * @throws IOException If an I/O error occurs.
     * @throws InterruptedException If the thread is interrupted.
     */
    @Override
    public void setup(Context context) throws IOException, InterruptedException {
        n = Integer.parseInt(context.getConfiguration().get("N"));
        countryComplaintsMap = new TreeMap<String, Integer>();
    }

    /**
     * Reduce method to count the number of complaints from each country.
     * 
     * @param key Country of origin of the complaint.
     * @param values Number of complaints from the country.
     * @param context Context of the job.
     * @throws IOException If an I/O error occurs.
     * @throws InterruptedException If the thread is interrupted.
     */
    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context)
            throws IOException, InterruptedException {
        int countryCount = 0;
        for (IntWritable value : values) {
            countryCount += value.get();
        }
        countryComplaintsMap.put(key.toString(), countryCount);
    }


    /**
     * Cleanup method to write the top N countries with the most complaints.
     * 
     * @param context Context of the job.
     * @throws IOException If an I/O error occurs.
     * @throws InterruptedException If the thread is interrupted.
     */
    @Override
    public void cleanup(Context context) throws IOException, InterruptedException {
        SortedSet<Map.Entry<String, Integer>> sortedCountryComplaintsMap =
                entriesSortedByValuesDescending(countryComplaintsMap);
        // Write the top N countries with the most complaints. If N is -1, write all countries.
        for (Map.Entry<String, Integer> entry : sortedCountryComplaintsMap) {
            if (n > 0 || n == -1) {
                context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
                if (n > 0) {
                    n--;
                }
            }
        }
    }
}
