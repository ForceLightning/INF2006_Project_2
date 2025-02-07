package com.inf2006.team6;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Logger;

/**
 * Reducer class for Task 2.
 */
public class TopReasonsReducer extends Reducer<Text, IntWritable, Text, Text> {
    /**
     * Logger for the class.
     */
    private static final Logger LOG = Logger.getLogger(TopReasonsReducer.class);

    /**
     * TreeMap to store the top reasons for the sentiment.
     */
    private TreeMap<Integer, String> topReasons = new TreeMap<>();

    /**
     * HashMap to store the top reasons for each airline.
     */
    private HashMap<String, TreeMap<Integer, String>> topReasonsPerAirline = new HashMap<>();

    /**
     * Reduce function for TopReasonsReducer. This function aggregates the count of each negative
     * reason for each airline.
     *
     * @param key composite key.
     * @param values list of values.
     * @param context context of the job.
     * @throws IOException if an I/O error occurs.
     * @throws InterruptedException if the thread is interrupted.
     */
    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context)
            throws IOException, InterruptedException {
        LOG.info("Reduce task started for key: " + key.toString());
        int sum = 0;
        for (IntWritable val : values) {
            sum += val.get();
        }

        // split the composite key to get the airline and the negative reason
        String[] parts = key.toString().split("_", 2);
        String airline = parts[0];
        String negativeReason = parts[1];

        if (!topReasonsPerAirline.containsKey(airline)) {
            topReasonsPerAirline.put(airline, new TreeMap<Integer, String>());
        }

        topReasons = topReasonsPerAirline.get(airline);
        topReasons.put(sum, negativeReason);

        // Keep only the top 5 reasons
        if (topReasons.size() > 5) {
            topReasons.remove(topReasons.firstKey());
        }

        topReasonsPerAirline.put(airline, topReasons);
        LOG.info("Reduce task completed for key: " + key.toString() + " with count: " + sum);
    }

    /**
     * Cleanup function for TopReasonsReducer. This function writes the final output, with the
     * airline as the key and the composite value of the negative reason and the count.
     *
     * @param context context of the job.
     * @throws IOException if an I/O error occurs.
     * @throws InterruptedException if the thread is interrupted.
     */
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        LOG.info("Cleanup started. Writing final output.");
        for (Map.Entry<String, TreeMap<Integer, String>> airlineEntry : topReasonsPerAirline
                .entrySet()) {
            String airline = airlineEntry.getKey();
            TreeMap<Integer, String> reasons = airlineEntry.getValue();
            // loop in descending order to get the top 5 reasons
            for (Map.Entry<Integer, String> entry : reasons.descendingMap().entrySet()) {
                int count = entry.getKey();
                String reason = entry.getValue();
                context.write(new Text(airline), new Text(reason + "_" + count));
            }
        }
        LOG.info("Cleanup completed. Final output written.");
    }
}
