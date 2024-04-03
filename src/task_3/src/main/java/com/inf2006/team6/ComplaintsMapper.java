package com.inf2006.team6;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.Hashtable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

/**
 * ComplaintsMapper class for task 3.
 */
public class ComplaintsMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
    Hashtable<String, String> countryCodesMap = new Hashtable<>(); // Hashtable to store country
                                                                   // codes to country names.

    /**
     * Setup function for ComplaintsMapper, uses the distributed cache to get the country codes.
     * 
     * @param context context of the job.
     * @throws IOException if an I/O error occurs.
     */
    @Override
    protected void setup(Mapper<LongWritable, Text, Text, IntWritable>.Context context)
            throws IOException {
        // Use the distributed cache to get the country codes.
        //
        if (context.getCacheFiles() != null && context.getCacheFiles().length > 0) {
            URI countryCodesPath = context.getCacheFiles()[0];
            File countryCodesFile = new File(countryCodesPath.getPath());
            BufferedReader br = new BufferedReader(new FileReader(countryCodesFile));
            String line = null;
            while (true) {
                line = br.readLine();
                if (line != null) {
                    String[] parts = line.split("\t");
                    countryCodesMap.put(parts[0], parts[1]);
                } else {
                    break; // finished reading
                }
            }
            br.close();
        }
    }

    /**
     * Map function for ComplaintsMapper.
     * 
     * @param key line number.
     * @param value line of text.
     * @param context context of the job.
     * @throws IOException if an I/O error occurs.
     * @throws InterruptedException if the thread is interrupted.
     */
    @Override
    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {

        // Splits the line by commas and retrieves the country and sentiment.
        //
        String[] parts = value.toString().split(",");
        String countryCode = parts[10];
        SentimentEnum sentiment = SentimentEnum.valueOf(parts[14].toUpperCase());
        SentimentEnum targetSentiment =
                SentimentEnum.valueOf(context.getConfiguration().get("sentiment").toUpperCase());

        // If the country and sentiment are not null, and the sentiment is the target sentiment,
        // write the country and 1 to the context.
        //
        if (countryCode != null && sentiment != null) {
            String country = countryCodesMap.get(countryCode);
            if (country != null && sentiment == targetSentiment) {
                context.write(new Text(country), new IntWritable(1));
            }
        }
    }
}
