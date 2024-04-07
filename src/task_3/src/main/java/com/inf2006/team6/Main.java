package com.inf2006.team6;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.chain.ChainMapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

/**
 * Main class to execute the job.
 */
public class Main {
    /**
     * Gets the index of a column in the header.
     *
     * @param header Header line.
     * @param columnName Name of the column.
     * @return Index of the column.
     * @throws RuntimeException If the column is not found.
     */
    private static int getColumnIndex(String[] header, String columnName) throws RuntimeException {
        for (int i = 0; i < header.length; i++) {
            if (header[i].toLowerCase().equals(columnName.toLowerCase())) {
                return i;
            }
        }
        throw new RuntimeException("Column " + columnName + " not found in header.");
    }

    /**
     * Sets the desired column indices in the configuration.
     *
     * @param conf Configuration object.
     * @param input Input path.
     * @return Configuration object with the column indices set.
     * @throws RuntimeException If an error occurs.
     */
    private static Configuration setColumnIndices(Configuration conf, Path input)
            throws RuntimeException {
        // Set the default values for the column indices.
        //
        conf.setInt("sentiment_idx", 11);
        conf.setInt("country_idx", 0);
        conf.set("header_column_idx_0", "_country");
        conf.setInt("num_columns", 21);

        // Attempt to read the header line from the input file.
        //
        try {
            BufferedReader br = new BufferedReader(new FileReader(input.toString()));
            String[] headerLine = br.readLine().split(",");

            // Set the column indices in the configuration. We need the index of the sentiment and
            // country columns, as well as an identifying column to be used as a check against the
            // header.
            //
            conf.set("header_column_idx_0", headerLine[0]);
            conf.setInt("num_columns", headerLine.length);

            // We perform the following operations after the header-relevant configuration
            // parameters have been set, so that any exceptions thrown during the process can be
            // caught and handled by the calling function after the configuration has been updated.
            //
            int sentimentIndex = getColumnIndex(headerLine, "airline_sentiment");
            int countryIndex = getColumnIndex(headerLine, "iso3");
            conf.setInt("sentiment_idx", sentimentIndex);
            conf.setInt("country_idx", countryIndex);

            // Print the column indices and the header column 0.
            //
            System.out.println("Sentiment index: " + sentimentIndex);
            System.out.println("Country index: " + countryIndex);
            System.out.println("Header column 0: " + headerLine[0]);
            System.out.println("Number of columns: " + headerLine.length);
            br.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            throw new RuntimeException("Error finding file " + input.toString());
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Error closing file " + input.toString());
        } catch (RuntimeException e) {
            e.printStackTrace();
            throw e;
        }
        return conf;
    }

    /**
     * Main method to execute the job.
     *
     * @param args Command line arguments
     * @throws Exception If an error occurs
     */
    public static void main(String[] args) throws Exception {
        // Initialise the configuration and parse the command line arguments.
        //
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        Path input = new Path(otherArgs[0]);
        Path output = new Path(otherArgs[1]);
        output.getFileSystem(conf).delete(output, true); // Delete the output directory if it exists
        String countryCodes = otherArgs[2];

        conf.set("N", otherArgs[3]);

        // Set the column indices in the configuration.
        //
        conf = setColumnIndices(conf, input);

        // Create the job instance.
        Job job = Job.getInstance(conf, "Complaints Analytics");
        job.setJarByClass(Main.class);

        // Add the country codes file to the distributed cache.
        //
        job.addCacheFile(new URI(countryCodes));

        // Configure the chain of mappers.
        //
        Configuration validationConf = new Configuration(false);
        ChainMapper.addMapper(job, ComplaintsValidationMapper.class, LongWritable.class, Text.class,
                LongWritable.class, Text.class, validationConf);
        Configuration complaintsConf = new Configuration(false);
        complaintsConf.set("sentiment", otherArgs[4]);
        ChainMapper.addMapper(job, ComplaintsMapper.class, LongWritable.class, Text.class,
                Text.class, IntWritable.class, complaintsConf);

        // Set Mapper and Reducer classes.
        //
        job.setMapperClass(ChainMapper.class);
        job.setReducerClass(ComplaintsReducer.class);

        // Set output formats.
        //
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        // Set input and output paths.
        //
        FileInputFormat.addInputPath(job, input);
        FileOutputFormat.setOutputPath(job, output);

        // Execute the job and wait for completion.
        //
        boolean status = job.waitForCompletion(true);
        System.exit(status ? 0 : 1);
    }
}
