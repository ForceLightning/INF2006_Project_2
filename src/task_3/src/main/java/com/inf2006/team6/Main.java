package com.inf2006.team6;

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
