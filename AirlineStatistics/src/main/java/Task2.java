import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Task2 {
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "airline sentiment analysis");

        job.setJarByClass(Task2.class);
        job.setMapperClass(TopReasonsMapper.class);
        job.setReducerClass(TopReasonsReducer.class);

        job.setOutputKeyClass(Text.class); 
        job.setOutputValueClass(Text.class); 

       
        job.setMapOutputKeyClass(Text.class); 
        job.setMapOutputValueClass(IntWritable.class); 

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

