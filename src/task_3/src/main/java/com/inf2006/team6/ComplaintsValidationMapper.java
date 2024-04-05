package com.inf2006.team6;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

/**
 * Validates the input data for complaints.
 */
public class ComplaintsValidationMapper extends Mapper<LongWritable, Text, LongWritable, Text> {
    /**
     * Maps the input data to the output if it is valid. {@link #isValid(String)}
     *
     * @param key The line number.
     * @param value The line of text.
     * @throws IOException if an I/O error occurs.
     * @throws InterruptedException if the thread is interrupted.
     */
    @Override
    protected void map(LongWritable key, Text value,
            Mapper<LongWritable, Text, LongWritable, Text>.Context context)
            throws IOException, InterruptedException {
        if (isValid(value.toString())) {
            context.write(key, value);
        }
    }

    /**
     * Checks if the input data is valid. The input data is valid if it has 27 columns and is not
     * the header.
     *
     * @param line The line of text.
     * @return True if the input data is valid, false otherwise.
     */
    private boolean isValid(String line) {
        String[] parts = line.split(",");
        return parts.length == 21 && !parts[0].equals("_country");
    }
}
