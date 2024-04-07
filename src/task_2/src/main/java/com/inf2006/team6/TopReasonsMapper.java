package com.inf2006.team6;

import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.log4j.Logger;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import java.io.StringReader;

public class TopReasonsMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
    private static final Logger LOG = Logger.getLogger(TopReasonsMapper.class);
    private static final IntWritable one = new IntWritable(1);
    private Text compositeKey = new Text();
    private boolean isHeaderProcessed = false;

    @Override
    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        // skip header line
        if (!isHeaderProcessed) {
            isHeaderProcessed = true;
            return;
        }

        LOG.info("Mapping task started for key: " + key.toString());

        CSVParser parser = CSVFormat.DEFAULT.withTrim().parse(new StringReader(value.toString()));

        // loop CSV record
        for (CSVRecord record : parser) {

            String airline = record.get(12).trim();
            String negativeReason1 = record.get(22).trim();
            String negativeReason2 = record.get(23).trim();

            if (airline.isEmpty() || airline.equals("NULL")
                    || (negativeReason1.isEmpty() || negativeReason1.equals("NULL")
                            || negativeReason1.equals("Unknown"))
                            && (negativeReason2.isEmpty() || negativeReason2.equals("NULL")
                                    || negativeReason2.equals("Unknown"))) {
                continue;
            }

            // If negativeReason1 is "Unknown", use negativeReason2 if it is valid
            String negativeReason =
                    "Unknown".equals(negativeReason1) && !negativeReason2.equals("NULL")
                            ? negativeReason2
                            : negativeReason1;

            // Only write out to context if the final negative reason is valid
            if (!negativeReason.isEmpty() && !negativeReason.equals("NULL")) {
                compositeKey.set(airline + "_" + negativeReason);
                context.write(compositeKey, one);
            }
        }

        LOG.info("Mapping task completed for key: " + key.toString());
    }
}
