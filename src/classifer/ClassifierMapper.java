package classifer;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.bayes.Algorithm;
import org.apache.mahout.classifier.bayes.BayesAlgorithm;
import org.apache.mahout.classifier.bayes.BayesParameters;
import org.apache.mahout.classifier.bayes.CBayesAlgorithm;
import org.apache.mahout.classifier.bayes.ClassifierContext;
import org.apache.mahout.classifier.bayes.Datastore;
import org.apache.mahout.classifier.bayes.InMemoryBayesDatastore;
import org.apache.mahout.classifier.bayes.InvalidDatastoreException;
import org.apache.mahout.common.nlp.NGrams;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

import com.sun.media.sound.InvalidDataException;

public class ClassifierMapper extends Mapper<Text, Text, Text, IntWritable> {

	private Text outKey = new Text();
	private static final IntWritable ONE = new IntWritable(1);
	
	private int gramSize = 1;
	

	private ClassifierContext classifierContext;
	private String defaultCategory;
	
	private static final Logger log = LoggerFactory.getLogger(ClassifierMapper.class);
	
	@Override
	protected void map(Text key, Text value, Context context)
			throws IOException, InterruptedException {
		String docLabelString = "";
		String userID = key.toString();
		List<String> ngrams = new NGrams(value.toString(), gramSize).generateNGramsWithoutLabel();
		try {
			ClassifierResult result;
			result = classifierContext.classifyDocument(ngrams.toArray(new String[ngrams.size()]), defaultCategory);
			docLabelString = result.getLabel();
		} catch (InvalidDatastoreException e) {
			log.error(e.toString(), e);
			context.getCounter(Counter.FAILDOCS).increment(1);
		}
		
		//key is userID and docLabel
		outKey.set(userID + "|" + docLabelString);
		context.write(outKey, ONE);
	}

	@Override
	protected void setup(Context context)
			throws IOException, InterruptedException {
		// get bayes parameters
		Configuration conf = context.getConfiguration();
		BayesParameters params = new BayesParameters(conf.get("bayes.parameters", ""));
		log.info("Bayes Parameter {}", params.print());
		
		Algorithm algorithm;
		Datastore datastore;
		if ("bayes".equalsIgnoreCase(params.get("classifierType"))) {
			algorithm = new BayesAlgorithm();
			datastore = new InMemoryBayesDatastore(params);
		} else if ("cbyes".equalsIgnoreCase(params.get("classifierType"))) {
			algorithm = new CBayesAlgorithm();
			datastore = new InMemoryBayesDatastore(params);
		} else {
			throw new IllegalArgumentException("Unrecognized classifier type: " + params.get("classifierType"));
		}
		
		classifierContext = new ClassifierContext(algorithm, datastore);
		try {
			classifierContext.initialize();
		} catch (InvalidDatastoreException e) {
			log.error(e.toString(), e);
		}
		
		defaultCategory = params.get("defaultCat");
		gramSize = params.getGramSize();
	}
	

}
