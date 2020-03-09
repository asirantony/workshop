import weka.classifiers.trees.J48;
 import weka.core.converters.ConverterUtils.DataSource;
 import weka.classifiers.Evaluation;
 import java.util.Random;
 import weka.core.Instances;
 public class My_First_Ml_Model {
 public static void main(String[] args) throws Exception {
             
 DataSource source = new DataSource("C:\\Program Files\\Weka-3-8\\data\\diabetes.arff");
 Instances data = source.getDataSet();
 
 if (data.classIndex() == -1)
   data.setClassIndex(data.numAttributes() - 1);
 String[] options = new String[1];
 options[0] = "-U";            
 J48 tree = new J48();         
 tree.setOptions(options);     
 tree.buildClassifier(data);    
        
 Evaluation eval = new Evaluation(data);
 eval.crossValidateModel(tree, data, 10, new Random(1));
    System.out.println(eval.toSummaryString("\nResults\n======\n", false));
   
   }
}
