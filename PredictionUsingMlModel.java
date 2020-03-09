package PredictionUsingMlModel;
import java.io.File;
import weka.classifiers.trees.J48;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.misc.SerializedClassifier;
import weka.core.Debug;
import weka.core.Instances;

 public class PredictionUsingMlModel{
    public static void main(String[] args) throws Exception {
        
 DataSource source = new DataSource("C:\\Program Files\\Weka-3-8-4\\data\\diabetes.arff");
 Instances data = source.getDataSet();
 
 if (data.classIndex() == -1)
   data.setClassIndex(data.numAttributes() - 1);
 
 J48 tree = new J48();           
 tree.buildClassifier(data);    
  
         Debug.saveToFile("D:\\Ml_model\\tree_Ml_model.modle", tree);  
                  
         SerializedClassifier loadmod = new SerializedClassifier();
         loadmod.setModelFile(new File("D:\\Ml_model\\tree_Ml_model.modle"));
                
         DataSource source1 = new DataSource("D:\\Ml_model\\diabetes.arff");
          Instances data1 = source1.getDataSet();    
                       
            if (data1.classIndex() == -1)
                 data1.setClassIndex(data1.numAttributes() - 1);           
           
double predicted=loadmod.classifyInstance(data1.instance(1));

        System.out.println("Predicted label"); 
        System.out.println(predicted); 
    }}
