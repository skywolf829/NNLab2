import Jama.*;
import Jama.examples.*;
import Jama.test.*;
import Jama.util.*;

ArrayList<Float> randomPoints = new ArrayList<Float>();
ArrayList<Float> randomResults = new ArrayList<Float>();
ArrayList<GaussianNode> nodes = new ArrayList<GaussianNode>();

int sampleSize = 75;
int numClusters = 15;
int numEpochs = 100;

float learningRate = 0.02;

int graphX = 30;
int graphY = 30;
int graphHeight = 500;
int graphWidth = 500;
float graphXmin = 0;
float graphXmax = 1;
float graphYmin = 0;
float graphYmax = 1;

void setup() {
  size(700, 700);
  randomSeed(8675309);
  for (int i = 0; i < sampleSize; i++) {
    randomPoints.add(random(0, 1));
    randomResults.add(project2Output(randomPoints.get(i)));
  }
  nodes = kMeans(randomPoints, numClusters);

  thread("train");
}



void draw() {
  background(255);
  stroke(0);
  fill(0);
  line(graphX, graphY, graphX, graphHeight*2 + graphY);
  line(0, graphY + graphHeight, graphX + graphWidth, graphHeight + graphY);

  text("1", graphX + 2, graphY);
  text("0", graphX + 2, graphY + graphHeight - 2);
  text("1", graphX + graphWidth + 2, graphY + graphHeight);
  text("RBF estimation for 0.5 + 0.4sin(2*PI*x)", graphX + (graphWidth / 3), graphY);

  fill(0, 0, 255);
  stroke(0);
  for (int i = 0; i < sampleSize; i++) {
    ellipse(graphToCanvas(new float[]{randomPoints.get(i), 
      randomResults.get(i)})[0], 
      graphToCanvas(new float[]{randomPoints.get(i), 
        randomResults.get(i)})[1], 
      5, 5);
  }
  for (float j = graphXmin; j < graphXmax; j += (graphXmax - graphXmin) / 100.0) {
    for (int i = 0; i < numClusters; i++) {
      float y = nodes.get(i).Activate(j);
      float[] center = graphToCanvas(
        new float[]{j, y}); 
      float[] centerNext = graphToCanvas(
        new float[]{j + (graphXmax - graphXmin) / 100.0, 
        nodes.get(i).Activate(j + (graphXmax - graphXmin) / 100.0)}); 
      stroke((i *191) % 255, (i*11) % 255, (i * 367) %255, 30);
      line(center[0], center[1], centerNext[0], centerNext[1]);
    }
    stroke(0);
    fill(255, 0, 0);
    float[] predicted = graphToCanvas(new float[]{j, predicted(j)});
    float[] predictedNext = graphToCanvas(new float[]{j + (graphXmax - graphXmin) / 100.0, 
      predicted(j + (graphXmax - graphXmin) / 100.0)});
    stroke(255, 0, 0);
    line(predicted[0], predicted[1], predictedNext[0], predictedNext[1]);
    float[] actual = graphToCanvas(new float[]{j, 0.5 + 0.4 * sin(2 * PI * j)});
    float[] actualNext = graphToCanvas(new float[]{j + (graphXmax - graphXmin) / 100.0, 
      0.5 + 0.4 * sin(2 * PI * (j + (graphXmax - graphXmin) / 100.0))});
    stroke(255, 200, 0);
    line(actual[0], actual[1], actualNext[0], actualNext[1]);
  }
}

void train() {

  int epoch = 0;
  while (epoch < numEpochs) {    
    for (int i = 0; i < sampleSize; i++) {
      float predicted = predicted(randomPoints.get(i));
      float error = randomResults.get(i) - predicted;
      for (int j = 0; j < numClusters; j++) {
        nodes.get(j).weight +=
          learningRate * error * (nodes.get(j).Gaussian(randomPoints.get(i)));    
        nodes.get(j).bias +=
          learningRate * error;
      }
    }
    epoch++;
    //delay(20);
  }
  for (int i = 0; i < numClusters; i++) {
    println(i + " " + nodes.get(i).weight + " " + nodes.get(i).bias);
  }
}
float predicted(float input) {
  float result = 0;
  for (int i = 0; i < numClusters; i++) {
    result += nodes.get(i).Activate(input);
  }
  return result;
}

float project2Output(float input) {
  return 0.5 + 0.4 * sin(2 * PI * input) + random(-0.1, 0.1);
}

ArrayList<GaussianNode> kMeans(ArrayList<Float> samples, int k) {
  ArrayList<Float> clusterCenters = new ArrayList<Float>(); 
  ArrayList<ArrayList<Float>> clusters = new ArrayList<ArrayList<Float>>(); 
  for (int i = 0; i < k; i++) {    
    clusters.add(new ArrayList<Float>());
    int r;
    boolean repeat = false;
    do {
      repeat = false;
      r = (int)random(0, samples.size());
      for (int j = 0; j < clusterCenters.size(); j++) {
        if (clusterCenters.get(j) == samples.get(r)) {
          repeat = true;
        }
      }
    } while (repeat);

    clusterCenters.add(samples.get(r));
  }

  boolean repeat = true; 

  while (repeat) {
    repeat = false;
    for (int i =0; i < k; i++) {
      clusters.set(i, new ArrayList<Float>());
    }
    for (int i = 0; i < samples.size(); i++) {
      int closest = 0; 
      for (int j = 0; j < k; j++) {
        if (sqrt(pow(samples.get(i) - clusterCenters.get(j), 2)) <
          sqrt(pow(samples.get(i) - clusterCenters.get(closest), 2))) {
          closest = j;
        }
      }
      clusters.get(closest).add(samples.get(i));
    }

    ArrayList<Float> oldClusterCenters = new ArrayList<Float>(); 
    for (int i = 0; i < k; i++) {
      oldClusterCenters.add(clusterCenters.get(i)); 
      float avg = 0; 
      for (int j = 0; j < clusters.get(i).size(); j++) {
        avg += clusters.get(i).get(j);
      }
      avg /= clusters.get(i).size(); 
      clusterCenters.set(i, avg);
    }

    for (int i = 0; i < k; i++) {
      if (abs(oldClusterCenters.get(i) - clusterCenters.get(i)) > 0.001) {
        repeat = true;
      }
    }
  }

  /*
  for (int i =0; i < k; i++) {
   //println(clusters.get(i).size());
   }
   */
  ArrayList<GaussianNode> toReturn = new ArrayList<GaussianNode>(); 
  for (int i = 0; i < k; i++) {
    float variance = 0; 
    if (clusters.get(i).size() == 1) {
      variance = 0;
    } else {
      for (int j = 0; j < clusters.get(i).size(); j++) {
        variance += pow(clusters.get(i).get(j) - clusterCenters.get(i), 2);
      }
      variance /= clusters.get(i).size();
    }
    GaussianNode g = new GaussianNode(clusterCenters.get(i), variance*10); 
    g.weight = random(-1, 1);
    g.bias = random(0, 1);
    //g.bias = 0.5;
    toReturn.add(g);
  }
  for (int i = 0; i < k; i++) {
    if (toReturn.get(i).variance == 0) {
      float avg = 0; 
      for (int j = 0; j < k; j++) {
        if (j != i) {
          avg += toReturn.get(j).variance;
        }
      }
      avg /= (toReturn.size() - 1); 
      toReturn.get(i).variance = avg;
    }
  }
  return toReturn;
}

float[] graphToCanvas(float[] input) {
  return new float[] {graphX + 
    (input[0] *(graphXmax - graphXmin) * (graphWidth)), 
    graphY + graphHeight -
    (input[1] * (graphYmax - graphYmin) * graphHeight) };
}