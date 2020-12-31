using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork {
    struct NetworkParameters {
        public int[] LayerNodeCounts { get; private set; }
        public int TrainingExamplesPercent { get; private set; }
        public double RegularizationConstant { get; private set; }
        public double LearningRate { get; private set; }
        public NetworkParameters(IEnumerable<int> layerNodeCounts, int trainingExamplesPercent, double regularizationConstant, double learningRate) {
            this.LayerNodeCounts = layerNodeCounts.ToArray();
            this.TrainingExamplesPercent = trainingExamplesPercent;
            this.RegularizationConstant = regularizationConstant;
            this.LearningRate = learningRate;
        }
    }
}
