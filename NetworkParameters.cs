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
        public double WeightGradFracChangeThreshold { get; private set; }
        public double CostFuncFracChangeThreshold { get; private set; }
        public bool UseCostFuncConvergence { get; private set; }
        public int MaxIterations { get; private set; }
        public int[] IterationsToCheckGradient { get; private set; }
    public NetworkParameters(IEnumerable<int> layerNodeCounts, int trainingExamplesPercent, double regularizationConstant, double learningRate, double weightGradFracChangeThreshold, double costFuncFracChangeThreshold, bool useCostFuncConvergence, int maxIterations, int[] iterationsToCheckGradient) {
            this.LayerNodeCounts = layerNodeCounts.ToArray();
            this.TrainingExamplesPercent = trainingExamplesPercent;
            this.RegularizationConstant = regularizationConstant;
            this.LearningRate = learningRate;
            this.WeightGradFracChangeThreshold = weightGradFracChangeThreshold;
            this.CostFuncFracChangeThreshold = costFuncFracChangeThreshold;
            this.UseCostFuncConvergence = useCostFuncConvergence;
            this.MaxIterations = maxIterations;
            this.IterationsToCheckGradient = iterationsToCheckGradient;
        }
    }
}
