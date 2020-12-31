using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

using NicUtils;

namespace NeuralNetwork
{
    class Network<LabelType> where LabelType : struct, System.IEquatable<LabelType>, System.IFormattable {
        public static MatrixBuilder<double> M = Matrix<double>.Build;
        public static VectorBuilder<double> V = Vector<double>.Build;
        public NetworkManager<LabelType> Manager { get; private set; }
        public int LayerCount { get; private set; }
        public InputLayer InLayer { get; private set; }
        public OutputLayer OutLayer { get; private set; }
        public List<Layer> AllLayers { get; private set; }
        public Network(int[] layerNodeCounts, NetworkManager<LabelType> manager) {
            this.Manager = manager;
            this.LayerCount = manager.Parameters.LayerNodeCounts.Length;
            this.AllLayers = new List<Layer>();
            CheckDataSize(Manager.Parameters.LayerNodeCounts);
            Initialize(Manager.Parameters.LayerNodeCounts);
        }
        private void CheckDataSize(int[] LayerNodeCounts) {
            if (LayerNodeCounts[0] != Manager.Data.DataSet.ColumnCount) {
                throw new Exception("The input layer must have the same number of (non-bias) nodes as the number of columns in the DataSet.");
            }
            for (int i = 1; i < LayerCount - 1; i++) {
                if (LayerNodeCounts[i] < LayerNodeCounts[^1]) {
                    Console.WriteLine($"Warning: Layer {i} has only {LayerNodeCounts[i]} nodes while the output layer has {LayerNodeCounts[^1]} nodes.");
                }
            }
            if (LayerNodeCounts[^1] != Manager.Data.DataLabelsUnique.Count) {
                throw new Exception("The output layer must have the same number of (non-bias) nodes as the number of unique Data Labels.");
            }
        }
        private void Initialize(int[] LayerNodeCounts) {
            // Create all the layers
            InLayer = new InputLayer(LayerNodeCounts[0]);
            AllLayers.Add(InLayer);
            for (int i = 1; i < LayerCount - 1; i++) {
                AllLayers.Add(new HiddenLayer(LayerNodeCounts[i]));
            }
            OutLayer = new OutputLayer(LayerNodeCounts[^1]);
            AllLayers.Add(OutLayer);

            // Initialize all the layers (includes setting neighbours, so not part of layer creation)
            for (int i = 0; i < LayerCount; i++) {
                Layer prevLayer = (i == 0) ? null : AllLayers[i - 1];
                Layer nextLayer = (i == LayerCount - 1) ? null : AllLayers[i + 1];
                AllLayers[i].Initialize(prevLayer, nextLayer);
            }
        }

        public void ForwardPropagate(Vector<double> inputValues) {
            InLayer.UpdateInput(inputValues);
            for (int i = 0; i < AllLayers.Count; i++) { AllLayers[i].ForwardPropagationUpdate(); }
        }

        public void BackPropagate(Vector<double> targetOutput) {
            OutLayer.UpdateTargetOutput(targetOutput);
            for (int i = AllLayers.Count - 1; i >= 0; i--) { AllLayers[i].BackPropagationUpdate(); }
        }

        public void Train() {
            List<Matrix<double>> WeightGrads = FindWeightGrads();
            //UpdateWeights(WeightGrads);

        }
        private List<Matrix<double>> FindWeightGrads() {
            // Get the TrainingData and TargetOutputs from the NetworkManager and verify that they each have the same number of examples
            Matrix<double> TrainingDataSet = Manager.Data.GetTrainingDataSet();
            Matrix<double> TrainingTargetOutputs = Manager.Data.GetTrainingTargetOutputs();
            if (TrainingDataSet.RowCount != TrainingTargetOutputs.RowCount) {
                throw new Exception("TrainingDataSet and TrainingTargetOutputs do not have the same number of rows (examples).");
            }

            // ForwardProp then BackProp for all of the training examples
            int exampleCount = TrainingDataSet.RowCount;
            for (int i = 0; i < exampleCount; i++) {
                ForwardPropagate(TrainingDataSet.Row(i));
                BackPropagate(TrainingTargetOutputs.Row(i));
            }

            // Calculate the gradients of the weights (all but last layer)
            List<Matrix<double>> WeightGrads = new List<Matrix<double>>();
            for (int i = 0; i < LayerCount - 1; i++) { WeightGrads.Add(CalcRegularizedWeightGrad(AllLayers[i], exampleCount)); }
            return WeightGrads;
        }
        //private void UpdateWeights(List<Matrix<double>> WeightGrads) { }
        private Matrix<double> CalcRegularizedWeightGrad(Layer layer, int exampleCount) {
            Matrix<double> unregGrad = layer.AccDeltas / exampleCount;
            Matrix<double> weights = layer.Weights;
            weights.SetSubMatrix(0, 0, M.Dense(layer.Weights.RowCount, 1));
            Matrix<double> regularization = weights * Manager.Parameters.RegularizationConstant / exampleCount;
            return unregGrad + regularization;
        }

    }

    
}
