using System;
using System.Collections.Generic;
using System.Linq;
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
        private Matrix<double> FwdPropOutputs;
        private bool FwdPropAllFlag;
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

            FwdPropAll();
        }

        public void FwdPropOne(Vector<double> inputValues) {
            InLayer.UpdateInput(inputValues);
            for (int i = 0; i < AllLayers.Count; i++) { AllLayers[i].FwdPropUpdate(); }
        }

        public void FwdPropAll() {
            Matrix<double> Activations = FwdPropAllOutputs(Manager.Data.TrainingDataSet);
            FwdPropOutputs = Activations.ExcludeFirstColumn();
            FwdPropAllFlag = true;
        }
        public List<LabelType> PredictAll() {
            Matrix<double> Activations = FwdPropAllOutputs(Manager.Data.TestingDataSet);
            List<LabelType> predictions = new List<LabelType>();
            for (int i = 0; i < Activations.RowCount; i++) {
                int indexOfMax = Activations.Row(i).SubVector(1, Activations.ColumnCount - 1).IndexOfMax();
                predictions.Add(Manager.Data.DataLabelsUnique.ElementAt(indexOfMax));
            }
            return predictions;
        }
        public double GetAccuracy() {
            List<LabelType> predictions = PredictAll();
            if (predictions.Count != Manager.Data.TestingLabels.Count) {
                throw new Exception("The label predictions and known labels are not the same lengths.");
            }
            IComparer<LabelType> comparer = Comparer<LabelType>.Default;
            int matches = 0;
            for (int i = 0; i < predictions.Count; i++) {
                if (comparer.Compare(predictions[i], Manager.Data.TestingLabels[i]) == 0) {
                    matches++;
                }
            }
            return (double)matches / predictions.Count;
        }

        public Matrix<double> FwdPropAllOutputs(Matrix<double> dataSet) {
            Matrix<double> Activations = M.Dense(dataSet.RowCount, dataSet.ColumnCount + 1, 1.0);
            Activations.SetSubMatrix(0, 1, dataSet);
            for (int i = 0; i < AllLayers.Count - 1; i++) {
                Matrix<double> RawValues = AllLayers[i].Weights * Activations.Transpose();
                RawValues = RawValues.Transpose();
                Activations = M.Dense(RawValues.RowCount, RawValues.ColumnCount + 1, 1.0);
                Activations.SetSubMatrix(0, 1, RawValues.Map(Sigmoid.Value));
            }
            return Activations;
        }

    public void BackPropOne(Vector<double> targetOutput, Vector<double> overrideOutput) {
            OutLayer.UpdateTargetOutput(targetOutput);
            OutLayer.SetActivations(overrideOutput);
            for (int i = AllLayers.Count - 1; i >= 0; i--) { AllLayers[i].BackPropUpdate(); }
        }

        public void BackPropAll() {
            if (!FwdPropAllFlag) { throw new Exception("Must run FwdPropAll() before BackPropAll() in order to populate FwdPropOutputs matrix."); }
            Matrix<double> TrainingTargetOutputs = Manager.Data.TrainingTargetOutputs;
            ResetErrorAccumulators();
            for (int i = 0; i < TrainingTargetOutputs.RowCount; i++) {
                BackPropOne(TrainingTargetOutputs.Row(i), FwdPropOutputs.Row(i));
            }
        }

        private void ResetErrorAccumulators() {
            foreach (Layer layer in AllLayers) { layer.ResetAccDeltas(); }
        }
        // gradients being calculated and applied for bias column causing errors?
        public List<double> Train() {
            int iterations = 0;
            bool converged = false;
            List<double> costFunctionHistory = new List<double>();

            // Get the value of the cost function for the initial weights
            costFunctionHistory.Add(GetCost());
            Console.WriteLine($"Initial costFunc = {costFunctionHistory.Last()}");

            while (!converged) {
                if (iterations == Manager.Parameters.MaxIterations) {
                    Console.WriteLine($"Did not converge by MaxIterations ({iterations}).");
                    return costFunctionHistory;
                }

                // Get the BackProp weight gradients
                List<Matrix<double>> weightGrads = FindWeightGrads();
                
                // Perform numerical gradient checking on specified iterations
                if (Manager.Parameters.IterationsToCheckGradient.Contains(iterations)) { CheckWeightGrads(weightGrads); }
                
                // Update the weights
                double minWeightFracChange = UpdateWeights(weightGrads);

                // Get the value of the cost function for the new weights
                costFunctionHistory.Add(GetCost());

                // Increment iterations
                iterations++;
                Console.WriteLine($"Finished iteration {iterations} with costFunc = {costFunctionHistory.Last()}, minWeightFracChange = {minWeightFracChange}");

                // Check convergence
                converged = CheckConvergence(costFunctionHistory, minWeightFracChange, iterations);
            }
            Console.WriteLine($"Converged on iteration {iterations}.");
            return costFunctionHistory;
        }
        
        private bool CheckConvergence(List<double> costFunctionHistory, double minWeightFracChange, int iterations) {
            if (Manager.Parameters.UseCostFuncConvergence) {
                double ultimateCost = costFunctionHistory.ElementAt(costFunctionHistory.Count - 1);
                double penultimateCost = costFunctionHistory.ElementAt(costFunctionHistory.Count - 2);
                return Math.Abs((ultimateCost - penultimateCost) / ultimateCost) < Manager.Parameters.CostFuncFracChangeThreshold;
            } else {
                return minWeightFracChange < Manager.Parameters.WeightGradFracChangeThreshold;
            }
        }
        private void CheckWeightGrads(List<Matrix<double>> weightGrads) {
            double e = 10E-4;
            for (int l = 0; l < AllLayers.Count - 1; l++) {
                Matrix<double> layerNumericGrads = M.Dense(AllLayers[l].Weights.RowCount, AllLayers[l].Weights.ColumnCount);
                for (int i = 0; i < AllLayers[l].Weights.RowCount; i++) {
                    for (int j = 0; j < AllLayers[l].Weights.ColumnCount; j++) {
                        AllLayers[l].Weights[i, j] += e;
                        double c_plus = GetCost();
                        AllLayers[l].Weights[i, j] -= 2 * e;
                        double c_minus = GetCost();
                        AllLayers[l].Weights[i, j] += e;
                        layerNumericGrads[i, j] = (c_plus - c_minus) / 2 / e;
                    }
                }
                Matrix<double> fracDiffs = (weightGrads[l] - layerNumericGrads).PointwiseDivide(layerNumericGrads);
                if (fracDiffs.ExcludeFirstColumn().Max() > 0.01) { throw new Exception("BackProp gradient does not match numerical gradient."); }
            }
        }
        private List<Matrix<double>> FindWeightGrads() {
            // Get the TrainingData and TargetOutputs from the NetworkManager and verify that they each have the same number of examples
            Matrix<double> TrainingDataSet = Manager.Data.TrainingDataSet;
            Matrix<double> TrainingTargetOutputs = Manager.Data.TrainingTargetOutputs;
            if (TrainingDataSet.RowCount != TrainingTargetOutputs.RowCount) {
                throw new Exception("TrainingDataSet and TrainingTargetOutputs do not have the same number of rows (examples).");
            }

            // Get all FwdProp outputs for current network weights, then BackPropOne for all of the training examples
            if (!FwdPropAllFlag) { FwdPropAll(); }
            BackPropAll();

            // Calculate the gradients of the weights (all but last layer)
            return (from i in Enumerable.Range(0, LayerCount - 1) select CalcRegularizedWeightGrad(AllLayers[i], TrainingDataSet.RowCount)).ToList();
        }
        private double UpdateWeights(List<Matrix<double>> WeightGrads) {
            List<double> layerMinWeightFracChanges = new List<double>();
            for (int i = 0; i < LayerCount - 1; i++) {
                layerMinWeightFracChanges.Add((Manager.Parameters.LearningRate * WeightGrads[i]).PointwiseDivide(AllLayers[i].Weights).Min());
                Matrix<double> layerWeights = AllLayers[i].Weights;
                Matrix<double> layerWeightsSub = layerWeights.SubMatrix(0, layerWeights.RowCount, 1, layerWeights.ColumnCount - 1);
                Matrix<double> weightGradsSub = WeightGrads[i].SubMatrix(0, WeightGrads[i].RowCount, 1, WeightGrads[i].ColumnCount - 1);
                AllLayers[i].Weights.SetSubMatrix(0, 1, layerWeightsSub - Manager.Parameters.LearningRate * weightGradsSub);
            }
            FwdPropAllFlag = false;
            return layerMinWeightFracChanges.Min();
        }
        private Matrix<double> CalcRegularizedWeightGrad(Layer layer, int exampleCount) {
            Matrix<double> unregGrad = layer.AccDeltas / exampleCount;
            Matrix<double> weights = layer.Weights;
            weights.SetSubMatrix(0, 0, M.Dense(layer.Weights.RowCount, 1));
            Matrix<double> regularization = weights * Manager.Parameters.RegularizationConstant / exampleCount;
            return unregGrad + regularization;
        }

        public double GetCost() {
            // Calculate unregularized cost
            Matrix<double> TrainingDataSet = Manager.Data.TrainingDataSet;
            Matrix<double> TrainingTargetOutputs = Manager.Data.TrainingTargetOutputs;
            int exampleCount = TrainingDataSet.RowCount;
            Matrix<double> unregCostTerms1 = -TrainingTargetOutputs.PointwiseMultiply(FwdPropOutputs.Map(Math.Log));
            Matrix<double> unregCostTerms2 = (TrainingTargetOutputs-1).PointwiseMultiply((1- FwdPropOutputs).Map(Math.Log));
            double unregCost = (unregCostTerms1 + unregCostTerms2).ColumnSums().Sum() / exampleCount;
            
            // Calculate regularization term
            double regSum = 0;
            for (int i = 0; i < LayerCount - 1; i++) {
                regSum += AllLayers[i].Weights.Map(x => x * x).RowSums().Sum();
            }
            double regularization = regSum * Manager.Parameters.RegularizationConstant / 2 / exampleCount;
            
            return unregCost + regularization;
        }
    }

    
}
