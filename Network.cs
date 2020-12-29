using System;
using NicUtils;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Double;


//how do you share code between VS projects? I assume one way involves compiling dlls, but this would require recompilation when the other project changes, so is there a code-sharing way?
//You sent 1 November
//looks like for code reuse you just 'Add existing item' to your solution
//You sent 1 November
//and for dlls you add a 'reference' then add 'using <namespace>'. think i've got it
namespace NeuralNetwork
{
    class Network {
        public static MatrixBuilder<double> M = Matrix<double>.Build;
        public static VectorBuilder<double> V = Vector<double>.Build;
        private class Layer {
            // NodeCount means number of non-bias nodes
            public enum LayerType { Input, Hidden, Output };
            public int NodeCount { get; }
            public Vector<double> RawValues { get; set; }
            public Vector<double> Activations { get; set; }
            public Vector<double> Gradients { get; set; }
            public LayerType Type;
            public Layer(int nodeCount, LayerType layerType) {
                this.NodeCount = nodeCount;
                this.RawValues = V.Dense(nodeCount);
                // Activations must include bias node, unless output layer
                if (Type == LayerType.Output) {
                    this.Activations = V.Dense(nodeCount);
                } else {
                    this.Activations = V.Dense(nodeCount + 1);
                    this.Activations[0] = 1;
                }
                this.Gradients = V.Dense(nodeCount);
                this.Type = layerType;
            }
            public void ForwardPropagationUpdate(Vector<double> newValues) {
                // If input layer, the newValues are activations
                if (Type == LayerType.Input) {
                    // Leave input layer bias node activation as 1
                    Activations.SetSubVector(1, NodeCount, newValues);
                    // Gradient does not include bias node
                    Gradients = Activations.SubVector(1, NodeCount).Map(Sigmoid.DerivativeFromValue);
                // If output layer, the newValues are raw node input values
                } else if (Type == LayerType.Output) {
                    RawValues = newValues;
                    // No bias node for output layer
                    Activations = newValues.Map(Sigmoid.Value);
                    Gradients = Activations.Map(Sigmoid.DerivativeFromValue);
                // If hidden layer, the newValues are raw node input values
                } else {
                    RawValues = newValues;
                    // Leave hidden layer bias node activation as 1
                    Activations.SetSubVector(1, NodeCount, newValues.Map(Sigmoid.Value));
                    // Gradient does not include bias node
                    Gradients = Activations.SubVector(1, NodeCount).Map(Sigmoid.DerivativeFromValue);
                }
            }
        }
      
        public int[] LayerNodeCounts { get; }
        private Layer[] Layers { get; }
        private Matrix<double>[] Weights { get; set; }
        private Vector<double>[] Deltas { get; set; }
        private Matrix<double>[] AccDeltas { get; set; }
        public Network(int[] layerNodeCounts, int DataSetColumnCount, int DataLabelsUniqueCount) {
            if (layerNodeCounts[0] != DataSetColumnCount) {
                throw new Exception("The input layer must have the same number of (non-bias) nodes as the number of columns in the DataSet.");
            }
            for (int i = 1; i < layerNodeCounts.Length - 1; i++) {
                if (layerNodeCounts[i] < layerNodeCounts[^1]) {
                    Console.WriteLine($"Warning: Layer {i} has only {layerNodeCounts[i]} nodes while the output layer has {layerNodeCounts[^1]} nodes.");
                }
            }
            if (layerNodeCounts[^1] != DataLabelsUniqueCount) {
                throw new Exception("The output layer must have the same number of (non-bias) nodes as the number of unique Data Labels.");
            }

            this.LayerNodeCounts = layerNodeCounts;
            this.Layers = new Layer[layerNodeCounts.Length];
            this.Weights = new Matrix<double>[layerNodeCounts.Length];
            this.Deltas = new Vector<double>[layerNodeCounts.Length];
            this.AccDeltas = new Matrix<double>[layerNodeCounts.Length];
            Initialize();
        }

        public void Initialize() {
            for (int i = 0; i < LayerNodeCounts.Length; i++) {
                // Layers
                Layer.LayerType layerType = Layer.LayerType.Hidden;
                if (i == 0) { layerType = Layer.LayerType.Input; }
                else if (i == LayerNodeCounts.Length - 1) { layerType = Layer.LayerType.Output; }
                Layers[i] = new Layer(LayerNodeCounts[i], layerType);
                
                // Deltas
                if (i > 0) { Deltas[i] = V.Dense(LayerNodeCounts[i]); }
                
                // Weights and AccDeltas (extended for bias nodes)
                if (i < LayerNodeCounts.Length - 1) {
                    double epsilon = Math.Sqrt(6.0 / (LayerNodeCounts[i] + LayerNodeCounts[i + 1]));
                    Weights[i] = M.Random(LayerNodeCounts[i + 1], LayerNodeCounts[i]+1, new ContinuousUniform()) * 2 * epsilon - epsilon;
                    AccDeltas[i] = M.Dense(LayerNodeCounts[i + 1], LayerNodeCounts[i]+1);
                }
            }
        }

        public void ForwardPropagate(Vector<double> inputValues) {
            // Input layer takes inputValues as activations
            Layers[0].ForwardPropagationUpdate(inputValues);
            // Subsequent layers take rawValues and use them to calculate activations and gradients
            for (int layerNumber = 1; layerNumber < LayerNodeCounts.Length; layerNumber++) {
                Vector<double> product = Weights[layerNumber - 1] * Layers[layerNumber - 1].Activations;
                Layers[layerNumber].ForwardPropagationUpdate(product);
            }
        }

        public void BackPropagate(Vector<double> targetOutputValues) {
            int nLayers = LayerNodeCounts.Length;
            Deltas[nLayers - 1] = Layers[nLayers - 1].Activations - targetOutputValues;
            for (int i = nLayers - 2; i > 0; i--) {
                var vec = Weights[i].ExcludeFirstColumn().Transpose() * Deltas[i + 1];
                Deltas[i] = vec.PointwiseMultiply(Layers[i].Gradients);
            }
            for (int i = nLayers - 2; i >= 0; i--) {
                AccDeltas[i] = AccDeltas[i] + Deltas[i + 1].ToColumnMatrix() * Layers[i].Activations.ToRowMatrix();
            }
        }
    }

    
}
