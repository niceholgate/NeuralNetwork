using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

using NicUtils;


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

        public int[] LayerNodeCounts { get; }
        public InputLayer InLayer { get; private set; }
        public OutputLayer OutLayer { get; private set; }
        public List<Layer> AllLayers { get; private set; }
        public Network(int[] layerNodeCounts, int DataSetColumnCount, int DataLabelsUniqueCount) {
            this.LayerNodeCounts = layerNodeCounts;
            this.AllLayers = new List<Layer>();
            CheckDataSize(DataSetColumnCount, DataLabelsUniqueCount);
            Initialize();
        }
        private void CheckDataSize(int DataSetColumnCount, int DataLabelsUniqueCount) {
            if (LayerNodeCounts[0] != DataSetColumnCount) {
                throw new Exception("The input layer must have the same number of (non-bias) nodes as the number of columns in the DataSet.");
            }
            for (int i = 1; i < LayerNodeCounts.Length - 1; i++) {
                if (LayerNodeCounts[i] < LayerNodeCounts[^1]) {
                    Console.WriteLine($"Warning: Layer {i} has only {LayerNodeCounts[i]} nodes while the output layer has {LayerNodeCounts[^1]} nodes.");
                }
            }
            if (LayerNodeCounts[^1] != DataLabelsUniqueCount) {
                throw new Exception("The output layer must have the same number of (non-bias) nodes as the number of unique Data Labels.");
            }
        }
        private void Initialize() {
            // Create all the layers
            InLayer = new InputLayer(LayerNodeCounts[0]);
            AllLayers.Add(InLayer);
            for (int i = 1; i < LayerNodeCounts.Length - 1; i++) {
                AllLayers.Add(new HiddenLayer(LayerNodeCounts[i]));
            }
            OutLayer = new OutputLayer(LayerNodeCounts[^1]);
            AllLayers.Add(OutLayer);

            // Initialize all the layers (includes setting neighbours, so not part of layer creation)
            for (int i = 0; i < LayerNodeCounts.Length; i++) {
                Layer prevLayer = (i == 0) ? null : AllLayers[i - 1];
                Layer nextLayer = (i == LayerNodeCounts.Length - 1) ? null : AllLayers[i + 1];
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
    }

    
}
