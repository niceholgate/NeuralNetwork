using MathNet.Numerics.LinearAlgebra;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;

using NicUtils;

namespace NeuralNetwork {
    class NetworkManager<LabelType> where LabelType : struct, System.IEquatable<LabelType>, System.IFormattable {
        public NetworkParameters Parameters { get; private set; }
        public LabelledData<LabelType> Data { get; private set; }
        public Network<LabelType> Network { get; private set; }
        public NetworkManager(string dataSetFilepath, string dataLabelsFilepath, NetworkParameters parameters) {
            InitializeParametersDataNetwork(parameters, dataSetFilepath, dataLabelsFilepath);
        }
        public void InitializeParametersDataNetwork(NetworkParameters parameters, string dataSetFilepath, string dataLabelsFilepath) {
            this.Parameters = parameters;
            this.Data = new LabelledData<LabelType>(dataSetFilepath, dataLabelsFilepath, Parameters.TrainingExamplesPercent);
            this.Network = new Network<LabelType>(Parameters.LayerNodeCounts.ToArray(), this);
        }
    }
    class Program {

        static void Main(string[] args) {
            IEnumerable<int> layerNodeCounts = new int[] { 400, 100, 80, 10 };
            int trainingExamplesPercent = 80;
            double regularizationConstant = 1.0;
            double learningRate = 0.2;
            NetworkParameters parameters = new NetworkParameters(layerNodeCounts,
                                                                trainingExamplesPercent,
                                                                regularizationConstant,
                                                                learningRate);
            NetworkManager<int> manager = new NetworkManager<int>("E:/dev/misc/handwriting_labelled/X.csv", "E:/dev/misc/handwriting_labelled/y.csv", parameters);
            manager.Network.Train();


            int aaa = 2;

        }
    }
}
