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

        public static List<int> PrimesUpTo(int n) {
            if (n <= 2) { return new List<int>(); }
            else if (n == 3) { return new List<int> { 2 }; }
            else if (n == 4) { return new List<int> { 2, 3 }; }
            List<int> primes = new List<int>{ 2, 3 };
            for (int i = 5; i < n; i += 2) {
                bool isPrime = true;
                for (int j = 3; j < i; j += 2) {
                    if ( i % j == 0 ) {
                        isPrime = false;
                        break;
                    }
                }
                if (isPrime) {
                    primes.Add(i);
                }
            }
            return primes;
        }
    }
    class Program {

        static void Main(string[] args) {

            List<int> primes = NetworkManager<int>.PrimesUpTo(30);
            foreach (int i in primes) {
                Console.WriteLine(i);
            }


            int j = 0;
            if ((j++ == 0) || (j++ == 2))
                Console.Out.WriteLine(j);

            float a = (float)1;
            float b = (float)1e-10;
            float sum = a + b;
            Console.Out.WriteLine(b);
            Console.Out.WriteLine(sum);

            int x = 0;
            while (x != 3)
                x += 2;
            Console.Out.WriteLine(x);

            IEnumerable<int> layerNodeCounts = new int[] { 400, 200, 10 };
            int trainingExamplesPercent = 80;
            double regularizationConstant = 0.1;
            double learningRate = 0.2;
            double weightGradFracChangeThreshold = 0.0000001;
            double costFuncFracChangeThreshold = 0.00000001;
            bool useCostFuncConvergence = true;
            int maxIterations = 30;
            int[] iterationsToCheckGradient = new int[] { };
            NetworkParameters parameters = new NetworkParameters(layerNodeCounts,
                                                                trainingExamplesPercent,
                                                                regularizationConstant,
                                                                learningRate,
                                                                weightGradFracChangeThreshold,
                                                                costFuncFracChangeThreshold,
                                                                useCostFuncConvergence,
                                                                maxIterations,
                                                                iterationsToCheckGradient);
            
            NetworkManager<int> manager = new NetworkManager<int>("E:/dev/misc/handwriting_labelled/X.csv", "E:/dev/misc/handwriting_labelled/y.csv", parameters);
            List<double> costFunctionHistory = manager.Network.Train();
            double acc = manager.Network.GetAccuracy();


            int aaa = 2;

        }
    }
}
