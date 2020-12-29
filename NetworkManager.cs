using MathNet.Numerics.LinearAlgebra;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using NicUtils;

namespace NeuralNetwork {
    class NetworkManager<LabelType> where LabelType : struct, System.IEquatable<LabelType>, System.IFormattable {
        private Matrix<double> DataSet { get; set;  }
        public Matrix<double> GetDataSet() { return DataSet; }
        private LabelType[] DataLabels { get; set; }
        private LabelType[] DataLabelsUnique { get; set; }
        public Network Net;
        public NetworkManager(Matrix<double> dataSet, LabelType[] dataLabels, IEnumerable<int> layerNodeCounts) {
            UpdateDataSetAndNetwork(dataSet, dataLabels, layerNodeCounts);
        }
        public void UpdateDataSetAndNetwork(Matrix<double> dataSet, LabelType[] dataLabels, IEnumerable<int> layerNodeCounts) {
            if (dataSet.RowCount != dataLabels.Length) {
                throw new Exception("The DataSet and DataLabels must have the same number of rows.");
            }
            DataSet = dataSet;
            DataLabels = dataLabels;
            DataLabelsUnique = dataLabels.Distinct().OrderBy(x => x).ToArray();
            CreateNetwork(layerNodeCounts);
        }
        private void CreateNetwork(IEnumerable<int> layerNodeCounts) {
            this.Net = new Network(layerNodeCounts.ToArray(), DataSet.ColumnCount, DataLabelsUnique.Length);
        }
        public Vector<double> GetExpectedOutput(int dataLabelIndex) {
            LabelType dataLabel = DataLabels[dataLabelIndex];
            Vector<double> expectedOutput = Vector<double>.Build.Dense(DataLabelsUnique.Length);
            for (int i = 0; i < DataLabelsUnique.Length; i++) {
                if (DataLabelsUnique[i].Equals(DataLabels[dataLabelIndex])) {
                    expectedOutput[i] = 1;
                    break;
                }
            }
            return expectedOutput;
        }

    }
    class Program {

        static void Main(string[] args) {

            //Network net = new Network(new int[] { 5, 10, 5 });
            //net.BackPropagate(new Vector<double> targetOutputValues);

            // Loading training data from CSVs
            List<int> yUnrolled = Enumerables<int>.Unroll2DEnumerable(CSVReader.ReadCSVInt("E:/dev/misc/handwriting_labelled/y.csv"));
            int[] y = yUnrolled.ToArray();
            Matrix<double> X = Matrix<double>.Build.DenseOfColumns(CSVReader.ReadCSVDouble("E:/dev/misc/handwriting_labelled/X.csv")).Transpose();

            

            NetworkManager<int> man = new NetworkManager<int>(X, y, new int[] { 400, 30, 10 });
            Vector<double> oneExample = man.GetDataSet().KeepRows(new int[] { 0 }).ToVector();
            man.Net.ForwardPropagate(oneExample);
            Vector<double> oneExampleExpectedOutput = man.GetExpectedOutput(0);
            man.Net.BackPropagate(oneExampleExpectedOutput);

            int aaa = 2;

        }
    }
}
