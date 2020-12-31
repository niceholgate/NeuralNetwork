using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

using NicUtils;

namespace NeuralNetwork {
    struct LabelledData<LabelType> {
        public Matrix<double> DataSet { get; private set; }
        public List<LabelType> DataLabels { get; private set; }
        public List<LabelType> DataLabelsUnique { get; private set; }
        public List<int> TrainingIndices { get; private set; }
        public List<int> TestingIndices { get; private set; }
        public LabelledData(string dataSetFilepath, string dataLabelsFilepath, int trainingExamplesPercent) {
            // Load DataSet of inputs
            DataSet = Matrix<double>.Build.DenseOfColumns(CSVReader<double>.ReadCSV(dataSetFilepath));

            // Load corresponding DataLabels and find the unique labels
            DataLabels = Enumerables<LabelType>.Unroll2DEnumerable(CSVReader<LabelType>.ReadCSV(dataLabelsFilepath));
            DataLabelsUnique = DataLabels.Distinct().OrderBy(x => x).ToList();
            
            // Check that number of DataLabels (rows) matches number of DataSet examples - transpose the DataSet if necessary
            if (DataSet.RowCount != DataLabels.Count) {
                if (DataSet.ColumnCount == DataLabels.Count) { DataSet = DataSet.Transpose(); }
                else { throw new Exception($"One DataSet dimension ({DataSet.RowCount} x {DataSet.ColumnCount}) must match the number of DataLabels ({DataLabels.Count})."); }
            }

            if (trainingExamplesPercent < 1 || trainingExamplesPercent > 100) {
                throw new Exception("trainingExamplesPercent must be an integer between 1 and 100 inclusive.");
            }
            // Number of training examples to randomly select
            int nTrainingExamples = Math.Max(1, (int)Math.Round(trainingExamplesPercent / 100.0 * DataLabels.Count));

            // Randomly move training indices to testing indices until the desired number are left
            TrainingIndices = new List<int>(Enumerable.Range(0, DataLabels.Count));
            TestingIndices = new List<int>();
            Random r = new Random();
            while (TrainingIndices.Count > nTrainingExamples) {
                int trainingRemoveIndex = r.Next(0, TrainingIndices.Count);
                TestingIndices.Add(TrainingIndices[trainingRemoveIndex]);
                TrainingIndices.RemoveAt(trainingRemoveIndex);
            }
            TestingIndices.Sort();
        }
        public Matrix<double> GetTrainingDataSet() {
            return DataSet.KeepRows(TrainingIndices);
        }
        public Matrix<double> GetTrainingTargetOutputs() {
            List<Vector<double>> vecs = new List<Vector<double>>();
            for (int i = 0; i < TrainingIndices.Count; i++) {
                vecs.Add(GetExpectedOutput(TrainingIndices[i]));
            }
            return Matrix<double>.Build.DenseOfRowVectors(vecs);
        }
        public Matrix<double> GetTestingDataSet() {
            return DataSet.KeepRows(TestingIndices);
        }
        public Matrix<double> GetTestingTargetOutputs() {
            List<Vector<double>> vecs = new List<Vector<double>>();
            for (int i = 0; i < TestingIndices.Count; i++) {
                vecs.Add(GetExpectedOutput(TestingIndices[i]));
            }
            return Matrix<double>.Build.DenseOfRowVectors(vecs);
        }
        private Vector<double> GetExpectedOutput(int dataLabelIndex) {
            LabelType dataLabel = DataLabels[dataLabelIndex];
            Vector<double> expectedOutput = Vector<double>.Build.Dense(DataLabelsUnique.Count);
            for (int i = 0; i < DataLabelsUnique.Count; i++) {
                if (DataLabelsUnique[i].Equals(dataLabel)) {
                    expectedOutput[i] = 1;
                    break;
                }
            }
            return expectedOutput;
        }
    }        
}
