using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

using NicUtils;

namespace NeuralNetwork {
    class LabelledData<LabelType> {
        public Matrix<double> DataSet { get; private set; }
        public List<LabelType> DataLabels { get; private set; }
        public List<LabelType> DataLabelsUnique { get; private set; }
        public Matrix<double> TestingDataSet { get; private set; }
        public Matrix<double> TestingTargetOutputs { get; private set; }
        public List<LabelType> TestingLabels { get; private set; }
        public Matrix<double> TrainingDataSet { get; private set; }
        public Matrix<double> TrainingTargetOutputs { get; private set; }
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

            // Calculate number of training examples to randomly select
            if (trainingExamplesPercent < 1 || trainingExamplesPercent > 100) {
                throw new Exception("trainingExamplesPercent must be an integer between 1 and 100 inclusive.");
            }
            int nTrainingExamples = Math.Max(1, (int)Math.Round(trainingExamplesPercent / 100.0 * DataLabels.Count));

            // Randomly move training indices to testing indices until the desired number are left
            List<int> trainingIndices = new List<int>(Enumerable.Range(0, DataLabels.Count));
            List<int> testingIndices = new List<int>();
            Random r = new Random();
            while (trainingIndices.Count > nTrainingExamples) {
                int trainingRemoveIndex = r.Next(0, trainingIndices.Count);
                testingIndices.Add(trainingIndices[trainingRemoveIndex]);
                trainingIndices.RemoveAt(trainingRemoveIndex);
            }
            testingIndices.Sort();

            CreateTrainingDataSet(trainingIndices);
            CreateTrainingTargetOutputs(trainingIndices);
            CreateTestingDataSet(testingIndices);
            CreateTestingTargetOutputs(testingIndices);
        }
        
        private void CreateTrainingDataSet(List<int> TrainingIndices) {
            TrainingDataSet = DataSet.KeepRows(TrainingIndices);
        }
        private void CreateTrainingTargetOutputs(List<int> TrainingIndices) {
            List<Vector<double>> vecs = new List<Vector<double>>();
            for (int i = 0; i < TrainingIndices.Count; i++) {
                vecs.Add(GetExpectedOutput(TrainingIndices[i]));
            }
            TrainingTargetOutputs = Matrix<double>.Build.DenseOfRowVectors(vecs);
        }
        private void CreateTestingDataSet(List<int> TestingIndices) {
            TestingDataSet = DataSet.KeepRows(TestingIndices);
        }
        private void CreateTestingTargetOutputs(List<int> TestingIndices) {
            List<Vector<double>> vecs = new List<Vector<double>>();
            TestingLabels = new List<LabelType>();
            for (int i = 0; i < TestingIndices.Count; i++) {
                vecs.Add(GetExpectedOutput(TestingIndices[i]));
                TestingLabels.Add(DataLabels.ElementAt(TestingIndices[i]));
            }
            TestingTargetOutputs = Matrix<double>.Build.DenseOfRowVectors(vecs);
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
