using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

using NicUtils;

namespace NeuralNetwork {
    //public interface ILayer {
    //    public void ForwardPropagationUpdate(ILayer previousLayer);
    //    public void BackPropagationUpdate(ILayer nextLayer);
    //}
    public abstract class Layer {
        public static MatrixBuilder<double> M = Matrix<double>.Build;
        public static VectorBuilder<double> V = Vector<double>.Build;
        // NodeCount refers to number of non-bias nodes
        public int NodeCount { get; }
        public Vector<double> Activations { get; protected set; }
        public Vector<double> RawValues { get; protected set; }
        public Vector<double> Gradients { get; protected set; }
        public Vector<double> Deltas { get; protected set; }
        public Matrix<double> Weights { get; protected set; }
        public Matrix<double> AccDeltas { get; protected set; }
        protected bool initFlag = false;
        protected Layer PrevLayer;
        protected Layer NextLayer;
        public Layer(int nodeCount) {
            this.NodeCount = nodeCount;

            // All layers have a bias node, even the output (can ignore)
            this.Activations = V.Dense(nodeCount + 1);
            this.Activations[0] = 1;

            this.RawValues = V.Dense(nodeCount);
            this.Gradients = V.Dense(nodeCount);
            this.Deltas = V.Dense(nodeCount);
        }
        protected void SetNeighbours(Layer prevLayer, Layer nextLayer) {
            PrevLayer = prevLayer;
            NextLayer = nextLayer;
        }
        protected void InitCheck() { if (!initFlag) { throw new Exception("The layer has not been initialized."); } }
        public abstract void Initialize(Layer prevLayer, Layer nextLayer);
        public abstract void ForwardPropagationUpdate();
        public abstract void BackPropagationUpdate();
    }
    
    public class InputLayer : Layer {
        private bool freshInputFlag = false;
        public Vector<double> Input;
        public InputLayer(int nodeCount) : base (nodeCount) { }
        public override void Initialize(Layer prevLayer, Layer nextLayer) {
            if (prevLayer != null || nextLayer == null) { throw new ArgumentException("prevLayer must be null and nextLayer non-null for InputLayer!"); }
            SetNeighbours(prevLayer, nextLayer);
            double epsilon = Math.Sqrt(6.0 / (NodeCount + nextLayer.NodeCount));
            this.Weights = M.Random(nextLayer.NodeCount, NodeCount + 1, new ContinuousUniform()) * 2 * epsilon - epsilon; ;
            this.AccDeltas = M.Dense(nextLayer.NodeCount, NodeCount + 1);
            initFlag = true;
        }
        public void UpdateInput(Vector<double> inputValues) {
            Input = inputValues;
            freshInputFlag = true;
        }
        public override void ForwardPropagationUpdate() {
            InitCheck();
            if (!freshInputFlag) { throw new Exception("Must call InputLayer.UpdateInput() method before InputLayer.ForwardPropagationUpdate()"); }
            // Leave input layer bias node activation as 1
            Activations.SetSubVector(1, NodeCount, Input);
            // Gradient does not include bias node
            Gradients = Activations.SubVector(1, NodeCount).Map(Sigmoid.DerivativeFromValue);
            freshInputFlag = false;
        }
        public override void BackPropagationUpdate() {
            InitCheck();
            AccDeltas = AccDeltas + NextLayer.Deltas.ToColumnMatrix() * Activations.ToRowMatrix();
        }
    }
    public class HiddenLayer : Layer {
        public HiddenLayer(int nodeCount) : base(nodeCount) { }
        public override void Initialize(Layer prevLayer, Layer nextLayer) {
            if (prevLayer == null || nextLayer == null) { throw new ArgumentException("prevLayer and nextLayer must both be non-null for HiddenLayer!"); }
            SetNeighbours(prevLayer, nextLayer);
            double epsilon = Math.Sqrt(6.0 / (NodeCount + nextLayer.NodeCount));
            this.Weights = M.Random(nextLayer.NodeCount, NodeCount + 1, new ContinuousUniform()) * 2 * epsilon - epsilon; ;
            this.AccDeltas = M.Dense(nextLayer.NodeCount, NodeCount + 1);
            initFlag = true;
        }
        public override void ForwardPropagationUpdate() {
            InitCheck();
            RawValues = PrevLayer.Weights * PrevLayer.Activations;
            Activations.SetSubVector(1, NodeCount, RawValues.Map(Sigmoid.Value));
            Gradients = Activations.SubVector(1, NodeCount).Map(Sigmoid.DerivativeFromValue);
        }
        public override void BackPropagationUpdate() {
            InitCheck();
            var vec = Weights.ExcludeFirstColumn().Transpose() * NextLayer.Deltas;
            Deltas = vec.PointwiseMultiply(Gradients);
            AccDeltas = AccDeltas + NextLayer.Deltas.ToColumnMatrix() * Activations.ToRowMatrix();
        }

    }
    public class OutputLayer : Layer {
        private bool freshTargetOutputFlag = false;
        public Vector<double> TargetOutput;
        public OutputLayer(int nodeCount) : base(nodeCount) { }
        public override void Initialize(Layer prevLayer, Layer nextLayer) {
            if (prevLayer == null || nextLayer != null) { throw new ArgumentException("prevLayer must be non-null and nextLayer null for OutputLayer!"); }
            SetNeighbours(prevLayer, nextLayer);
            initFlag = true;
        }
        public override void ForwardPropagationUpdate() {
            InitCheck();
            RawValues = PrevLayer.Weights * PrevLayer.Activations;
            Activations.SetSubVector(1, NodeCount, RawValues.Map(Sigmoid.Value));
            Gradients = Activations.SubVector(1, NodeCount).Map(Sigmoid.DerivativeFromValue);
        }
        public void UpdateTargetOutput(Vector<double> targetOutput) {
            TargetOutput = targetOutput;
            freshTargetOutputFlag = true;
        }
        public override void BackPropagationUpdate() {
            InitCheck();
            Deltas = Activations.SubVector(1, NodeCount) - TargetOutput;
            freshTargetOutputFlag = false;
        }
    }
}
