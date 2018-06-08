using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading.Tasks;

using TensorFlow;

namespace TensorFlowTest
{
    class Program
    {
        static void Main(string[] args)
        {

            FileStream fs = new FileStream("E:\\SVN\\DeepLearning\\MNIST\\train-images.idx3-ubyte", FileMode.Open);
            fs.Position = 16;
            byte[] content = new byte[fs.Length - 16];
            fs.Read(content, 0, content.Length);
            fs.Close();

            var loaded = TFTensor.FromBuffer(new TFShape(60000, 28, 28, 1), content, 0, content.Length);
            using (var session = new TFSession())
            {
                var graph = session.Graph;
                TFOutput input;
                TFOutput output;
                var (handle, flow) = graph.TensorArrayV3(graph.Const(1), TFDataType.Float, new TFShape(28, 28, 1), true);
                input = graph.Placeholder(TFDataType.UInt8, new TFShape(60000, 28, 28, 1));
                output = graph.TensorArraySplitV3(handle, graph.Div(graph.Cast(input, TFDataType.Float), graph.Const(255f)), graph.Const(new long[] { 60000L }), flow);                //output = handle;
                //output = graph.Cast(input, TFDataType.Float);
                var result = session.Run(new TFOutput[] { input }, new TFTensor[] { loaded }, new TFOutput[] { output });

                output = graph.TensorArrayReadV3(handle, graph.Const(0), flow, TFDataType.Float);
                result = session.Run(new TFOutput[] { }, new TFTensor[] { }, new TFOutput[] { output });
            }

        }
    }
}
