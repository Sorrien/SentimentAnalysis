using System;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;

namespace SentimentAnalysisClassifier
{
    class Program
    {
        const string _dataPath = @"data\amazon_and_imdb.txt";
        const string _testDataPath = @"data\yelp_labelled.txt";

        static void Main(string[] args)
        {
            var model = Train();
            Console.WriteLine("Training Completed");
            
            var input = "";
            while (input != "x")
            {
                Console.WriteLine("Please enter a positive or negative statement.");
                input = Console.ReadLine();
                //List<SentimentData> sentiments = new List<SentimentData>()
                //    {
                //        new SentimentData
                //        {
                //            SentimentText = "Contoso's 11 is a wonderful experience",
                //            Sentiment = 0
                //        },
                //        new SentimentData
                //        {
                //            SentimentText = "Really bad",
                //            Sentiment = 0
                //        },
                //        new SentimentData
                //        {
                //            SentimentText = "Joe versus the Volcano Coffee Company is a great film.",
                //            Sentiment = 0
                //        }
                //    };
                List<SentimentData> sentiments = new List<SentimentData>()
                {
                    new SentimentData
                    {
                        SentimentText = input,
                        Sentiment = 0
                    }
                };
                Predict(model, sentiments);
            }
            Evaluate(model);
            Console.ReadLine();
        }

        public static PredictionModel<SentimentData, SentimentPrediction> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<SentimentData>(_dataPath, useHeader: false, separator: "tab"));
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 20, NumTrees = 20, MinDocumentsInLeafs = 10 });
            PredictionModel<SentimentData, SentimentPrediction> model = pipeline.Train<SentimentData, SentimentPrediction>();
            return model;
        }
        public static List<bool> Predict(PredictionModel<SentimentData, SentimentPrediction> model, List<SentimentData> sentiments)
        {

            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);
            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");
            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => new { sentiment, prediction });
            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Positive" : "Negative")}");
            }
            Console.WriteLine();
            return predictions.Select(x => x.Sentiment).ToList();
        }

        public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var testData = new TextLoader<SentimentData>(_testDataPath, useHeader: false, separator: "tab");
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }
    }
}
