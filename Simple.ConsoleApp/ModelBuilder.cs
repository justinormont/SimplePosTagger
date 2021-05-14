using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Simple.Model.DataModels;

namespace Simple.ConsoleApp
{
    public static class ModelBuilder
    {
        private static string TRAIN_DATA_FILEPATH = @"../../../../train.tsv";
        private static string TEST_DATA_FILEPATH = @"../../../../test.tsv";
        private static string MODEL_FILEPATH = @"../../../../Simple.Model/MLModel.zip";

        // Create MLContext to be shared across the model creation workflow objects 
        // Set a random seed for repeatable/deterministic results across multiple trainings.
        private static MLContext mlContext = new MLContext(seed: 1);

        public static void CreateModel()
        {
            // Load Data
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: TRAIN_DATA_FILEPATH,
                                            hasHeader: true,
                                            separatorChar: '\t',
                                            allowQuoting: true,
                                            allowSparse: false);

            IDataView testDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: TEST_DATA_FILEPATH,
                                            hasHeader: true,
                                            separatorChar: '\t',
                                            allowQuoting: true,
                                            allowSparse: false);
            // Build training pipeline
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(mlContext);

            // Train Model
            ITransformer mlModel = TrainModel(mlContext, trainingDataView, trainingPipeline);

            // Evaluate quality of Model
            EvaluateModel(mlContext, mlModel, testDataView);

            // Save model
            SaveModel(mlContext, mlModel, MODEL_FILEPATH, trainingDataView.Schema);
        }

        public static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                
                // Lower case, remove dicretics
                .Append(mlContext.Transforms.Text.NormalizeText(inputColumnName: "Word", outputColumnName: "WordNormalized"))

                // One-hot encode the word
                .Append(mlContext.Transforms.Categorical.OneHotHashEncoding(inputColumnName: "WordNormalized", outputColumnName: "WordOneHot"))

                // Pre-traiend word embedding on the word to tag
                .Append(mlContext.Transforms.Concatenate("WordInArray", new[] { "WordNormalized" }))
                .Append(mlContext.Transforms.Text.ApplyWordEmbedding(inputColumnName: "WordInArray", outputColumnName: "WordEmbedding", modelKind: Microsoft.ML.Transforms.Text.WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding))

                // String statistics (length, vowelCount, numberCount, ...) on title, img1Desc, img2Desc, img3Desc
                .Append(mlContext.Transforms.CopyColumns("text", "Word"))
                .Append(mlContext.Transforms.CustomMapping(new StringStatisticsFeaturizer.StringStatisticsAction().GetMapping(), "StringStatistics"))
                .Append(mlContext.Transforms.Concatenate("StringStatsOnWord", new[] { "length", "vowelCount", "consonantCount", "numberCount", "underscoreCount", "letterCount", "wordCount", "wordLengthAverage", "lineCount", "startsWithVowel", "endsInVowel", "endsInVowelNumber", "lowerCaseCount", "upperCaseCount", "upperCasePercent", "letterPercent", "numberPercent", "longestRepeatingChar", "longestRepeatingVowel" }))

                // Add context of words before/after -- creates columns ContextBefore/ContextAfter
                .Append(mlContext.Transforms.CopyColumns("text", "Context"))
                .Append(mlContext.Transforms.CustomMapping(new AddContextFeaturizer.AddContextAction().GetMapping(), "AddContext"))

                // Unigrams+Bigrams+Trichargrams on the context before
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "ContextBefore", outputColumnName: "ContextBeforeNGrams"))

                // Unigrams+Bigrams+Trichargrams on the context before
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "ContextAfter", outputColumnName: "ContextAfterNGrams"))

                // Unigrams+Bigrams+Trichargrams on the full sentence
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Context", outputColumnName: "ContextNGrams"))

                // Merge to a single feature vector
                .Append(mlContext.Transforms.Concatenate("Features", new[] { "ContextNGrams", "ContextBeforeNGrams", "ContextAfterNGrams", "WordOneHot", "WordNum", "WordEmbedding", "StringStatsOnWord" }))

                // Normalize
                .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))

                // Caching for speed
                .AppendCacheCheckpoint(mlContext);

            // Set the training algorithm 
            var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", numberOfIterations: 10, featureColumnName: "Features"), labelColumnName: "Label")
                                      .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }

        public static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Console.WriteLine("=============== Training  model ===============");

            ITransformer model = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine("=============== End of training process ===============");
            return model;
        }

        private static void EvaluateModel(MLContext mlContext, ITransformer mlModel, IDataView testDataView)
        {
            // Evaluate the model and show accuracy stats
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            IDataView predictions = mlModel.Transform(testDataView);

            // Obtuse method of getting the number of classes
            VBuffer<ReadOnlyMemory<char>> classNamesVBuffer = default;
            predictions.Schema["Score"].GetSlotNames(ref classNamesVBuffer);
            var numClasses = classNamesVBuffer.Length;
            string[] classNames = classNamesVBuffer.DenseValues().Select(a => a.ToString()).ToArray();

            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score", topKPredictionCount: numClasses); // Todo: fix bug to allow for `topKPredictionCount: Int32.MaxValue` 
            PrintMulticlassClassificationMetrics(metrics, classNames);
            Console.WriteLine($"===== Finished Evaluating Model's accuracy with Test data =====");
        }

        private static void SaveModel(MLContext mlContext, ITransformer mlModel, string modelRelativePath, DataViewSchema modelInputSchema)
        {
            // Save/persist the trained model to a .ZIP file
            Console.WriteLine($"=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, modelInputSchema, GetAbsolutePath(modelRelativePath));
            Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        public static void PrintMulticlassClassificationMetrics(MulticlassClassificationMetrics metrics, string[] classNames)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"Accuracy (micro-avg):              {metrics.MicroAccuracy:0.0000}   # 0..1, higher is better");
            Console.WriteLine($"Accuracy (macro):                  {metrics.MacroAccuracy:0.0000}   # 0..1, higher is better");
            Console.WriteLine($"Top-K accuracy:                    [{string.Join(", ", metrics?.TopKAccuracyForAllK?.Select(a => $"{a:0.0000}") ?? new string[] { "Set topKPredictionCount in evaluator to view" })}]   # 0..1, higher is better");
            Console.WriteLine($"Log-loss reduction:                {metrics.LogLossReduction:0.0000;-0.000}   # -Inf..1, higher is better");
            Console.WriteLine($"Log-loss:                          {metrics.LogLoss:0.0000}   # 0..Inf, lower is better");
            Console.WriteLine("\nPer class metrics");
            for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
            {
                Console.WriteLine($"LogLoss for class {i} ({classNames[i] + "):",-11}   {metrics.PerClassLogLoss[i]:0.0000}   # 0..Inf, lower is better");
            }
            for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
            {
                Console.WriteLine($"Precision for class {i} ({classNames[i] + "):",-11} {metrics.ConfusionMatrix.PerClassPrecision[i]:0.0000}   # 0..1, higher is better");
            }
            for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
            {
                Console.WriteLine($"Recall for class {i} ({classNames[i] + "):",-11}    {metrics.ConfusionMatrix.PerClassRecall[i]:0.0000}   # 0..1, higher is better");
            }
            Console.WriteLine("");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine($"************************************************************");
        }
    }
}
