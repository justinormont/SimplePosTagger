using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System.Collections.Generic;

namespace Simple.ConsoleApp
{
    public class AddContextFeaturizer
    {
        #region StringStatistics CustomMapping
        [CustomMappingFactoryAttribute("AddContext")]
        public class AddContextAction : CustomMappingFactory<RowWithStringAndPosition, RowWithContext>
        {

            public static Action<RowWithStringAndPosition, RowWithContext> CustomAction = (RowWithStringAndPosition input, RowWithContext output) =>
            {
                string str = input.Text is string ? (string)(object)input.Text : string.Join(" ", input.Text);
                char[] text = str.ToCharArray();

                string[] split = str.Split(' ');

                // Note: These are written for clarity; for speed, a single pass of the character array could be done.
                output.ContextBefore = String.Join(' ', split.Take((int)input.WordNum));
                output.ContextAfter = String.Join(' ', split.TakeLast(split.Length - (int)input.WordNum - 1));
            };

            public override Action<RowWithStringAndPosition, RowWithContext> GetMapping() => CustomAction;
        }

        public class RowWithStringAndPosition
        {
            [ColumnName("text")] // Harded coded to use an input column named "text"
            public string Text { get; set; }

            [ColumnName("WordNum")] // Harded coded to use an input column named "WordIndex"
            public float WordNum { get; set; }

        }

        public class RowWithContext
        {
            [ColumnName("ContextBefore")]
            public string ContextBefore { get; set; }

            [ColumnName("ContextAfter")]
            public string ContextAfter { get; set; }
        }
        #endregion
    }
}