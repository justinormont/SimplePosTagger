using Microsoft.ML.Data;

namespace Simple.Model.DataModels
{
    public class ModelInput
    {
        [ColumnName("Label"), LoadColumn(0)]
        public string Label { get; set; }


        [ColumnName("WordNum"), LoadColumn(1)]
        public float WordNum { get; set; }


        [ColumnName("Word"), LoadColumn(2)]
        public string Word { get; set; }


        [ColumnName("Context"), LoadColumn(3)]
        public string Context { get; set; }
    }
}
