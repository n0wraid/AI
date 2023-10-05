using System;
using System.Collections.Generic;
using System.Linq;

class Program
{
    static void Main()
    {
        // Белгіленген деректер нүктелері бар үлгі деректер жинағы (мүмкіндіктер мен белгілер)
        List<DataPoint> dataset = new List<DataPoint>
        {
            new DataPoint(new double[] { 2.0, 3.0 }, "A"),
            new DataPoint(new double[] { 3.0, 4.0 }, "A"),
            new DataPoint(new double[] { 5.0, 7.0 }, "B"),
            new DataPoint(new double[] { 6.0, 8.0 }, "B"),
        };

        // Жіктелетін жаңа деректер нүктесі
        DataPoint newDataPoint = new DataPoint(new double[] { 4.0, 5.0 });
        Console.WriteLine("Enter k: ");
        int k = Convert.ToInt32(Console.ReadLine()); // k мәнін орнатыңыз (қаралатын көршілер саны)

        string predictedLabel = ClassifyKNN(dataset, newDataPoint, k);

        Console.WriteLine($"Predicted Label: {predictedLabel}");
        Console.ReadLine();
    }

    // Екі деректер нүктесі арасындағы евклидтік қашықтық
    static double EuclideanDistance(double[] point1, double[] point2)
    {
        double sum = 0.0;
        for (int i = 0; i < point1.Length; i++)
        {
            sum += Math.Pow(point1[i] - point2[i], 2);
        }
        return Math.Sqrt(sum);
    }

    // k-Nearest Neighbors көмегімен деректер нүктесін жіктеңіз
    static string ClassifyKNN(List<DataPoint> dataset, DataPoint newDataPoint, int k)
    {
        // newDataPoint және деректер жиынындағы барлық нүктелер арасындағы қашықтықты есептеңіз
        var distances = dataset.Select(dp => new
        {
            DataPoint = dp,
            Distance = EuclideanDistance(newDataPoint.Features, dp.Features)
        }).OrderBy(d => d.Distance).Take(k);

        // k-ең жақын көршілер арасында әрбір белгінің пайда болуын санаңыз
        var labelCounts = new Dictionary<string, int>();
        foreach (var neighbor in distances)
        {
            if (!labelCounts.ContainsKey(neighbor.DataPoint.Label))
            {
                labelCounts[neighbor.DataPoint.Label] = 0;
            }
            labelCounts[neighbor.DataPoint.Label]++;
        }

        //  k - ең жақын көршілер арасында ең көп кездесетін белгіні табыңыз
        string predictedLabel = labelCounts.OrderByDescending(kv => kv.Value).First().Key;

        return predictedLabel;
    }
}

// Мүмкіндіктері мен белгісі бар деректер нүктесін көрсетеді
class DataPoint
{
    public double[] Features { get; }
    public string Label { get; }

    public DataPoint(double[] features, string label = "")
    {
        Features = features;
        Label = label;
    }
}