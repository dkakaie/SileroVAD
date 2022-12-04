namespace SileroVAD
{
    public class SpeechSegment
    {
        public float Score { get; set; }
        public int Start { get; set; }
        public int End { get; set; }
        public double StartSeconds { get; set; }
        public double StopSeconds { get; set; }
    }
}