using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;

namespace SileroVAD
{
    /// <summary>
    ///     SileroVAD Processor
    /// </summary>
    public class SileroProcessor
    {
        private const int WindowSizeSamples = 1536;
        private const int SamplingRate = 16000;
        private const int MinSilenceDurationMs = 100;
        private const int MinSilenceSamples = SamplingRate * MinSilenceDurationMs / 1000;
        private const int MinSpeechDurationMs = 250;
        private const int MinSpeechSamples = SamplingRate * MinSpeechDurationMs / 1000;
        private const int SpeechPadMs = 30;
        private const int SpeechPadSamples = SamplingRate * SpeechPadMs / 1000;
        private readonly float _negativeThreshold;

        private readonly InferenceSession _session;
        private readonly float _threshold;

        /// <summary>
        ///     Initializes a new instance of the <see cref="SileroProcessor" /> class.
        /// </summary>
        /// <param name="modelPath">Path to silero_vad.onnx file.</param>
        /// <param name="threshold">Threshold above which to consider as speech.</param>
        /// <exception cref="System.IO.FileNotFoundException">@"VAD model {modelPath} was not found.",</exception>
        public SileroProcessor(string modelPath, float threshold = 0.5f)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($@"VAD model {modelPath} was not found.", modelPath);
            _session = new InferenceSession(modelPath);
            _threshold = threshold;
            _negativeThreshold = _threshold - 0.15f;
        }

        ~SileroProcessor()
        {
            _session.Dispose();
        }

        /// <summary>
        ///     Reads wav or mp3 file and returns speech segments.
        /// </summary>
        /// <param name="file">Path to file.</param>
        /// <returns>Speech segments</returns>
        public IEnumerable<SpeechSegment> ProcessFile(string file)
        {
            var memStream = new MemoryStream();
            float[] samples;

            if (file.EndsWith(".mp3"))
                using (var reader = new Mp3FileReader(file))
                {
                    var outFormat = new WaveFormat(16000, 16, 1);
                    using (var resampler = new MediaFoundationResampler(reader, outFormat))
                    {
                        resampler.ResamplerQuality = 60;
                        WaveFileWriter.WriteWavFileToStream(memStream, resampler);
                        memStream.Seek(0, SeekOrigin.Begin);
                    }
                }
            else
                using (var reader = new WaveFileReader(file))
                {
                    var format = reader.ToSampleProvider().WaveFormat;
                    if (format.SampleRate != 16000 || format.Channels != 1)
                    {
                        var outFormat = new WaveFormat(16000, 16, 1);
                        using (var resampler = new MediaFoundationResampler(reader, outFormat))
                        {
                            resampler.ResamplerQuality = 60;
                            WaveFileWriter.WriteWavFileToStream(memStream, resampler);
                            memStream.Seek(0, SeekOrigin.Begin);
                        }
                    }
                    else
                    {
                        memStream = new MemoryStream(File.ReadAllBytes(file));
                    }
                }

            using (var reader = new WaveFileReader(memStream))
            {
                samples = new float[reader.SampleCount];
                reader.ToSampleProvider().Read(samples, 0, (int)reader.SampleCount);
            }

            return Process(samples);
        }

        /// <summary>
        ///     Detect speech segments by providing audio samples. Source must be mono, 16khz wav.
        /// </summary>
        /// <param name="wavData">Audio samples.</param>
        /// <returns>Speech segments</returns>
        public IEnumerable<SpeechSegment> Process(float[] wavData)
        {
            var chunks = wavData.Chunkify(WindowSizeSamples);

            var cParameters = new float[128];
            var hParameters = new float[128];

            //Model Tensor Dimensions
            var audioDimensions = new[] { 1, WindowSizeSamples };
            var cDimensions = new[] { 2, 1, 64 };
            var hDimensions = new[] { 2, 1, 64 };

            var probs = new List<float>();

            foreach (var chunk in chunks)
            {
                var container = new List<NamedOnnxValue>();
                var audioTensor = chunk.ToTensor().Reshape(audioDimensions);
                var cTensor = cParameters.ToArray().ToTensor().Reshape(cDimensions);
                var hTensor = hParameters.ToTensor().Reshape(hDimensions);

                container.Add(NamedOnnxValue.CreateFromTensor("input", audioTensor));
                container.Add(NamedOnnxValue.CreateFromTensor("c0", cTensor));
                container.Add(NamedOnnxValue.CreateFromTensor("h0", hTensor));

                using (var results = _session.Run(container))
                {
                    var output = results.First(x => x.Name == "output");
                    var hnTensor = results.First(x => x.Name == "hn").AsTensor<float>();
                    var cnTensor = results.First(x => x.Name == "cn").AsTensor<float>();

                    hParameters = hnTensor.ToArray();
                    cParameters = cnTensor.ToArray();

                    probs.Add(output.AsEnumerable<float>().Last());
                }
            }

            return PostProcess(probs.ToArray(), wavData.Length);
        }

        /// <summary>
        ///     Processes the outputs of Silero network.
        /// </summary>
        /// <param name="probs">Raw probs output as predicted by the network.</param>
        /// <param name="wavDataLength">Length of the wav data.</param>
        /// <returns></returns>
        private IEnumerable<SpeechSegment> PostProcess(float[] probs, int wavDataLength)
        {
            var speeches = new List<SpeechSegment>();
            var currentSpeech = new SpeechSegment();
            var tempEnd = 0;
            var triggered = false;
            var index = 0;

            for (var j = 0; j < probs.Length; j++, index++)
            {
                var prob = probs[index];
                if (prob >= _threshold && tempEnd != 0) tempEnd = 0;
                if (prob >= _threshold && !triggered)
                {
                    triggered = true;
                    currentSpeech = new SpeechSegment
                    {
                        Score = prob,
                        Start = WindowSizeSamples * index
                    };
                    continue;
                }

                if (!(prob < _negativeThreshold) || !triggered) continue;
                if (tempEnd == 0) tempEnd = WindowSizeSamples * index;

                if (WindowSizeSamples * index - tempEnd < MinSilenceSamples)
                {
                    currentSpeech.End = tempEnd;
                }
                else
                {
                    if (currentSpeech.End - currentSpeech.Start > MinSpeechSamples) speeches.Add(currentSpeech);
                    tempEnd = 0;
                    currentSpeech = null;
                    triggered = false;
                }
            }

            if (currentSpeech != null)
            {
                currentSpeech.End = wavDataLength;
                speeches.Add(currentSpeech);
            }

            index = 0;
            for (var j = 0; j < speeches.Count; j++, index++)
            {
                var seg = speeches[index];
                if (index == 0) seg.Start = Math.Max(0, seg.Start - SpeechPadSamples);

                if (index != speeches.Count - 1)
                {
                    var silenceDuration = speeches[index + 1].Start - seg.End;
                    if (silenceDuration < 2 * SpeechPadSamples)
                    {
                        var quotient = Math.DivRem(silenceDuration, 2, out _);
                        seg.End += quotient;
                        speeches[index + 1].Start = Math.Max(0, speeches[index + 1].Start - quotient);
                    }
                    else
                    {
                        seg.End += SpeechPadSamples;
                    }
                }
                else
                {
                    seg.End = Math.Min(wavDataLength, seg.End + SpeechPadSamples);
                }

                seg.StartSeconds = Math.Round((double)seg.Start / SamplingRate, 1);
                seg.StopSeconds = Math.Round((double)seg.End / SamplingRate, 1);
            }

            return speeches;
        }
    }
}