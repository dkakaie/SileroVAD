<h1 align="center">Silero VAD for .NET</h1>
<br/>
<p  align="center">
<img alt="alt_text" src="https://github.com/dkakaie/SileroVAD/blob/master/screenshot.png?raw=true" />
</p>

**Silero VAD** - pre-trained enterprise-grade [Voice Activity Detector](https://en.wikipedia.org/wiki/Voice_activity_detection) for .NET.

<br/>
<h2 align="center">Key Features</h2>
<br/>

- **Pure .NET solution**
  No dependencies other than ONNX Runtime for inference and NAudio for audio loading.
  
 - **Supports MP3 and WAV files**
  Automatic MP3 conversion using NAudio (MediaFoundation, Windows Vista and up).
<br/>
<h2 align="center">Limitations</h2>
<br/>

- **16khz audio only**
  This is a limitation of the ONNX model used.
