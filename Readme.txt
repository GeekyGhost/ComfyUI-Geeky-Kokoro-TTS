# 🔊 Geeky Kokoro TTS for ComfyUI

A powerful and feature-rich custom node for ComfyUI that integrates the Kokoro TTS (Text-to-Speech) system, allowing you to generate natural-sounding speech from text within your ComfyUI workflows. This extended version includes advanced voice manipulation features, batch processing capabilities, and improved text preprocessing.

![Geeky Kokoro TTS Node Banner](https://raw.githubusercontent.com/yourusername/geeky-kokoro-tts/main/images/banner.png)

## ✨ Features

### Core Features
- **Multiple Language Support**: Supports English (US and UK) voices
- **Voice Selection**: 27+ voices to choose from (male and female options)
- **Voice Blending**: Combine two different voices with adjustable blend ratio
- **Speed Control**: Adjust speech rate from 0.5x to 2.0x
- **GPU Acceleration**: Utilize GPU for faster generation (with fallback to CPU)
- **Save to Multiple Formats**: WAV, MP3, FLAC, and OGG

### Advanced Features
- **Pitch Adjustment**: Basic pitch shifting for voice customization
- **Text Preprocessing**: Multiple modes for handling different text formats
- **Emotion Emphasis**: Add emotional inflections to generated speech
- **Pause Control**: Automatically add natural pauses at punctuation
- **Batch Processing**: Process the same text with multiple voices simultaneously
- **Random Voice Selection**: Randomly select voices from specific categories
- **Text Splitting**: Handle long texts by automatically splitting and processing in chunks

## 🔧 Installation

### Automatic Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/geeky-kokoro-tts.git
```

2. Run the installer script:
```bash
cd geeky-kokoro-tts
python install.py
```

The installer will:
- Detect your ComfyUI installation
- Install the node files
- Install required dependencies
- Download the Kokoro model files (if needed)

### Manual Installation

1. Clone this repository into your ComfyUI's `custom_nodes` folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/geeky-kokoro-tts.git geeky_kokoro_tts
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the required model files:
```bash
mkdir -p ComfyUI/custom_nodes/geeky_kokoro_tts/models
cd ComfyUI/custom_nodes/geeky_kokoro_tts/models
wget https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
wget https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin
```

4. Restart ComfyUI

## 📚 Usage Guide

### Basic Text-to-Speech

1. Add the "🔊 Geeky Kokoro TTS" node to your workflow
2. Enter the text you want to convert to speech
3. Select a voice from the dropdown
4. Adjust the speed as needed
5. Connect to a "💾 Save Audio File" or "🎧 Preview Audio" node

### Voice Blending

1. Enable the "enable_blending" option
2. Select primary voice
3. Select secondary voice in the optional inputs
4. Adjust the blend ratio (0.0 to 1.0)
   - 1.0 = 100% primary voice
   - 0.5 = 50% primary + 50% secondary
   - 0.0 = 100% secondary voice

### Text Preprocessing

The node offers several text preprocessing options:

- **Default**: Basic cleaning and normalization
- **Remove Brackets**: Removes text within brackets and parentheses
- **Expand Abbreviations**: Expands common abbreviations (e.g., "Dr." → "Doctor")
- **Normalize Numbers**: Converts numbers to words for more natural speech

### Emotion Emphasis

Add emotional inflection to your speech:

- **None**: Standard speech with no specific emphasis
- **Happy**: Adds exclamations and happy emphasis
- **Sad**: Adds hesitations and slower pace for sad tone
- **Excited**: Adds exclamations and speed up for excitement
- **Calm**: Adds pauses for a calm delivery
- **Question**: Converts statements to questions

### Working with Long Texts

For long texts, use the "✂️ Text Splitter TTS" node:

1. Input your long text
2. Choose a splitting mode (Sentences, Paragraphs, or Fixed Length)
3. Set the maximum chunk size
4. Connect to output nodes as usual

### Batch Processing

To process the same text with multiple voices at once:

1. Add the "⚡ Batch Voice Processor" node
2. Input your text
3. Select a voice category (US Female, US Male, UK Female, UK Male)
4. Set the number of voices to process
5. Connect the multiple audio outputs to separate save/preview nodes

### Random Voice Selection

For creative applications or testing:

1. Add the "🎲 Random Voice Selector" node
2. Choose a voice category or "All Voices"
3. Set a seed for reproducibility
4. Pass the selected voice to the TTS node

## 🎭 Available Voices

### US English Voices
| Voice Name | Description |
|------------|-------------|
| 🇺🇸 🚺 Heart ❤️ | Female US English voice |
| 🇺🇸 🚺 Bella 🔥 | Female US English voice |
| 🇺🇸 🚺 Nicole 🎧 | Female US English voice |
| 🇺🇸 🚺 Aoede | Female US English voice |
| 🇺🇸 🚺 Kore | Female US English voice |
| 🇺🇸 🚺 Sarah | Female US English voice |
| 🇺🇸 🚺 Nova | Female US English voice |
| 🇺🇸 🚺 Sky | Female US English voice |
| 🇺🇸 🚺 Alloy | Female US English voice |
| 🇺🇸 🚺 Jessica | Female US English voice |
| 🇺🇸 🚺 River | Female US English voice |
| 🇺🇸 🚹 Michael | Male US English voice |
| 🇺🇸 🚹 Fenrir | Male US English voice |
| 🇺🇸 🚹 Puck | Male US English voice |
| 🇺🇸 🚹 Echo | Male US English voice |
| 🇺🇸 🚹 Eric | Male US English voice |
| 🇺🇸 🚹 Liam | Male US English voice |
| 🇺🇸 🚹 Onyx | Male US English voice |
| 🇺🇸 🚹 Adam | Male US English voice |

### UK English Voices
| Voice Name | Description |
|------------|-------------|
| 🇬🇧 🚺 Emma | Female UK English voice |
| 🇬🇧 🚺 Isabella | Female UK English voice |
| 🇬🇧 🚺 Alice | Female UK English voice |
| 🇬🇧 🚺 Lily | Female UK English voice |
| 🇬🇧 🚹 George | Male UK English voice |
| 🇬🇧 🚹 Fable | Male UK English voice |
| 🇬🇧 🚹 Lewis | Male UK English voice |
| 🇬🇧 🚹 Daniel | Male UK English voice |

## 🧙‍♂️ Advanced Voice Customization

### Customizing Pronunciation

For custom pronunciations, you can use Markdown link syntax and /slashes/:
- `[Kokoro](/kˈOkəɹO/)` - Specify exact pronunciation
- Use punctuation `;:,.!?—…"()""` to adjust intonation
- Add stress markers `ˈ` and `ˌ`
- Lower stress with `[1 level](-1)` or `[2 levels](-2)`
- Raise stress with `[words](+1)` or `[words](+2)`

### Pitch Adjustment

The pitch adjustment slider allows you to:
- Raise the pitch (values > 0) for a higher-pitched voice
- Lower the pitch (values < 0) for a deeper voice
- Note: This is a basic implementation and effects vary by voice

### Pause Control

The "pauses_at_punctuation" option automatically adds natural pauses at punctuation marks, which can significantly improve the rhythm and naturalness of longer texts.

## 🔮 Workflow Examples

### Basic Text-to-Speech Workflow
![Basic TTS Workflow](https://raw.githubusercontent.com/yourusername/geeky-kokoro-tts/main/images/basic_workflow.png)

### Voice Blending Workflow
![Voice Blending Workflow](https://raw.githubusercontent.com/yourusername/geeky-kokoro-tts/main/images/voice_blending.png)

### Long Text Processing Workflow
![Long Text Workflow](https://raw.githubusercontent.com/yourusername/geeky-kokoro-tts/main/images/long_text_workflow.png)

### Batch Processing Workflow
![Batch Processing Workflow](https://raw.githubusercontent.com/yourusername/geeky-kokoro-tts/main/images/batch_workflow.png)

### Integration with Image Generation
![Advanced Workflow](https://raw.githubusercontent.com/yourusername/geeky-kokoro-tts/main/images/advanced_workflow.png)

## 💡 Tips and Tricks

1. **Memory Optimization**: If you're processing large amounts of text, consider using the text splitter node to break it into manageable chunks.

2. **Voice Consistency**: When processing multiple chunks of text, use the same voice and settings for all chunks to maintain consistency.

3. **Custom Voices**: Try different blend ratios between voices to create your own custom voice combinations.

4. **Emotional Speech**: Combine emotion emphasis with appropriate speed settings for more realistic emotional speech (e.g., "Excited" with higher speed, "Sad" with lower speed).

5. **Pitch Effects**: Slight pitch adjustments (+/- 0.2) can make a significant difference in voice character without sounding unnatural.

6. **GPU Acceleration**: GPU processing is significantly faster for longer texts, but CPU processing may be more stable in some environments.

## 🔍 Troubleshooting

- **GPU Issues**: If you encounter GPU-related errors, try switching to CPU mode by unchecking the "use_gpu" option.
- **Memory Errors**: If you run into memory issues, try processing shorter text segments using the text splitter node.
- **Voice Not Found**: Make sure the model files are correctly installed in the `models` directory.
- **Audio Distortion**: If audio sounds distorted, try reducing the pitch adjustment to a value closer to 0.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests through GitHub.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgments

- [Original Kokoro TTS Project](https://github.com/nazdridoy/kokoro-tts)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- All contributors and testers who have helped improve this node