# 🔊 Geeky Kokoro TTS and Voice Mod for ComfyUI

A powerful and feature-rich custom node collection for ComfyUI that integrates the Kokoro TTS (Text-to-Speech) system with advanced voice modification capabilities. This package allows you to generate natural-sounding speech and apply various voice effects within ComfyUI workflows.

> **Important Note**: The kokoro version 0.8.4 (installed via `pip install kokoro==0.8.4`) does not have the pickle issue that affects some other versions.

While this repository uses an MIT license, the models themselves may have different licensing terms. Please check the model licenses before using them in your projects.

<img width="460" alt="Screenshot of Geeky Kokoro TTS in ComfyUI" src="https://github.com/user-attachments/assets/0033cc71-40f1-44c3-b668-94f87d7eab9a" />

## ✨ Features

### Geeky Kokoro TTS Node
- **Multiple Language Support**: English (US and UK) voices
- **Voice Selection**: 27+ high-quality voices to choose from (male and female options)
- **Voice Blending**: Combine two different voices with adjustable blend ratio
- **Speed Control**: Adjust speech rate from 0.5x to 2.0x
- **GPU Acceleration**: Utilize GPU for faster generation (with fallback to CPU)

### Geeky Kokoro Voice Mod Node
- **Voice Morphing**: Transform voices into different characters (Child, Monster, Singer, etc.)
- **Pitch and Formant Control**: Adjust pitch and formant independently
- **Effects Processing**: Apply various audio effects (reverb, echo, distortion, etc.)
- **Presets System**: One-click voice transformations with predefined settings
- **Character Effects**: Special voice effects like Robot, Darth Vader, and more

## 🔧 Installation

### Prerequisites
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed and working
- Python 3.8 or newer
- PyTorch 2.0+ (already included with ComfyUI)

### Option 1: Install via ComfyUI-Manager (Recommended)
1. Open ComfyUI
2. Click on "Manager" in the menu
3. Go to "Install Custom Nodes" tab
4. Search for "Geeky Kokoro TTS"
5. Click "Install" and restart ComfyUI when prompted

### Option 2: Manual Installation

1. Clone this repository into your ComfyUI's `custom_nodes` folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/GeekyGhost/ComfyUI-Geeky-Kokoro-TTS.git
```

2. Install the required dependencies:
```bash
cd ComfyUI-Geeky-Kokoro-TTS
pip install -r requirements.txt

# Optional but recommended for better audio processing:
pip install resampy==0.4.2
pip install librosa>=0.10.0
```

3. Download the required model files manually:
```bash
mkdir -p models
cd models
# Download the model file (about 83 MB)
wget https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
# Download the voices file (about 1.3 MB)
wget https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin
```

4. Restart ComfyUI

### Important Installation Notes

1. **Directory Structure**: When installing manually, ensure the node is in a folder named `ComfyUI-Geeky-Kokoro-TTS` (not `geeky_kokoro_tts`) under your ComfyUI's custom_nodes directory.

2. **Model Files**: The node will look for model files in:
   - `ComfyUI/custom_nodes/ComfyUI-Geeky-Kokoro-TTS/models/` 
   - Hugging Face cache directory (automatically downloaded when using ComfyUI-Manager)

3. **First Run**: On first run, the node will download additional voice data from Hugging Face if needed.

### Troubleshooting Installation

#### Dependency Issues
- **resampy installation failures**: If resampy installation fails, try installing numba first:
  ```bash
  pip install numba
  pip install resampy==0.4.2
  ```
- **librosa issues**: If librosa fails to install, the Voice Mod node will fall back to basic implementations:
  ```bash
  # Try with specific versions
  pip install llvmlite==0.39.0
  pip install numba==0.56.4
  pip install librosa==0.10.0
  ```
- **docopt dependency errors**: If you encounter docopt-related errors, install it explicitly:
  ```bash
  pip install docopt==0.6.2
  ```

#### Model Files Issues
If model download fails, you can manually download from these URLs:
- Kokoro model: https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
- Voices file: https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin

Place these files in the `ComfyUI/custom_nodes/ComfyUI-Geeky-Kokoro-TTS/models/` directory.

#### Model Location Note
When installed via ComfyUI-Manager, the model files are stored in your HuggingFace cache directory:
- Windows: `C:\Users\{username}\.cache\huggingface\`
- Mac/Linux: `~/.cache/huggingface/`

If you're switching between manual and manager installation, you may need to ensure the models are in the correct location.

#### Import Failed Issues
If you see "IMPORT FAILED" in your console log:
1. Make sure you've installed all required dependencies
2. Check that your directory is named `ComfyUI-Geeky-Kokoro-TTS` (not `geeky_kokoro_tts`)
3. Ensure the model files are in the correct location
4. Look for specific error messages in the ComfyUI console/log

## 📚 Detailed Usage Guide

### Geeky Kokoro TTS Node: Complete Parameter Guide

The TTS node converts text to speech with customizable voice settings:

#### Required Parameters:

1. **Text** (String, multiline):
   - Enter the text you want to convert to speech
   - Supports punctuation for natural pauses (commas, periods, question marks)
   - Long texts will be automatically chunked for processing
   - Can include basic formatting such as line breaks

2. **Voice** (Dropdown):
   - 27+ voices categorized by country (US/UK) and gender
   - Each voice has unique characteristics and tone
   - Default: "🇺🇸 🚺 Heart ❤️"
   - Voice selection significantly impacts the output quality and style

3. **Speed** (Float, 0.5 to 2.0):
   - Controls the speech rate
   - 1.0 = normal speed
   - Values below 1.0 slow down speech
   - Values above 1.0 speed up speech 
   - Default: 1.0
   - Recommended range: 0.8-1.2 for most natural sounding results

4. **Use GPU** (Boolean):
   - Enable GPU acceleration for faster processing
   - Only works if CUDA is available
   - Default: True (if GPU is available), False otherwise
   - Automatically falls back to CPU if GPU processing fails

#### Optional Parameters:

1. **Enable Blending** (Boolean):
   - Toggles voice blending feature
   - Default: False
   - When enabled, allows mixing two different voices

2. **Second Voice** (Dropdown):
   - Secondary voice for blending
   - Only used when "Enable Blending" is True
   - Default: "🇺🇸 🚺 Sarah"

3. **Blend Ratio** (Float, 0.0 to 1.0):
   - Controls the mix ratio between primary and secondary voices
   - 1.0 = 100% primary voice
   - 0.5 = 50% primary + 50% secondary (even blend)
   - 0.0 = 100% secondary voice
   - Default: 0.5
   - **Start with subtle blending (0.7-0.9) for most natural results**

#### Outputs:

1. **Audio** (AUDIO):
   - Audio waveform that can be connected to:
     - Audio playback nodes
     - Audio save nodes
     - Voice Mod node for further processing

2. **Text Processed** (STRING):
   - The processed text after normalization
   - Useful for debugging or verifying text chunking

#### Voice Blending Tips:

- **Compatible Voices**: Blending works best with voices from the same language (US+US or UK+UK)
- **Gender Mixing**: Try blending male and female voices for unique character voices
- **Subtle Blending**: Start with a blend ratio of 0.7-0.8 for more natural results
- **Voice Characteristics**: The primary voice's rhythm and pacing usually dominate
- **Creative Uses**: 
  - Blend "Heart" and "Michael" (70/30) for a warm, authoritative voice
  - Blend "Emma" and "George" (50/50) for a neutral UK English voice
  - Blend "Puck" and "Echo" (60/40) for a unique narrative voice

### Geeky Kokoro Voice Mod Node: Detailed Settings Guide

The Voice Mod node lets you apply effects and transformations to audio from the TTS node:

#### Required Parameters:

1. **Audio** (AUDIO):
   - Input audio signal, typically from the TTS node
   - Required for the node to function

2. **Effect Blend** (Float, 0.0 to 1.0):
   - Controls the blend ratio between original and processed audio
   - 0.0 = original audio only (no effects)
   - 1.0 = processed audio only (full effects)
   - Default: 1.0
   - **Start with 0.5-0.7 for more subtle effects**

3. **Output Volume** (Float, -60.0 to 60.0 dB):
   - Adjusts the final output volume in decibels
   - 0.0 = original volume
   - Positive values increase volume
   - Negative values decrease volume
   - -60.0 = effectively muted
   - Default: 0.0
   - **Be careful with values above 6.0 to prevent distortion**

4. **Voice Profile** (Dropdown):
   - Predefined voice transformation presets
   - Options:
     - "None": No profile applied
     - "Cinematic": Deep, resonant voice with reverb (good for trailers)
     - "Monster": Deep, growling voice with distortion
     - "Singer": Optimized for singing with compression and EQ
     - "Robot": Mechanical, synthesized voice
     - "Child": Higher pitch and formants for child-like voice
     - "Darth Vader": Deep, breathing voice with echo
     - "Custom": Use manual settings instead of a preset
   - Default: "None"

5. **Profile Intensity** (Float, 0.0 to 1.0):
   - Controls how strongly the voice profile is applied
   - 0.0 = profile disabled
   - 1.0 = full effect
   - Default: 0.7
   - **Start with 0.4-0.6 for more natural results**
   - **Increase gradually as needed**

#### Optional Parameters (Manual Mode):

1. **Manual Mode** (Boolean):
   - Enables manual control of individual effects
   - Default: False
   - When True, ignores the voice profile and uses individual effect settings
   - When False, only the voice profile is applied (if selected)

2. **Pitch Shift** (Float, -12.0 to 12.0 semitones):
   - Changes the pitch of the voice
   - Positive values = higher pitch
   - Negative values = lower pitch
   - Each 1.0 represents one semitone (musical half-step)
   - Default: 0.0
   - **Start with small adjustments (±2.0) for subtle effects**
   - **Extreme values can sound unnatural**

3. **Formant Shift** (Float, -5.0 to 5.0):
   - Adjusts the formant frequencies (vocal tract characteristics)
   - Positive values = smaller vocal tract (child-like)
   - Negative values = larger vocal tract (bigger/deeper)
   - Default: 0.0
   - **Combine with pitch shift for more natural voice transformations**
   - **Keep values between -2.0 and 2.0 for most natural results**

4. **Reverb Amount** (Float, 0.0 to 1.0):
   - Adds reverberation/echo effect
   - 0.0 = no reverb
   - 1.0 = maximum reverb
   - Default: 0.0
   - **Start with 0.1-0.3 for subtle room ambiance**
   - **Values above 0.5 create dramatic, spacious effects**

5. **Echo Delay** (Float, 0.0 to 1.0):
   - Adds distinct echo repeats
   - 0.0 = no echo
   - 1.0 = maximum echo
   - Default: 0.0
   - **Start with 0.1-0.2 for subtle effect**
   - **Higher values create longer delay between repeats**

6. **Distortion** (Float, 0.0 to 1.0):
   - Adds harmonic distortion to the voice
   - 0.0 = no distortion
   - 1.0 = heavy distortion
   - Default: 0.0
   - **Even small values (0.1-0.2) create noticeable effects**
   - **Useful for robot or monster voices**

7. **Compression** (Float, 0.0 to 1.0):
   - Applies dynamic range compression
   - Makes quiet parts louder and reduces dynamic range
   - 0.0 = no compression
   - 1.0 = heavy compression
   - Default: 0.0
   - **Start with 0.3-0.5 for professional voice quality**
   - **Higher values create radio/broadcast-like sound**

8. **EQ Settings**:
   - **EQ Bass** (Float, -1.0 to 1.0):
     - Adjusts low frequencies
     - Positive values boost bass
     - Negative values reduce bass
     - Default: 0.0
     - **±0.2 produces subtle but noticeable changes**

   - **EQ Mid** (Float, -1.0 to 1.0):
     - Adjusts mid frequencies
     - Positive values emphasize human voice range
     - Negative values make voice more hollow
     - Default: 0.0
     - **Most sensitive band for voice quality**

   - **EQ Treble** (Float, -1.0 to 1.0):
     - Adjusts high frequencies
     - Positive values increase clarity and brightness
     - Negative values reduce sibilance and harshness
     - Default: 0.0
     - **Start with small adjustments (±0.1) for subtle improvements**

9. **Use GPU** (Boolean):
   - Enables GPU acceleration for effects processing
   - Only applies to certain effects
   - Default: False
   - **Generally more useful for the TTS node than the Voice Mod node**

#### Output:

- **Audio** (AUDIO):
  - Processed audio waveform
  - Can be connected to audio playback or save nodes

#### Voice Profile Recommended Settings:

| Profile | Best Used For | Recommended Intensity | Additional Tips |
|---------|---------------|------------------------|-----------------|
| Cinematic | Trailers, narration | 0.4-0.7 | Add compression (0.3) for more punch |
| Monster | Creature voices, villains | 0.3-0.6 | Start low; higher settings sound less human |
| Singer | Music, singing voices | 0.5-0.8 | Works best with slow to moderate speed |
| Robot | Mechanical/AI characters | 0.4-0.7 | Combine with echo (0.1) for sci-fi effect |
| Child | Young characters, cute voices | 0.3-0.5 | Subtle settings sound more realistic |
| Darth Vader | Villains, masked characters | 0.4-0.6 | Add compression (0.2) and bass boost (0.3) |

## 🧪 Effect Combinations and Advanced Techniques

### Creating Custom Voice Types

Experiment with these combinations to create specific voice types:

1. **Elderly Voice**:
   - Pitch Shift: -1.0 to -2.0
   - Formant Shift: -0.5 to -1.0
   - EQ Mid: -0.2
   - EQ Treble: 0.3
   - Add subtle distortion (0.1) for roughness

2. **Broadcast/Radio Voice**:
   - Compression: 0.7
   - EQ Bass: 0.2
   - EQ Mid: 0.4
   - EQ Treble: -0.1
   - Keep pitch and formant at default

3. **Telephone Voice**:
   - EQ Bass: -0.8
   - EQ Treble: -0.5
   - Compression: 0.5
   - Add slight distortion (0.15)

4. **Ghost/Ethereal Voice**:
   - Reverb Amount: 0.7
   - Echo Delay: 0.3
   - Pitch Shift: 1.0
   - Formant Shift: 0.5
   - Add subtle compression (0.2)

5. **Giant/Titan Voice**:
   - Pitch Shift: -5.0
   - Formant Shift: -3.0
   - Reverb Amount: 0.4
   - EQ Bass: 0.6
   - Keep effect blend around 0.7 for realism

### Multi-Node Processing

For more advanced workflows, try these techniques:

1. **Voice Layering**:
   - Create two separate TTS nodes with the same text
   - Process each through separate Voice Mod nodes with different settings
   - Mix the outputs with an audio mixer node
   - Great for creating chorus effects or otherworldly voices

2. **Progressive Effects**:
   - Chain multiple Voice Mod nodes together
   - Apply subtle effects at each stage
   - Example chain: Basic voice → Pitch/Formant → Reverb/Echo → EQ/Compression
   - This creates more natural transitions between effects

3. **Dynamic Text Processing**:
   - Split text by character/emotion
   - Process each segment with appropriate voice settings
   - Combine the audio segments afterward
   - Creates more dynamic and expressive narration

## 🔍 Advanced Technical Details

### Effect Processing Architecture

The Voice Mod node uses a sophisticated multi-layered approach to audio processing:

1. **Quality Tiers**:
   - **High Quality**: Uses librosa and resampy for premium sound quality
   - **Medium Quality**: Falls back to scipy-based algorithms if resampy is unavailable
   - **Basic Quality**: Uses numpy-only implementations if both advanced libraries are missing

2. **Processing Sequence**:
   - Voice profile or manual parameters are evaluated
   - Effects are applied in optimized order (pitch → formant → spatial → tone → dynamics)
   - Blending is applied with original audio
   - Volume adjustment and soft clipping prevent distortion

3. **Resource Management**:
   - Automatic threading for model loading
   - Efficient memory usage with tensor conversions
   - Error recovery with graceful fallbacks
   - Automatic cleanup of temporary resources

### Parameter Sensitivity Guide

Different effects have different sensitivity to parameter changes:

| Effect | Sensitivity | Noticeable Change Threshold | Notes |
|--------|-------------|----------------------------|-------|
| Pitch Shift | High | ±0.5 semitones | Most noticeable effect |
| Formant Shift | Medium | ±0.3 units | Best combined with pitch |
| Reverb | Low | 0.1 units | Subtle at low values |
| Echo | Medium | 0.1 units | Creates distinct repeats |
| Distortion | Very High | 0.05 units | Use sparingly |
| Compression | Low | 0.2 units | Cumulative effect |
| EQ Bass | Medium | 0.2 units | Affects voice weight |
| EQ Mid | High | 0.1 units | Affects voice clarity |
| EQ Treble | Medium | 0.1 units | Affects articulation |

## 🎭 Available Voices

### US English Voices
| Voice Name | Description | Best For |
|------------|-------------|----------|
| 🇺🇸 🚺 Heart ❤️ | Warm female voice | Narration, friendly content |
| 🇺🇸 🚺 Bella 🔥 | Energetic female voice | Marketing, upbeat content |
| 🇺🇸 🚺 Nicole 🎧 | Clear female voice | Audiobooks, instructional |
| 🇺🇸 🚺 Aoede | Melodic female voice | Poetic content, storytelling |
| 🇺🇸 🚺 Kore | Soft female voice | Calm narration, meditative |
| 🇺🇸 🚺 Sarah | Neutral female voice | General purpose, business |
| 🇺🇸 🚺 Nova | Modern female voice | Tech content, contemporary |
| 🇺🇸 🚺 Sky | Bright female voice | Cheerful content, children's stories |
| 🇺🇸 🚺 Alloy | Smooth female voice | Professional narration |
| 🇺🇸 🚺 Jessica | Articulate female voice | Educational content |
| 🇺🇸 🚺 River | Flowing female voice | Nature content, documentaries |
| 🇺🇸 🚹 Michael | Deep male voice | Authoritative content |
| 🇺🇸 🚹 Fenrir | Strong male voice | Dramatic narration |
| 🇺🇸 🚹 Puck | Playful male voice | Light-hearted content |
| 🇺🇸 🚹 Echo | Resonant male voice | Atmospheric narration |
| 🇺🇸 🚹 Eric | Clear male voice | Business, instructional |
| 🇺🇸 🚹 Liam | Young male voice | Modern content, casual |
| 🇺🇸 🚹 Onyx | Rich male voice | Luxury, premium content |
| 🇺🇸 🚹 Adam | Neutral male voice | All-purpose narration |

### UK English Voices
| Voice Name | Description | Best For |
|------------|-------------|----------|
| 🇬🇧 🚺 Emma | Refined female UK voice | Formal content, documentaries |
| 🇬🇧 🚺 Isabella | Elegant female UK voice | Upscale content, sophisticated |
| 🇬🇧 🚺 Alice | Clear female UK voice | Educational, storytelling |
| 🇬🇧 🚺 Lily | Gentle female UK voice | Children's content, soft narration |
| 🇬🇧 🚹 George | Professional male UK voice | Business, authoritative content |
| 🇬🇧 🚹 Fable | Narrative male UK voice | Storytelling, fiction |
| 🇬🇧 🚹 Lewis | Modern male UK voice | Tech content, contemporary |
| 🇬🇧 🚹 Daniel | Neutral male UK voice | All-purpose UK narration |

## 💡 Optimization Tips and Best Practices

1. **Text Processing Tips**:
   - Use proper punctuation for natural pauses
   - Break long paragraphs into sentences
   - Use shorter sentences for more reliable processing
   - Avoid unusual symbols or characters when possible

2. **Performance Optimization**:
   - Process shorter text segments for faster results and lower memory usage
   - Use GPU acceleration only when processing longer texts
   - Disable unused effects to reduce processing time
   - Save generated audio for reuse in iterative workflows

3. **Voice Selection Guidelines**:
   - Match voice to content type (e.g., Heart for friendly, Michael for authoritative)
   - US voices typically work better for casual content
   - UK voices often suit formal or educational content
   - Test multiple voices with the same content to find the best match

4. **Effect Parameter Guidelines**:
   - **Always start with low settings** (30-40% of maximum) and increase gradually
   - Apply effects in stages rather than all at once
   - Use effect_blend parameter to control overall intensity
   - Save your favorite parameter combinations for reuse

5. **Memory Usage Considerations**:
   - Process texts under 1000 characters for optimal performance
   - Close other GPU-intensive applications when using GPU acceleration
   - Monitor system resource usage during processing
   - Consider using CPU mode if experiencing GPU memory errors

## 🔍 Troubleshooting

### Common Issues and Solutions

1. **Audio Quality Issues**:
   - **Choppy or stuttering audio**: Reduce effects intensity or process shorter text segments
   - **Robotic artifacts**: Lower formant shift values and use higher quality libraries
   - **Unnatural pauses**: Check text punctuation and formatting
   - **Distortion**: Lower output volume and reduce effect intensity

2. **Performance Issues**:
   - **Slow processing**: Switch to CPU mode for short texts, GPU for longer ones
   - **High memory usage**: Process text in smaller chunks
   - **CUDA errors**: Ensure your GPU drivers are updated and CUDA is properly installed
   - **Timeout errors**: Increase node timeout settings in ComfyUI configuration

3. **Effect-Specific Issues**:
   - **Weird pitch shift artifacts**: Install resampy for better quality
   - **Reverb sounds metallic**: Reduce reverb amount to 0.3-0.4
   - **Compression pumping**: Reduce compression to 0.3-0.5
   - **Voice blending sounds unnatural**: Adjust blend ratio closer to 1.0 (more primary voice)

4. **Diagnostic Steps**:
   - Check ComfyUI console for error messages
   - Try disabling effects one by one to identify problematic ones
   - Test with default settings to establish a baseline
   - Verify model files are properly installed and accessible

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests if you have improvements or bug fixes.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgments

- [Kokoro TTS Project](https://github.com/nazdridoy/kokoro-tts) - The foundation of the TTS engine
- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) - Great Huggingface repo with the model files
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The UI framework this node integrates with
- [librosa](https://librosa.org/) - Audio processing library used for high-quality effects
- [scipy](https://scipy.org/) - Scientific computing library used for audio signal processing
- [resampy](https://github.com/bmcfee/resampy) - High-quality audio resampling library