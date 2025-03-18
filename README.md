# 🔊 Geeky Kokoro TTS and Voice Mod for ComfyUI

pip install kokoro==0.8.4 does not have the pickle issue

While this repo uses an MIT license, the models themselves do not. Make sure you check the model licenses as well. 

A powerful and feature-rich custom node collection for ComfyUI that integrates the Kokoro TTS (Text-to-Speech) system with advanced voice modification capabilities. This package allows you to generate natural-sounding speech and apply various voice effects within ComfyUI workflows.

<img width="1044" alt="Screenshot 2025-03-18 075032" src="https://github.com/user-attachments/assets/a4ca7129-153e-4bfb-9e5f-7510d598d099" />


## ✨ Features

### Geeky Kokoro TTS Node
- **Multiple Language Support**: English (US and UK) voices
- **Voice Selection**: 27+ voices to choose from (male and female options)
- **Voice Blending**: Combine two different voices with adjustable blend ratio
- **Speed Control**: Adjust speech rate from 0.5x to 2.0x
- **GPU Acceleration**: Utilize GPU for faster generation (with fallback to CPU)

### Geeky Kokoro Voice Mod Node (Work in Progress)
- **Voice Morphing**: Transform voices into different characters (Child, Teen, Elder, etc.)
- **Pitch and Formant Control**: Adjust pitch and formant independently
- **Effects Processing**: Apply various audio effects (reverb, echo, distortion, etc.)
- **Presets System**: One-click voice transformations with predefined settings
- **Character Effects**: Special voice effects like Robot, Telephone, Megaphone, etc.

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

3. **First Run**: On first run, the node will download additional voice data from Hugging Face if needed

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

## 📚 Usage Guide

### Geeky Kokoro TTS Node

1. Add the "🔊 Geeky Kokoro TTS" node to your workflow
2. Connect inputs:
   - **text**: Enter the text you want to convert to speech
   - **voice**: Select a voice from the dropdown
   - **speed**: Adjust the speech rate (0.5 to 2.0)
   - **use_gpu**: Enable GPU acceleration (if available)
3. Optional parameters:
   - **enable_blending**: Turn on voice blending
   - **second_voice**: Select a second voice for blending
   - **blend_ratio**: Adjust the mix between primary and secondary voices
4. Outputs:
   - **audio**: Connect to audio playback or save nodes
   - **text_processed**: The processed text after normalization

#### Voice Blending Example
To create a custom voice blend:
1. Enable the "enable_blending" toggle
2. Select a primary voice (e.g., "🇺🇸 🚺 Heart ❤️")
3. Choose a secondary voice (e.g., "🇬🇧 🚹 George")
4. Adjust the blend ratio (0.0 to 1.0):
   - 1.0 = 100% primary voice
   - 0.5 = 50% primary + 50% secondary
   - 0.0 = 100% secondary voice

This can create unique voice combinations that aren't available as standard voices.

### Geeky Kokoro Voice Mod Node (Beta)

> ⚠️ **Note**: The Voice Mod node is currently in beta. Some effects may not work as expected and may be subject to change.

1. Add the "🔊 Geeky Kokoro Advanced Voice" node to your workflow
2. Connect the audio input (typically from the TTS node)
3. Choose between:
   - **Presets**: Quick voice transformations (e.g., "Chipmunk", "Robot Voice", "Podcast")
   - **Custom Settings**: Manually configure individual effects

#### Effect Groups (Custom Settings)
The Voice Mod node organizes effects into logical groups that can be enabled independently:

- **Voice Morphing**: Transform voice character (Child, Masculine, Elder, etc.)
- **Pitch & Formant**: Adjust pitch, formant shift, and auto-tune
- **Time Effects**: Change playback speed or add vibrato
- **Spatial Effects**: Add reverb and echo
- **Tone Controls**: Adjust EQ bands (bass, mids, treble) and add harmonics
- **Effects**: Apply distortion, tremolo, bitcrush, and noise reduction
- **Dynamics**: Compression and analog warmth simulation
- **Character Effects**: Special transformations like Robot, Telephone, Whisper, etc.

#### Using Presets
For quick voice transformations, use the preset system:
1. Select a preset from the dropdown (e.g., "Robot Voice", "Chipmunk", "Deep Voice")
2. Adjust the preset_strength parameter (0.0 to 1.0) to control intensity
3. Set effect_blend (0.0 to 1.0) to mix with the original voice

## 🔍 Technical Details

### Audio Processing Architecture

The Voice Mod node uses a multi-layered approach to audio processing:

1. **Primary Implementation**: Uses librosa and resampy for high-quality processing
2. **Fallback Layer 1**: Uses scipy-based algorithms when resampy is unavailable
3. **Fallback Layer 2**: Uses numpy-only implementations when scipy is unavailable

This ensures the node can function even when optional dependencies are missing, but with potentially reduced quality.

### Key Code Components

#### Fallback System
The `audio_utils.py` file contains fallback implementations for when specialized audio libraries aren't available:

```python
# Example of the phase vocoder fallback when resampy isn't available
def stft_phase_vocoder(audio, sr, n_steps, bins_per_octave=12):
    """
    Phase vocoder pitch shifting using STFT, more advanced than simple resampling
    """
    if abs(n_steps) < 0.01:
        return audio
        
    # Convert steps to rate
    rate = 2.0 ** (-n_steps / bins_per_octave)
    
    # STFT parameters
    n_fft = 2048
    hop_length = n_fft // 4
    
    # Compute STFT
    D = stft(audio, n_fft=n_fft, hop_length=hop_length)
    
    # Create new spectrogram with adjusted phase progression
    time_steps = D.shape[1]
    new_time_steps = int(time_steps / rate)
    
    # Phase advance
    phase_adv = np.linspace(0, np.pi * rate, D.shape[0])[:, np.newaxis]
    
    # Time-stretch and phase manipulation logic...
    
    # Invert STFT
    y_shift = istft(D_stretch, hop_length=hop_length, length=len(audio))
    
    return y_shift
```

#### Voice Morphing
The voice morphing system uses a combination of effects to create realistic voice transformations:

```python
# Simplified example of voice morphing parameters
morph_params = {
    "Child": {
        "pitch_shift": 4.0,
        "formant_shift": 2.0,
        "brightness": 0.4, 
        "breathiness": 0.3,
        "bass_boost": -0.3,
        "mid_boost": 0.3,
        "compression": 0.2
    },
    "Elder": {
        "pitch_shift": -1.0,
        "formant_shift": -0.5,
        "brightness": -0.2, 
        "breathiness": 0.4,
        "bass_boost": 0.2,
        "mid_boost": -0.2,
        "compression": 0.0,
        "tremolo": 0.2
    },
    # Other voice types...
}
```

### GPU Acceleration
The TTS node leverages GPU acceleration for the Kokoro model when available:

```python
# GPU loading in TTS node with fallback
if use_gpu and True not in self.MODEL and torch.cuda.is_available():
    try:
        with self.MODEL_LOCK:
            self.MODEL[True] = KModel(repo_id='hexgrad/Kokoro-82M').to('cuda').eval()
    except Exception as e:
        print(f"GPU load failed: {e}. Using CPU.")
        use_gpu = False
```

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

## 🧙‍♂️ Advanced Techniques

### Voice Mod Presets
The Voice Mod node includes several presets for common voice transformations:

| Preset | Description |
|--------|-------------|
| Chipmunk | High-pitched, child-like voice |
| Deep Voice | Low-pitched, authoritative voice |
| Robot Voice | Mechanical, synthesized voice |
| Phone Call | Classic telephone audio quality |
| Elder Voice | Aged voice with characteristic tremolo |
| Ethereal | Dreamlike, reverb-heavy voice |
| Monster | Deep, distorted, threatening voice |
| Ghost | Eerie, spectral voice with reverb |
| Podcast | Optimized for clarity and warmth (like professional audio) |
| Movie Trailer | Deep, compressed voice for dramatic announcements |

### Known Limitations

- **Auto-tune effect**: Requires librosa; falls back to a basic chorus effect when unavailable
- **Formant shifting**: Most effective with librosa installed
- **High-quality reverb**: Best results with scipy installed
- **Voice morphing**: Some combinations of effects may produce unexpected results
- **Processing time**: Some effects (especially reverb and auto-tune) can be CPU-intensive

## 💡 Tips and Tricks

1. **Memory Optimization**: Process shorter text segments when working with complex Voice Mod effects
2. **Voice Consistency**: Use the same voice and settings for multiple text segments to maintain consistency
3. **Custom Voices**: Try different blend ratios between voices to create unique combinations
4. **Pitch Effects**: Subtle pitch adjustments (+/- 1.0) often sound more natural than extreme values
5. **GPU Acceleration**: Use GPU for faster TTS processing, especially with longer texts
6. **Fallback Quality**: Install resampy and librosa for the best audio quality in Voice Mod effects

## 🔍 Troubleshooting

- **GPU Issues**: If you encounter GPU-related errors, try switching to CPU mode by unchecking the "use_gpu" option
- **Memory Errors**: If you run into memory issues, try processing shorter text segments
- **Audio Distortion**: For distorted output, try reducing effect intensities or disabling some effect groups
- **Missing Dependencies**: Check the console for warnings about missing libraries (librosa, resampy, etc.)
- **Model Load Errors**: Ensure the model files are correctly installed in the proper directory
- **"Defaulting repo_id" Warning**: This is normal on first run and doesn't affect functionality
- **Directory Naming Issues**: If you see import errors, ensure your node is installed in a directory named `ComfyUI-Geeky-Kokoro-TTS`

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
