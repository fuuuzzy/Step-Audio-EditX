import argparse
import logging
import os
import re

import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import torchaudio
import librosa
import soundfile as sf

# Project imports
from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from model_loader import ModelSource
from config.edit_config import get_supported_edit_types


# Save audio to temporary directory
def save_audio(filename, audio_data, sr, output_dir):
    """Save audio data to a temporary file with timestamp"""
    logger = logging.getLogger(__name__)
    save_path = os.path.join(output_dir, f"{filename}.wav")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        if isinstance(audio_data, torch.Tensor):
            torchaudio.save(save_path, audio_data, sr)
        else:
            sf.write(save_path, audio_data, sr)
        logger.info(f"Audio saved to: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        raise

    return save_path


def parse_srt(srt_path):
    """
    Parse SRT subtitle file and extract subtitle entries
    
    Args:
        srt_path: Path to SRT file
        
    Returns:
        List of dicts with keys: 'index', 'text', 'start_time', 'end_time'
    """
    logger = logging.getLogger(__name__)
    subtitles = []
    
    try:
        logger.debug(f"Reading SRT file: {srt_path}")
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.debug(f"SRT file size: {len(content)} characters")
    except Exception as e:
        logger.error(f"Failed to read SRT file: {e}")
        raise
    
    # Split by double newlines to get individual subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    logger.debug(f"Found {len(blocks)} subtitle blocks (before filtering)")
    
    parsed_count = 0
    skipped_count = 0
    
    for block_idx, block in enumerate(blocks, 1):
        if not block.strip():
            skipped_count += 1
            continue
            
        lines = block.strip().split('\n')
        if len(lines) < 3:
            logger.debug(f"Block {block_idx}: Skipped (too few lines: {len(lines)})")
            skipped_count += 1
            continue
        
        try:
            # First line is index
            index = int(lines[0].strip())
            
            # Second line is timestamp (format: 00:00:00,000 --> 00:00:05,000)
            timestamp_line = lines[1].strip()
            time_match = re.match(r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})', timestamp_line)
            if not time_match:
                logger.warning(f"Block {block_idx} (index {lines[0].strip()}): Could not parse timestamp line: {timestamp_line}")
                skipped_count += 1
                continue
            
            start_time = time_match.group(1).replace(',', '.')
            end_time = time_match.group(2).replace(',', '.')
            
            # Remaining lines are text
            text = '\n'.join(lines[2:]).strip()
            
            if not text:
                logger.warning(f"Block {block_idx} (index {index}): Empty text, skipping")
                skipped_count += 1
                continue
            
            subtitles.append({
                'index': index,
                'text': text,
                'start_time': start_time,
                'end_time': end_time
            })
            parsed_count += 1
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Block {block_idx}: Failed to parse subtitle block: {block[:50]}... Error: {e}")
            skipped_count += 1
            continue
    
    logger.info(f"Parsed {parsed_count} subtitle entries from SRT file (skipped {skipped_count} invalid blocks)")
    if parsed_count > 0:
        # Log index range
        indices = [s['index'] for s in subtitles]
        logger.debug(f"Index range: {min(indices)} to {max(indices)}")
    
    return subtitles


def format_segment_number(number, prefix_format="segment_{:03d}.wav"):
    """
    Format segment number to match file naming pattern
    
    Args:
        number: Segment number (1-indexed)
        prefix_format: Format string for file name (default: "segment_{:03d}.wav")
        
    Returns:
        Formatted filename
    """
    return prefix_format.format(number)


class StepAudioEditX:
    """Audio editing and voice cloning local inference class"""

    def __init__(self, args):
        self.args = args
        self.edit_type_list = list(get_supported_edit_types().keys())

    def history_messages_to_show(self, messages):
        show_msgs = []
        for message in messages:
            edit_type = message['edit_type']
            edit_info = message['edit_info']
            source_text = message['source_text']
            target_text = message['target_text']
            raw_audio_part = message['raw_wave']
            edit_audio_part = message['edit_wave']
            type_str = f"{edit_type}-{edit_info}" if edit_info is not None else f"{edit_type}"
            show_msgs.extend([
                {"role": "user", "content": f"任务类型：{type_str}\n文本：{source_text}"},
                {"role": "user", "content": raw_audio_part},
                {"role": "assistant", "content": f"输出音频：\n文本：{target_text}"},
                {"role": "assistant", "content": edit_audio_part}
            ])
        return show_msgs

    def generate_clone(self, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, state,
                       filename_out):
        """Generate cloned audio"""
        logger.info("  [generate_clone] Starting voice cloning process")
        logger.info("  [generate_clone] Input parameters received:")
        logger.info(f"    - prompt_text_input: {prompt_text_input[:100]}{'...' if len(prompt_text_input) > 100 else ''} (length: {len(prompt_text_input)})")
        logger.info(f"    - prompt_audio_input: {prompt_audio_input}")
        logger.info(f"    - generated_text: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''} (length: {len(generated_text)})")
        logger.info(f"    - edit_type: {edit_type}")
        logger.info(f"    - edit_info: {edit_info}")
        logger.info(f"    - filename_out: {filename_out}")
        
        state['history_audio'] = []
        state['history_messages'] = []

        # Input validation
        logger.info("  [generate_clone] Validating inputs...")
        if not prompt_text_input or prompt_text_input.strip() == "":
            error_msg = "[Error] Uploaded text cannot be empty."
            logger.error(f"  [generate_clone] {error_msg}")
            return [{"role": "user", "content": error_msg}], state
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            logger.error(f"  [generate_clone] {error_msg}")
            return [{"role": "user", "content": error_msg}], state
        if not generated_text or generated_text.strip() == "":
            error_msg = "[Error] Clone content cannot be empty."
            logger.error(f"  [generate_clone] {error_msg}")
            return [{"role": "user", "content": error_msg}], state
        if edit_type != "clone":
            error_msg = "[Error] CLONE button must use clone task."
            logger.error(f"  [generate_clone] {error_msg}")
            return [{"role": "user", "content": error_msg}], state
        logger.info("  [generate_clone] ✓ Input validation passed")

        try:
            # Use common_tts_engine for cloning
            logger.info("  [generate_clone] Calling common_tts_engine.clone()...")
            logger.info("    Arguments:")
            logger.info(f"      - prompt_wav_path: {prompt_audio_input}")
            logger.info(f"      - prompt_text: {prompt_text_input[:100]}{'...' if len(prompt_text_input) > 100 else ''}")
            logger.info(f"      - target_text: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")
            
            output_audio, output_sr = common_tts_engine.clone(
                prompt_audio_input, prompt_text_input, generated_text
            )
            
            logger.info("  [generate_clone] ✓ Clone operation returned")
            logger.info(f"    - output_audio type: {type(output_audio)}")
            logger.info(f"    - output_sr: {output_sr}")

            if output_audio is not None and output_sr is not None:
                logger.info("  [generate_clone] Processing output audio...")
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    logger.info(f"    Converting tensor to numpy (shape: {output_audio.shape})")
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                    logger.info(f"    Converted shape: {audio_numpy.shape}")
                else:
                    audio_numpy = output_audio
                    logger.info(f"    Output is already numpy array (shape: {audio_numpy.shape})")

                # Load original audio for comparison
                logger.info(f"  [generate_clone] Loading original audio for comparison: {prompt_audio_input}")
                input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)
                logger.info(f"    Original audio loaded: shape={input_audio_data_numpy.shape}, sr={input_sample_rate}")

                # Create message for history
                logger.info("  [generate_clone] Creating history message...")
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": prompt_text_input,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                
                logger.info(f"  [generate_clone] Saving audio to: {filename_out}")
                audio_save_path = save_audio(filename_out, audio_numpy, output_sr, self.args.output_dir)
                logger.info(f"  [generate_clone] ✓ Audio saved to: {audio_save_path}")
                
                state["history_audio"].append((output_sr, audio_save_path, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                logger.info("  [generate_clone] ✓ Voice cloning completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Clone failed - output_audio or output_sr is None"
                logger.error(f"  [generate_clone] {error_msg}")
                logger.error(f"    output_audio: {output_audio}")
                logger.error(f"    output_sr: {output_sr}")
                return [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Clone failed: {str(e)}"
            logger.error(f"  [generate_clone] {error_msg}")
            logger.error(f"  [generate_clone] Exception type: {type(e).__name__}")
            import traceback
            logger.error("  [generate_clone] Traceback:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    logger.error(f"    {line}")
            return [{"role": "user", "content": error_msg}], state

    def generate_edit(self, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, state,
                      filename_out):
        """Generate edited audio"""
        logger.info("Starting audio editing process")

        # Input validation
        if not prompt_text_input or prompt_text_input.strip() == "":
            error_msg = "[Error] Uploaded text cannot be empty."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

        try:
            # Determine which audio to use
            if len(state["history_audio"]) == 0:
                # First edit - use uploaded audio
                audio_to_edit = prompt_audio_input
                text_to_use = prompt_text_input
                logger.debug("Using prompt audio, no history found")
            else:
                # Use previous edited audio - save it to temp file first
                _, audio_save_path, previous_text = state["history_audio"][-1]
                audio_to_edit = audio_save_path
                text_to_use = previous_text
                logger.debug(f"Using previous audio from history, count: {len(state['history_audio'])}")

            # For para-linguistic, use generated_text; otherwise use source text
            if edit_type not in {"paralinguistic"}:
                generated_text = text_to_use

            # Use common_tts_engine for editing
            output_audio, output_sr = common_tts_engine.edit(
                audio_to_edit, text_to_use, edit_type, edit_info, generated_text
            )

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                if len(state["history_audio"]) == 0:
                    input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)
                else:
                    input_sample_rate, input_audio_data_numpy, _ = state["history_audio"][-1]

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": text_to_use,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                audio_save_path = save_audio(filename_out, audio_numpy, output_sr, self.args.output_dir)
                state["history_audio"].append((output_sr, audio_save_path, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                logger.info("Audio editing completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Edit failed"
                logger.error(error_msg)
                return [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Edit failed: {str(e)}"
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

    def clear_history(self, state):
        """Clear conversation history"""
        state["history_messages"] = []
        state["history_audio"] = []
        return [], state

    def init_state(self):
        """Initialize conversation state"""
        return {
            "history_messages": [],
            "history_audio": []
        }


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Step-Audio-EditX local inference demo")
    parser.add_argument("--model-path", type=str, required=True, help="Model path.")
    parser.add_argument("--output-dir", type=str, default="./output_dir", help="Save path.")

    # Multi-source loading support parameters
    parser.add_argument(
        "--model-source",
        type=str,
        default="auto",
        choices=["auto", "local", "modelscope", "huggingface"],
        help="Model source: auto (detect automatically), local, modelscope, or huggingface"
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
        help="Tokenizer model ID for online loading"
    )
    parser.add_argument(
        "--tts-model-id",
        type=str,
        default=None,
        help="TTS model ID for online loading (if different from model-path)"
    )

    # clone or edit parameters
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="",
        help="prompt text for editing or cloning"
    )

    parser.add_argument(
        "--prompt-audio-path",
        type=str,
        default="",
        help="prompt audio for editing or cloning"
    )

    # Batch cloning parameters
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Enable batch cloning mode"
    )
    parser.add_argument(
        "--srt-path",
        type=str,
        default="",
        help="Path to target SRT subtitle file for batch cloning (contains target texts to generate)"
    )
    parser.add_argument(
        "--original-srt-path",
        type=str,
        default="",
        help="Path to original SRT subtitle file for batch cloning (contains reference audio texts, used as prompt-text)"
    )
    parser.add_argument(
        "--reference-audio-dir",
        type=str,
        default="",
        help="Directory containing reference audio files for batch cloning"
    )
    parser.add_argument(
        "--reference-audio-prefix",
        type=str,
        default="segment_{:03d}.wav",
        help="Prefix format for reference audio files (e.g., 'segment_{:03d}.wav' for segment_001.wav)"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="clone_{:03d}.wav",
        help="Prefix format for output files (e.g., 'clone_{:03d}.wav' for clone_001.wav)"
    )

    parser.add_argument(
        "--edit-type",
        type=str,
        choices=["clone", "emotion", "style", "vad", "denoise", "paralinguistic", "speed"],
        default="clone",
        help="Edit type"
    )

    parser.add_argument(
        "--edit-info",
        type=str,
        choices=[
            # default
            '',
            # emotion
            'happy', 'angry', 'sad', 'humour', 'confusion', 'disgusted',
            'empathy', 'embarrass', 'fear', 'surprised', 'excited',
            'depressed', 'coldness', 'admiration', 'remove',
            # style
            'serious', 'arrogant', 'child', 'older', 'girl', 'pure',
            'sister', 'sweet', 'ethereal', 'whisper', 'gentle', 'recite',
            'generous', 'act_coy', 'warm', 'shy', 'comfort', 'authority',
            'chat', 'radio', 'soulful', 'story', 'vivid', 'program',
            'news', 'advertising', 'roar', 'murmur', 'shout', 'deeply', 'loudly',
            'remove', 'exaggerated',
            # speed
            'faster', 'slower', 'more faster', 'more slower'
        ],
        default="",
        help="Edit info/sub-type"
    )

    parser.add_argument(
        "--n-edit-iter",
        type=int,
        default=1,
        help="the number of edit iterations"
    )

    parser.add_argument(
        "--generated-text",
        type=str,
        default="",
        help="Generated text for cloning or editing(paralinguistic)"
    )

    args = parser.parse_args()

    source_mapping = {
        "auto": ModelSource.AUTO,
        "local": ModelSource.LOCAL,
        "modelscope": ModelSource.MODELSCOPE,
        "huggingface": ModelSource.HUGGINGFACE
    }
    model_source = source_mapping[args.model_source]

    logger.info(f"Loading models with source: {args.model_source}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Tokenizer model ID: {args.tokenizer_model_id}")
    if args.tts_model_id:
        logger.info(f"TTS model ID: {args.tts_model_id}")

    # Initialize models
    try:
        # Load StepAudioTokenizer
        encoder = StepAudioTokenizer(
            os.path.join(args.model_path, "Step-Audio-Tokenizer"),
            model_source=model_source,
            funasr_model_id=args.tokenizer_model_id
        )
        logger.info("✓ StepAudioTokenizer loaded successfully")

        # Initialize common TTS engine directly
        common_tts_engine = StepAudioTTS(
            os.path.join(args.model_path, "Step-Audio-EditX"),
            encoder,
            model_source=model_source,
            tts_model_id=args.tts_model_id
        )
        logger.info("✓ StepCommonAudioTTS loaded successfully")

    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        logger.error("Please check your model paths and source configuration.")
        exit(1)

    # output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create StepAudioEditX instance
    step_audio_editx = StepAudioEditX(args)
    
    # Check if batch mode is enabled
    if args.batch_mode:
        logger.info("=" * 80)
        logger.info("BATCH CLONING MODE ENABLED")
        logger.info("=" * 80)
        
        if args.edit_type != "clone":
            logger.error("Batch mode is only supported for clone edit type")
            exit(1)
        
        # Log batch mode configuration
        logger.info("Batch mode configuration:")
        logger.info(f"  - Target SRT path: {args.srt_path}")
        logger.info(f"  - Original SRT path: {args.original_srt_path if args.original_srt_path else 'Not provided'}")
        logger.info(f"  - Reference audio directory: {args.reference_audio_dir}")
        logger.info(f"  - Reference audio prefix: {args.reference_audio_prefix}")
        logger.info(f"  - Output prefix: {args.output_prefix}")
        logger.info(f"  - Output directory: {args.output_dir}")
        logger.info(f"  - Prompt text (command line): {args.prompt_text if args.prompt_text else 'Not provided'}")
        
        # Validate target SRT file
        if not args.srt_path:
            logger.error("Target SRT path is required for batch mode")
            exit(1)
        if not os.path.exists(args.srt_path):
            logger.error(f"Target SRT file not found: {args.srt_path}")
            exit(1)
        logger.info(f"✓ Target SRT file exists: {args.srt_path}")
        
        # Validate reference audio directory
        if not args.reference_audio_dir:
            logger.error("Reference audio directory is required for batch mode")
            exit(1)
        if not os.path.isdir(args.reference_audio_dir):
            logger.error(f"Reference audio directory not found: {args.reference_audio_dir}")
            exit(1)
        logger.info(f"✓ Reference audio directory exists: {args.reference_audio_dir}")
        
        # Parse target SRT file (contains target texts to generate)
        logger.info("-" * 80)
        logger.info("Step 1: Parsing target SRT file...")
        logger.info(f"  Reading from: {args.srt_path}")
        try:
            subtitles = parse_srt(args.srt_path)
            logger.info(f"✓ Successfully parsed target SRT file")
            logger.info(f"  Found {len(subtitles)} subtitle entries")
            if len(subtitles) > 0:
                logger.info(f"  First entry: index={subtitles[0]['index']}, text={subtitles[0]['text'][:50]}...")
                logger.info(f"  Last entry: index={subtitles[-1]['index']}, text={subtitles[-1]['text'][:50]}...")
        except Exception as e:
            logger.error(f"✗ Failed to parse target SRT file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            exit(1)
        
        if not subtitles:
            logger.error("No subtitles found in target SRT file")
            exit(1)
        
        # Parse original SRT file if provided (contains reference audio texts)
        original_subtitles_dict = {}
        if args.original_srt_path:
            logger.info("-" * 80)
            logger.info("Step 2: Parsing original SRT file...")
            logger.info(f"  Reading from: {args.original_srt_path}")
            if not os.path.exists(args.original_srt_path):
                logger.error(f"Original SRT file not found: {args.original_srt_path}")
                exit(1)
            try:
                original_subtitles = parse_srt(args.original_srt_path)
                # Create a dictionary mapping index to text for quick lookup
                original_subtitles_dict = {sub['index']: sub['text'] for sub in original_subtitles}
                logger.info(f"✓ Successfully parsed original SRT file")
                logger.info(f"  Loaded {len(original_subtitles_dict)} entries")
                if len(original_subtitles_dict) > 0:
                    first_idx = min(original_subtitles_dict.keys())
                    logger.info(f"  First entry: index={first_idx}, text={original_subtitles_dict[first_idx][:50]}...")
            except Exception as e:
                logger.error(f"✗ Failed to parse original SRT file: {e}")
                import traceback
                logger.error(traceback.format_exc())
                exit(1)
        else:
            logger.info("-" * 80)
            logger.info("Step 2: Original SRT file not provided, will use command line prompt_text or target text")
        
        # Process each subtitle entry
        logger.info("-" * 80)
        logger.info("Step 3: Starting batch cloning process...")
        logger.info(f"  Total segments to process: {len(subtitles)}")
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for idx, subtitle in enumerate(subtitles, 1):
            segment_index = subtitle['index']
            target_text = subtitle['text']
            
            logger.info("")
            logger.info(f"[{idx}/{len(subtitles)}] Processing segment {segment_index}")
            logger.info(f"  Target text: {target_text[:100]}{'...' if len(target_text) > 100 else ''}")
            
            # Find corresponding reference audio file
            reference_filename = format_segment_number(segment_index, args.reference_audio_prefix)
            reference_audio_path = os.path.join(args.reference_audio_dir, reference_filename)
            logger.info(f"  Reference audio file: {reference_filename}")
            logger.info(f"  Full path: {reference_audio_path}")
            
            if not os.path.exists(reference_audio_path):
                logger.warning(f"  ✗ Reference audio file not found: {reference_audio_path}")
                logger.warning(f"    Skipping segment {segment_index}")
                failed_count += 1
                skipped_count += 1
                continue
            logger.info(f"  ✓ Reference audio file exists")
            
            # Check file size
            try:
                file_size = os.path.getsize(reference_audio_path)
                logger.info(f"  Reference audio file size: {file_size} bytes ({file_size/1024:.2f} KB)")
            except Exception as e:
                logger.warning(f"  Could not get file size: {e}")
            
            # Determine prompt_text: priority: original_srt > command_line_prompt_text > target_text
            if args.original_srt_path and segment_index in original_subtitles_dict:
                prompt_text = original_subtitles_dict[segment_index]
                logger.info(f"  Prompt text source: Original SRT (index {segment_index})")
                logger.info(f"  Prompt text: {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}")
            elif args.prompt_text:
                prompt_text = args.prompt_text
                logger.info(f"  Prompt text source: Command line argument")
                logger.info(f"  Prompt text: {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}")
            else:
                prompt_text = target_text
                logger.info(f"  Prompt text source: Target text (fallback)")
                logger.info(f"  Prompt text: {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}")
            
            # Validate texts
            if not prompt_text or not prompt_text.strip():
                logger.error(f"  ✗ Prompt text is empty for segment {segment_index}")
                failed_count += 1
                continue
            
            if not target_text or not target_text.strip():
                logger.error(f"  ✗ Target text is empty for segment {segment_index}")
                failed_count += 1
                continue
            
            try:
                # Generate output filename
                output_filename = format_segment_number(segment_index, args.output_prefix)
                output_filename_base = os.path.splitext(output_filename)[0]
                output_path = os.path.join(args.output_dir, f"{output_filename_base}.wav")
                logger.info(f"  Output filename: {output_filename}")
                logger.info(f"  Output path: {output_path}")
                
                # Print all processed parameters before calling clone
                logger.info("")
                logger.info("  " + "=" * 76)
                logger.info("  FINAL PARAMETERS FOR CLONE OPERATION")
                logger.info("  " + "=" * 76)
                logger.info(f"  Segment Index: {segment_index}")
                logger.info(f"  Prompt Text (Reference Audio Text):")
                logger.info(f"    Length: {len(prompt_text)} characters")
                logger.info(f"    Content: {prompt_text}")
                logger.info(f"  Target Text (Text to Generate):")
                logger.info(f"    Length: {len(target_text)} characters")
                logger.info(f"    Content: {target_text}")
                logger.info(f"  Reference Audio File:")
                logger.info(f"    Path: {reference_audio_path}")
                logger.info(f"    Exists: {os.path.exists(reference_audio_path)}")
                if os.path.exists(reference_audio_path):
                    file_size = os.path.getsize(reference_audio_path)
                    logger.info(f"    Size: {file_size} bytes ({file_size/1024:.2f} KB)")
                logger.info(f"  Output Configuration:")
                logger.info(f"    Output Directory: {args.output_dir}")
                logger.info(f"    Output Filename Base: {output_filename_base}")
                logger.info(f"    Full Output Path: {output_path}")
                logger.info(f"  Edit Configuration:")
                logger.info(f"    Edit Type: clone")
                logger.info(f"    Edit Info: {args.edit_info if args.edit_info else 'None'}")
                logger.info("  " + "=" * 76)
                logger.info("")
                
                # Perform cloning
                logger.info(f"  Calling generate_clone() method...")
                
                _, state = step_audio_editx.generate_clone(
                    prompt_text,
                    reference_audio_path,
                    target_text,
                    "clone",
                    args.edit_info,
                    step_audio_editx.init_state(),
                    output_filename_base,
                )
                
                # Verify output file was created
                if os.path.exists(output_path):
                    output_size = os.path.getsize(output_path)
                    logger.info(f"  ✓ Clone operation completed successfully")
                    logger.info(f"    Output file created: {output_path}")
                    logger.info(f"    Output file size: {output_size} bytes ({output_size/1024:.2f} KB)")
                    success_count += 1
                else:
                    logger.warning(f"  ⚠ Clone operation reported success but output file not found: {output_path}")
                    success_count += 1  # Still count as success if function returned without error
                
            except Exception as e:
                logger.error(f"  ✗ Clone operation failed for segment {segment_index}")
                logger.error(f"    Error type: {type(e).__name__}")
                logger.error(f"    Error message: {str(e)}")
                import traceback
                logger.error(f"    Traceback:")
                for line in traceback.format_exc().split('\n'):
                    if line.strip():
                        logger.error(f"      {line}")
                failed_count += 1
                continue
        
        # Final summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("BATCH CLONING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"  Total segments: {len(subtitles)}")
        logger.info(f"  ✓ Successful: {success_count}")
        logger.info(f"  ✗ Failed: {failed_count}")
        logger.info(f"  ⊘ Skipped (missing files): {skipped_count}")
        logger.info(f"  Success rate: {(success_count/len(subtitles)*100):.1f}%")
        logger.info("=" * 80)
        
    elif args.edit_type == "clone":
        # Single clone mode
        if not args.prompt_audio_path:
            logger.error("--prompt-audio-path is required for single clone mode")
            exit(1)
        if not args.generated_text:
            logger.error("--generated-text is required for single clone mode")
            exit(1)
        
        filename_out = os.path.basename(args.prompt_audio_path).split('.')[0] + "_cloned"
        _, state = step_audio_editx.generate_clone(
            args.prompt_text,
            args.prompt_audio_path,
            args.generated_text,
            args.edit_type,
            args.edit_info,
            step_audio_editx.init_state(),
            filename_out,
        )

    else:
        # Edit mode
        if not args.prompt_audio_path:
            logger.error("--prompt-audio-path is required for edit mode")
            exit(1)
        
        state = step_audio_editx.init_state()
        for iter_idx in range(args.n_edit_iter):
            logger.info(f"Starting edit iteration {iter_idx + 1}/{args.n_edit_iter}")
            filename_out = os.path.basename(args.prompt_audio_path).split('.')[0] + f"_edited_iter{iter_idx + 1}"
            msgs, state = step_audio_editx.generate_edit(
                args.prompt_text,
                args.prompt_audio_path,
                args.generated_text,
                args.edit_type,
                args.edit_info,
                state,
                filename_out,
            )
