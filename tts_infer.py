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
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Failed to read SRT file: {e}")
        raise
    
    # Split by double newlines to get individual subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    
    for block in blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        try:
            # First line is index
            index = int(lines[0].strip())
            
            # Second line is timestamp (format: 00:00:00,000 --> 00:00:05,000)
            timestamp_line = lines[1].strip()
            time_match = re.match(r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})', timestamp_line)
            if not time_match:
                logger.warning(f"Could not parse timestamp line: {timestamp_line}")
                continue
            
            start_time = time_match.group(1).replace(',', '.')
            end_time = time_match.group(2).replace(',', '.')
            
            # Remaining lines are text
            text = '\n'.join(lines[2:]).strip()
            
            subtitles.append({
                'index': index,
                'text': text,
                'start_time': start_time,
                'end_time': end_time
            })
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse subtitle block: {block[:50]}... Error: {e}")
            continue
    
    logger.info(f"Parsed {len(subtitles)} subtitle entries from SRT file")
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
        logger.info("Starting voice cloning process")
        state['history_audio'] = []
        state['history_messages'] = []

        # Input validation
        if not prompt_text_input or prompt_text_input.strip() == "":
            error_msg = "[Error] Uploaded text cannot be empty."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if not generated_text or generated_text.strip() == "":
            error_msg = "[Error] Clone content cannot be empty."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if edit_type != "clone":
            error_msg = "[Error] CLONE button must use clone task."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

        try:
            # Use common_tts_engine for cloning
            output_audio, output_sr = common_tts_engine.clone(
                prompt_audio_input, prompt_text_input, generated_text
            )

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": prompt_text_input,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                audio_save_path = save_audio(filename_out, audio_numpy, output_sr, self.args.output_dir)
                state["history_audio"].append((output_sr, audio_save_path, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                logger.info("Voice cloning completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Clone failed"
                logger.error(error_msg)
                return [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Clone failed: {str(e)}"
            logger.error(error_msg)
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
        if args.edit_type != "clone":
            logger.error("Batch mode is only supported for clone edit type")
            exit(1)
        
        if not args.srt_path or not os.path.exists(args.srt_path):
            logger.error(f"SRT file not found: {args.srt_path}")
            exit(1)
        
        if not args.reference_audio_dir or not os.path.isdir(args.reference_audio_dir):
            logger.error(f"Reference audio directory not found: {args.reference_audio_dir}")
            exit(1)
        
        # Parse target SRT file (contains target texts to generate)
        logger.info("Parsing target SRT file...")
        subtitles = parse_srt(args.srt_path)
        
        if not subtitles:
            logger.error("No subtitles found in target SRT file")
            exit(1)
        
        # Parse original SRT file if provided (contains reference audio texts)
        original_subtitles_dict = {}
        if args.original_srt_path:
            if not os.path.exists(args.original_srt_path):
                logger.error(f"Original SRT file not found: {args.original_srt_path}")
                exit(1)
            logger.info("Parsing original SRT file...")
            original_subtitles = parse_srt(args.original_srt_path)
            # Create a dictionary mapping index to text for quick lookup
            original_subtitles_dict = {sub['index']: sub['text'] for sub in original_subtitles}
            logger.info(f"Loaded {len(original_subtitles_dict)} entries from original SRT file")
        
        # Process each subtitle entry
        logger.info(f"Starting batch cloning for {len(subtitles)} segments...")
        success_count = 0
        failed_count = 0
        
        for subtitle in subtitles:
            segment_index = subtitle['index']
            target_text = subtitle['text']
            
            # Find corresponding reference audio file
            reference_filename = format_segment_number(segment_index, args.reference_audio_prefix)
            reference_audio_path = os.path.join(args.reference_audio_dir, reference_filename)
            
            if not os.path.exists(reference_audio_path):
                logger.warning(f"Reference audio file not found: {reference_audio_path}, skipping segment {segment_index}")
                failed_count += 1
                continue
            
            logger.info(f"Processing segment {segment_index}: {target_text[:50]}...")
            
            # Determine prompt_text: priority: original_srt > command_line_prompt_text > target_text
            if args.original_srt_path and segment_index in original_subtitles_dict:
                prompt_text = original_subtitles_dict[segment_index]
                logger.debug(f"Using prompt_text from original SRT for segment {segment_index}")
            elif args.prompt_text:
                prompt_text = args.prompt_text
                logger.debug(f"Using prompt_text from command line argument for segment {segment_index}")
            else:
                prompt_text = target_text
                logger.debug(f"Using target_text as prompt_text for segment {segment_index}")
            
            try:
                # Generate output filename
                output_filename = format_segment_number(segment_index, args.output_prefix)
                output_filename_base = os.path.splitext(output_filename)[0]
                
                # Perform cloning
                _, state = step_audio_editx.generate_clone(
                    prompt_text,
                    reference_audio_path,
                    target_text,
                    "clone",
                    args.edit_info,
                    step_audio_editx.init_state(),
                    output_filename_base,
                )
                
                success_count += 1
                logger.info(f"✓ Successfully cloned segment {segment_index}")
                
            except Exception as e:
                logger.error(f"✗ Failed to clone segment {segment_index}: {e}")
                failed_count += 1
                continue
        
        logger.info(f"Batch cloning completed: {success_count} succeeded, {failed_count} failed")
        
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
