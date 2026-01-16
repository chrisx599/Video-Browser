import base64
import numpy as np
import io
from PIL import Image
try:
    from decord import VideoReader, cpu
except ImportError:
    print("⚠️ Decord not installed. Vision features will be disabled.")
    VideoReader = None

def extract_frames_from_video(video_path: str, num_frames: int = 10):
    """
    Extracts frames from a video file using uniform sampling.
    Returns a list of base64-encoded image strings.
    """
    if VideoReader is None:
        return []

    try:
        # 1. Load video
        # ctx=cpu(0) ensures we use CPU. Use gpu(0) if CUDA is available/configured.
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        if total_frames == 0:
            return []

        # 2. Calculate indices for uniform sampling
        # We skip the very first and last frames to avoid black screens or intros/outros
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        # 3. Batch get frames (Decord is optimized for this)
        frames_batch = vr.get_batch(indices).asnumpy()
        
        base64_images = []
        
        # 4. Convert to Base64
        for frame_array in frames_batch:
            # frame_array is (Height, Width, 3) RGB
            img = Image.fromarray(frame_array)
            
            # Resize if too large (GPT-4o accepts up to 20MB, but smaller is faster/cheaper)
            # Resize to max dimension 512px helps token usage
            img.thumbnail((512, 512))
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=70)
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            base64_images.append(img_str)
            
        return base64_images

    except Exception as e:
        print(f"❌ [Vision Tool] Error extracting frames: {e}")
        return []



def extract_frames_with_timestamps(video_path: str, num_frames: int = 16):
    """
    Extracts frames with timestamps from a video file.
    Returns a list of dicts: {'timestamp': float, 'image': base64_str}
    """
    if VideoReader is None:
        return []

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        
        if total_frames == 0:
            return []

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames_batch = vr.get_batch(indices).asnumpy()
        
        results = []
        for i, frame_array in enumerate(frames_batch):
            idx = indices[i]
            timestamp = idx / fps
            
            img = Image.fromarray(frame_array)
            img.thumbnail((512, 512))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=70)
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            results.append({"timestamp": timestamp, "image": img_str})
            
        return results

    except Exception as e:
        print(f"❌ [JIT Vision] Error extracting frames: {e}")
        return []

def extract_frames_from_window(video_path: str, start_time: float, end_time: float, fps_sample: float = 1.0, max_frames: int = 32):
    """
    Extracts frames from a specific time window.
    Returns: list[dict] -> [{'timestamp': float, 'image': base64_str}, ...]
    """
    if VideoReader is None:
        return []

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        start_frame = max(0, start_frame)
        end_frame = min(total_frames - 1, end_frame)
        
        if end_frame <= start_frame:
            return []
            
        duration = end_time - start_time
        num_frames = int(duration * fps_sample)
        
        # Clamp
        if num_frames > max_frames:
            num_frames = max_frames
        if num_frames < 1:
            num_frames = 1
            
        indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        frames_batch = vr.get_batch(indices).asnumpy()
        
        results = []
        for i, frame_array in enumerate(frames_batch):
            idx = indices[i]
            timestamp = idx / fps
            
            img = Image.fromarray(frame_array)
            img.thumbnail((512, 512))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=70)
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            results.append({"timestamp": timestamp, "image": img_str})
            
        return results
        
    except Exception as e:
        print(f"❌ [Vision Tool] Error extracting window frames: {e}")
        return []