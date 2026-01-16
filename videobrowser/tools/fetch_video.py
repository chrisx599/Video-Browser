from langchain_community.document_loaders import YoutubeLoader, YoutubeAudioLoader, GoogleApiYoutubeLoader, BiliBiliLoader
from langchain_core.documents import Document
import sys
import os
import glob
import requests
import yt_dlp
import shutil
from pytubefix import YouTube
from openai import OpenAI
from videobrowser.config import get_config
from videobrowser.utils.parser import clean_vtt_text, extract_youtube_id
from videobrowser.utils.cache import cache_manager

# Patch pytube if needed (though usually importing pytubefix directly is better)
try:
    import pytubefix
    sys.modules["pytube"] = pytubefix
except ImportError:
    pass

def fetch_with_oxylabs(video_url: str) -> list[Document]:
    """
    Fetches the transcript using Oxylabs Realtime API.
    """
    config = get_config()
    username = config.transcript.oxylabs_username or os.getenv("OXYLABS_USERNAME")
    password = config.transcript.oxylabs_password or os.getenv("OXYLABS_PASSWORD")

    if not username or not password:
        print("⚠️ Oxylabs credentials missing. Please set OXYLABS_USERNAME and OXYLABS_PASSWORD.")
        return []
    
    video_id = extract_youtube_id(video_url)
    payload = {
        'source': 'youtube_transcript',
        'query': video_id,
        'context': [
            {'key': 'language_code', 'value': 'en'},
            {'key': 'transcript_origin', 'value': 'auto_generated'}
        ]
    }

    try:
        response = requests.post(
            'https://realtime.oxylabs.io/v1/queries',
            auth=(username, password),
            json=payload,
            timeout=60 # Add timeout for safety
        )
        response.raise_for_status()
        data = response.json()
        
        results = data.get('results', [])
        if not results:
             print(f"⚠️ Oxylabs returned no results for {video_url}")
             return []

        # Parse nested structure
        # results -> [0] -> content -> [list of segments]
        # segment -> transcriptSegmentRenderer -> snippet -> runs -> [0] -> text
        
        content = results[0].get('content', [])
        transcript_parts = []
        
        if isinstance(content, list):
            for item in content:
                # Handle the specific structure provided by user
                renderer = item.get('transcriptSegmentRenderer')
                if renderer:
                    snippet = renderer.get('snippet', None)
                    if snippet:
                        runs = snippet.get('runs', [])
                        for run in runs:
                            text = run.get('text')
                            if text:
                                transcript_parts.append(text)
        
        transcript_text = " ".join(transcript_parts)
        
        if not transcript_text:
             print(f"⚠️ Could not parse transcript text from Oxylabs response for {video_id}")
             transcript_text = "" # Explicitly set to empty to trigger subtitle fallback
        else:
             return [Document(
                page_content=transcript_text,
                metadata={"source": video_url, "provider": "oxylabs"}
            )]

    except Exception as e:
        print(f"⚠️ Oxylabs fetch failed for {video_url}: {e}")
        transcript_text = "" # Ensure transcript_text is empty to trigger subtitle fallback
    
    # If transcript is empty, try to get subtitles
    if not transcript_text:
        print(f"🔄 No transcript found, attempting to fetch subtitles for {video_id}...")
        subtitle_payload = {
            'source': 'youtube_subtitles',
            'query': video_id,
            'context': [
                {'key': 'language_code', 'value': 'en'},
                {'key': 'subtitle_origin', 'value': 'auto_generated'}
            ]
        }
        try:
            subtitle_response = requests.post(
                'https://realtime.oxylabs.io/v1/queries',
                auth=(username, password),
                json=subtitle_payload,
                timeout=60
            )
            subtitle_response.raise_for_status()
            subtitle_data = subtitle_response.json()

            subtitle_results = subtitle_data.get('results', [])
            if not subtitle_results:
                print(f"⚠️ Oxylabs returned no subtitle results for {video_url}")
                return []
            
            # Parse subtitle structure
            # results -> [0] -> content -> auto_generated -> <language_code> -> events
            # event -> segs -> [0] -> utf8
            
            content_block = subtitle_results[0].get('content', {})
            auto_generated = content_block.get('auto_generated', {})
            english_subtitles = auto_generated.get('en', {}) # Assuming 'en' for now
            events = english_subtitles.get('events', [])
            
            subtitle_parts = []
            for event in events:
                segs = event.get('segs', [])
                for seg in segs:
                    text = seg.get('utf8')
                    if text:
                        subtitle_parts.append(text)
            
            transcript_text = " ".join(subtitle_parts)

            if not transcript_text:
                print(f"⚠️ Could not parse subtitle text from Oxylabs response for {video_id}")
                # print(f"DEBUG subtitle content sample: {subtitle_results}") # Debugging aid
                return []
            
            print(f"✅ Successfully fetched subtitles for {video_id}")

        except Exception as e:
            print(f"⚠️ Oxylabs subtitle fetch failed for {video_url}: {e}")
            return []

    if transcript_text:
        return [Document(
            page_content=transcript_text,
            metadata={"source": video_url, "provider": "oxylabs"}
        )]
    else:
        return []

def fetch_with_ytdlp(video_url: str) -> list[Document]:
    """
    Fetches transcript using yt-dlp (downloading subtitles).
    """
    import tempfile
    
    # Create a safe temp directory for this download
    video_id = extract_youtube_id(video_url)
    temp_dir = f"data/temp/subs/{video_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'outtmpl': f'{temp_dir}/%(id)s',
        'quiet': True,
        'no_warnings': True,
    }
    
    if os.path.exists("data/cookies.txt"):
        ydl_opts['cookiefile'] = "data/cookies.txt"
        print("🍪 Using cookies.txt for yt-dlp...")

    try:
        print(f"🔍 Fetching transcript for {video_url} using yt-dlp...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Check if live before downloading
            info = ydl.extract_info(video_url, download=False)
            if info.get('is_live'):
                print(f"⚠️ Skipping live video: {video_url}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return []
                
            ydl.download([video_url])
        
        # Find the vtt file
        vtt_files = glob.glob(f"{temp_dir}/*.vtt")
        if not vtt_files:
            print(f"⚠️ yt-dlp: No subtitles found for {video_url}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return []
            
        vtt_path = vtt_files[0]
        
        # Read raw content
        with open(vtt_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
            
        # Clean using utility function
        transcript_text = clean_vtt_text(raw_content)
        
        # Cleanup temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if not transcript_text:
             return []

        return [Document(
            page_content=transcript_text,
            metadata={"source": video_url, "provider": "ytdlp"}
        )]

    except Exception as e:
        print(f"⚠️ yt-dlp fetch failed for {video_url}: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return []

def fetch_with_whisper(video_url: str) -> list[Document]:
    """
    Fetches transcript using yt-dlp to download audio and OpenAI Whisper to transcribe.
    """
    config = get_config()
    
    # 1. Check Audio/Video Cache
    is_video_source = False
    if config.cache.enabled and cache_manager.has_audio(video_url):
        audio_path = cache_manager.get_audio_path(video_url)
        print(f"📦 Cache hit for audio file: {audio_path}, using for transcription...")
        use_temp = False
    elif config.cache.enabled and cache_manager.has_video(video_url):
        audio_path = cache_manager.get_video_path(video_url)
        print(f"📦 Cache hit for video file: {audio_path}, using for transcription...")
        is_video_source = True
        use_temp = False
        
    # 2. Download Audio
    else:
        video_id = extract_youtube_id(video_url)
        downloader = config.watcher.video_downloader
        
        # Paths
        if config.cache.enabled:
            audio_dir = str(cache_manager.audio_dir)
            use_temp = False
        else:
            audio_dir = f"data/temp/audio/{video_id}"
            os.makedirs(audio_dir, exist_ok=True)
            use_temp = True
            
        audio_path = os.path.join(audio_dir, f"{video_id}.mp3")
        downloaded_via_pytube = False

        # Attempt Pytubefix if configured
        if downloader == "pytubefix":
            try:
                print(f"🔍 Fetching audio for {video_url} using pytubefix...")
                yt = YouTube(video_url, use_oauth=True, allow_oauth_cache=True)
                # Filter for audio
                stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
                if stream:
                    # Download raw
                    raw_name = f"{video_id}_raw"
                    # pytube appends extension automatically, but we can't easily predict it (mp4/webm)
                    # so we let it download and find it
                    out_file = stream.download(output_path=audio_dir, filename=raw_name)
                    
                    # Convert to MP3 16k mono for Whisper
                    import subprocess
                    cmd = [
                        "ffmpeg", "-i", out_file, 
                        "-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k", 
                        "-y", audio_path
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if os.path.exists(audio_path):
                        downloaded_via_pytube = True
                        # Clean raw file
                        if os.path.exists(out_file) and out_file != audio_path:
                            os.remove(out_file)
                            
            except Exception as e:
                print(f"⚠️ pytubefix audio fetch failed: {e}. Falling back to yt-dlp...")

        # Fallback to yt-dlp if pytube failed or wasn't used
        if not downloaded_via_pytube and not os.path.exists(audio_path):
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '32',
                }],
                'outtmpl': f'{audio_dir}/{video_id}.%(ext)s' if config.cache.enabled else f'{audio_dir}/%(id)s.%(ext)s',
                'quiet': True,
                'no_warnings': True,
            }
            
            if os.path.exists("data/cookies.txt"):
                ydl_opts['cookiefile'] = "data/cookies.txt"
                print("🍪 Using cookies.txt for yt-dlp...")
            
            try:
                print(f"🔍 Fetching audio for {video_url} using yt-dlp...")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    if info.get('is_live'):
                        print(f"⚠️ Skipping live video for Whisper: {video_url}")
                        if use_temp: shutil.rmtree(audio_dir, ignore_errors=True)
                        return []
                    ydl.download([video_url])
            except Exception as e:
                print(f"⚠️ Whisper download failed (yt-dlp) for {video_url}: {e}")
                if use_temp: shutil.rmtree(audio_dir, ignore_errors=True)
                return []
                
        # Final Verification
        if not config.cache.enabled and not os.path.exists(audio_path):
             # Find it if name differed (yt-dlp dynamic ext)
             found = glob.glob(f"{audio_dir}/*.mp3")
             if found:
                 audio_path = found[0]
             else:
                 print(f"⚠️ No audio file found for {video_url}")
                 if use_temp: shutil.rmtree(audio_dir, ignore_errors=True)
                 return []
        
        if not os.path.exists(audio_path):
             print(f"⚠️ Audio path valid but file missing: {audio_path}")
             return []

    try:
        # Transcribe with Whisper
        api_key = config.llm.default.api_key
        
        if not api_key:
             print("⚠️ OpenAI API Key missing. Cannot use Whisper.")
             return []

        client = OpenAI(api_key=api_key)
        
        processing_path = audio_path
        is_temp_conversion = False
        
        # Always extract audio if source is a video file
        if is_video_source:
            # Determine target path for extracted audio
            if config.cache.enabled:
                target_audio_path = cache_manager.get_audio_storage_path(video_url, ext="mp3")
                
                # Check if already exists
                if os.path.exists(target_audio_path):
                    print(f"📦 Cache hit for extracted audio: {target_audio_path}")
                    processing_path = target_audio_path
                    is_temp_conversion = False
                else:
                    print(f"🔄 Extracting audio from cached video to: {target_audio_path}...")
                    try:
                        import subprocess
                        # Use ffmpeg to extract
                        cmd = [
                            "ffmpeg", "-i", audio_path, 
                            "-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k", 
                            "-y", target_audio_path
                        ]
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        if os.path.exists(target_audio_path):
                            processing_path = target_audio_path
                            print(f"✅ Audio extracted and cached.")
                        else:
                            print("⚠️ FFmpeg failed to create output file.")
                    except Exception as fe:
                        print(f"⚠️ FFmpeg processing failed: {fe}")
            else:
                # Use temp path
                import uuid
                import subprocess
                
                temp_extract_dir = "data/temp/whisper_extract"
                os.makedirs(temp_extract_dir, exist_ok=True)
                target_audio_path = os.path.join(temp_extract_dir, f"{uuid.uuid4()}.mp3")
                
                print(f"🔄 Extracting audio from video to temp: {target_audio_path}...")
                try:
                    cmd = [
                        "ffmpeg", "-i", audio_path, 
                        "-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k", 
                        "-y", target_audio_path
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if os.path.exists(target_audio_path):
                        processing_path = target_audio_path
                        is_temp_conversion = True
                        print(f"✅ Audio extracted to temp.")
                    else:
                        print("⚠️ FFmpeg failed to create output file.")
                except Exception as fe:
                    print(f"⚠️ FFmpeg processing failed: {fe}")

        with open(processing_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="verbose_json"
            )
            
        # Extract segments with timestamps
        segments = []
        duration = 0.0
        
        # The response object has a 'segments' attribute which is a list of objects
        if hasattr(response, 'segments'):
            duration = getattr(response, "duration", 0.0)
            for seg in response.segments:
                # Access attributes directly as per OpenAI python lib
                segments.append({
                    "start": getattr(seg, "start", 0.0),
                    "end": getattr(seg, "end", 0.0),
                    "text": getattr(seg, "text", "").strip()
                })
        # If accessing as dict (just in case it changes or is different version)
        elif isinstance(response, dict) and "segments" in response:
             duration = response.get("duration", 0.0)
             for seg in response["segments"]:
                segments.append({
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text", "").strip()
                })
        
        transcript_text = response.text
        
        # Cleanup temp conversion file
        if is_temp_conversion and os.path.exists(processing_path):
            os.remove(processing_path)
            # Try to remove dir if empty
            try:
                os.rmdir(os.path.dirname(processing_path))
            except:
                pass
        
        # Cleanup if temp download (video file)
        if use_temp:
            shutil.rmtree(os.path.dirname(audio_path), ignore_errors=True)
        
        if not transcript_text:
             return []
             
        return [Document(
            page_content=transcript_text,
            metadata={
                "source": video_url, 
                "provider": "whisper",
                "audio_duration": duration,
                "segments": segments
            }
        )]
        
    except Exception as e:
        print(f"⚠️ Whisper transcription failed for {video_url}: {e}")
        return []

def fetch_transcript_with_timestamps(video_url: str) -> list[dict]:
    """
    Fetches transcript with segment-level timestamps.
    Returns: list[dict] where each dict is {"start": float, "end": float, "text": str}
    """
    config = get_config()
    
    # 1. Try Cache
    if config.cache.enabled:
        cached_segments = cache_manager.get_transcript_with_timestamps(video_url)
        if cached_segments:
            print(f"📦 Cache hit for transcript with timestamps: {video_url}")
            return cached_segments
    
    docs = []
    if config.transcript.provider == "whisper":
        docs = fetch_with_whisper(video_url)
    
    # Add other providers if they support timestamps later
    
    if docs and "segments" in docs[0].metadata:
        segments = docs[0].metadata["segments"]
        
        # 2. Save to Cache
        if config.cache.enabled:
            cache_manager.save_transcript_with_timestamps(video_url, segments)
            
        return segments
    
    return []

def fetch_youtube_video_transcript(video_url: str) -> list:
    """
    Fetches the transcript (subtitles) of a YouTube video.
    Returns a list of LangChain Document objects.
    """
    config = get_config()
    
    # 1. Try Cache
    if config.cache.enabled:
        cached_text = cache_manager.get_transcript(video_url)
        if cached_text:
            print(f"📦 Cache hit for transcript: {video_url}")
            return [Document(page_content=cached_text, metadata={"source": video_url, "provider": "cache"})]

    # 2. Fetch from Provider
    docs = []
    if config.transcript.provider == "oxylabs":
        print(f"🔍 Fetching transcript for {video_url} using Oxylabs...")
        docs = fetch_with_oxylabs(video_url)
    elif config.transcript.provider == "ytdlp":
        docs = fetch_with_ytdlp(video_url)
    elif config.transcript.provider == "whisper":
        docs = fetch_with_whisper(video_url)
    else:
        # Default to local
        try:
            loader = YoutubeLoader.from_youtube_url(
                video_url, 
                add_video_info=True,
            )
            docs = loader.load()
        except Exception as e:
            print(f"⚠️ Transcript fetch failed for {video_url}: {e}")
            docs = []

    # 3. Save to Cache
    if config.cache.enabled and docs:
        cache_manager.save_transcript(video_url, docs[0].page_content)

    return docs


def download_video_file(video_url: str) -> str:
    """
    Downloads the video file (low resolution for efficiency) using either yt-dlp or pytubefix.
    Returns the absolute path to the downloaded file.
    """
    config = get_config()
    
    # 1. Check Cache
    if config.cache.enabled and cache_manager.has_video(video_url):
        path = cache_manager.get_video_path(video_url)
        print(f"📦 Cache hit for video file: {path}")
        return path

    # 2. Determine output directory
    if config.cache.enabled:
        output_dir = str(cache_manager.video_dir)
    else:
        output_dir = "data/temp/video"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    video_id = extract_youtube_id(video_url)

    # 3. Choose Downloader
    downloader = config.watcher.video_downloader
    
    if downloader == "ytdlp":
        out_tmpl = os.path.join(output_dir, f"{video_id}.%(ext)s")
        ydl_opts = {
            'format': 'best[height<=360][ext=mp4]/best[height<=360]/best',
            'outtmpl': out_tmpl,
            'quiet': True,
            'no_warnings': True,
        }

        # Add cookie file if it exists
        cookie_file = "data/cookies.txt"
        if os.path.exists(cookie_file):
            ydl_opts['cookiefile'] = cookie_file
            print(f"🍪 Using cookies for video download from {cookie_file}")

        try:
            print(f"⬇️ Downloading video '{video_id}' using yt-dlp...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Check if live before downloading
                info = ydl.extract_info(video_url, download=False)
                if info.get('is_live'):
                    print(f"⚠️ Skipping live video for download: {video_url}")
                    return None
                    
                info = ydl.extract_info(video_url, download=True)
                file_path = ydl.prepare_filename(info)
                
            if os.path.exists(file_path):
                print(f"✅ Video downloaded to: {file_path}")
                return file_path
        except Exception as e:
            print(f"❌ yt-dlp download failed: {e}")
            
    else: # Default to pytubefix or if explicitly chosen
        try:
            yt = YouTube(video_url, use_oauth=True, allow_oauth_cache=True)
            
            # Check if live
            if yt.vid_info.get('playabilityStatus', {}).get('liveStreamability'):
                print(f"⚠️ Skipping live video for download (pytubefix): {video_url}")
                return None
                
            # Prioritize 360p mp4 for speed and compatibility, fallback to lowest resolution
            stream = yt.streams.filter(res="360p", file_extension="mp4", progressive=True).first()
            if not stream:
                stream = yt.streams.filter(file_extension="mp4", progressive=True).order_by("resolution").asc().first()
                
            if not stream:
                raise ValueError("No suitable MP4 stream found.")
                
            print(f"⬇️ Downloading '{yt.title}' ({stream.resolution}) using pytubefix...")
            
            # If using cache, force ID as filename to match cache logic
            if config.cache.enabled:
                 filename = f"{video_id}.mp4"
            else:
                 filename = stream.default_filename
                 filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ' or c=='.']).rstrip()
                 
            file_path = stream.download(output_path=output_dir, filename=filename)
            return file_path
            
        except Exception as e:
            print(f"❌ pytubefix download failed: {e}")
            
    return None

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=zd_57PFJkNM" # Example

    # Test Transcript
    # docs = fetch_with_ytdlp(video_url)
    docs = fetch_youtube_video_transcript(video_url)
    if docs:
        print(f"✅ Transcript found ({len(docs[0].page_content)} chars)")
        print(docs[0].page_content[:500] + "...\n")  # Print first 500 chars
    else:
        print("❌ No transcript found.")
    
    # Test Download
    # path = download_video_file(video_url)
    # if path:
    #     print(f"✅ Video downloaded to: {path}")
