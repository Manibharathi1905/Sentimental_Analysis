# utils/multimodal_generator.py
# Image and Video Generation for Responses

import os
import warnings
import shutil
import torch
import numpy as np
from config import config

# Optional heavy deps: diffusers, moviepy, pillow
try:
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
except Exception:
    StableDiffusionPipeline = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    import moviepy.editor as mp
except Exception:
    mp = None

class MultimodalGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Image pipeline (optional) - lazy-loaded to avoid heavy downloads at import
        self.image_pipe = None
        self._image_pipe_loading = False
        
        # Video: For simplicity, use text-to-image + animation; real project: integrate AnimateDiff
        # Here, simulate video by overlaying text on image clips
    
    def generate_image(self, prompt: str, emotion: str) -> str:
        """
        Generate empathetic image based on emotion.
        Returns path to saved image.
        """
        full_prompt = f"A compassionate, supportive image for someone feeling {emotion}, warm colors, serene landscape, empathetic vibe"

        # Ensure outputs dir
        os.makedirs("outputs", exist_ok=True)

        # If we have a diffusion pipeline, use it; otherwise, fallback to a simple placeholder image
        image_path = f"outputs/image_{emotion}_{torch.randint(0, 10000, (1,)).item()}.png"
        # Ensure pipeline available (may trigger download on first call)
        self._ensure_image_pipe()
        if self.image_pipe is not None:
            image = self.image_pipe(full_prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
            image.save(image_path)
            return image_path

        # Fallback: create a plain image with PIL if available
        if Image is not None:
            img = Image.new('RGB', (512, 512), color=(240, 240, 240))
            try:
                draw = ImageDraw.Draw(img)
                text = f"Supportive image for {emotion}"
                # Try to load a default font; if not available, PIL will use a basic font
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                draw.text((20, 240), text, fill=(30, 30, 30), font=font)
            except Exception:
                warnings.warn("Failed to draw text onto fallback image.")
            img.save(image_path)
            return image_path

        # As a last resort, return empty string so caller can skip image display
        return ""

    def _ensure_image_pipe(self):
        """Lazy-load the StableDiffusionPipeline on first use."""
        if self.image_pipe is not None or self._image_pipe_loading:
            return
        # If diffusers not installed, nothing to do
        if StableDiffusionPipeline is None:
            return

        # If config.image_model_name is falsy (empty string / None), treat as disabled
        if not getattr(config, "image_model_name", None):
            warnings.warn("Image model is disabled via config.image_model_name; using PIL fallback.")
            self.image_pipe = None
            return
        # Respect explicit enable flag in config to avoid accidental downloads
        if not getattr(config, "image_generation_enabled", False):
            warnings.warn("Image generation disabled via config.image_generation_enabled; using PIL fallback.")
            self.image_pipe = None
            return

        # Mark loading to avoid re-entrant calls
        self._image_pipe_loading = True
        try:
            self.image_pipe = StableDiffusionPipeline.from_pretrained(
                config.image_model_name,
                use_safetensors=True
            )
            try:
                self.image_pipe = self.image_pipe.to(self.device)
            except Exception:
                pass
        except Exception:
            self.image_pipe = None
        finally:
            self._image_pipe_loading = False

    def is_image_supported(self) -> bool:
        """Return True if either diffusers or Pillow is available for creating images."""
        return (StableDiffusionPipeline is not None) or (Image is not None)

    def is_video_supported(self) -> bool:
        """Return True if moviepy is available and there is a way to create frames (diffusers or Pillow)."""
        # moviepy requires ffmpeg to write videos. Check for ffmpeg executable.
        ffmpeg_ok = shutil.which("ffmpeg") is not None
        return (mp is not None) and ffmpeg_ok and ((StableDiffusionPipeline is not None) or (Image is not None))
    
    def generate_video(self, story_text: str, prompt: str) -> str:
        """
        Generate simple motivational video: Image from prompt + text overlay + fade.
        Returns path to saved video.
        """
        # If moviepy or image pipeline aren't available, skip generating video
        # Ensure image pipeline if needed
        self._ensure_image_pipe()
        if mp is None or (self.image_pipe is None and Image is None):
            warnings.warn("Video generation skipped: moviepy or image libraries not available.")
            return ""

        # Ensure outputs dir
        os.makedirs("outputs", exist_ok=True)

        # Generate base image
        if self.image_pipe is not None:
            base_img = self.image_pipe(prompt, num_inference_steps=20).images[0]
            # base_img might be PIL.Image already
            if Image is not None and not isinstance(base_img, Image.Image):
                try:
                    base_img = Image.fromarray(np.array(base_img))
                except Exception:
                    pass
        else:
            # Use Pillow fallback image
            base_img = Image.new('RGB', (1280, 720), color=(0, 0, 0)) if Image is not None else None

        if base_img is None:
            warnings.warn("Cannot create video: no base image available.")
            return ""

        # Draw text overlay onto the base image using PIL (avoids TextClip/ImageMagick)
        try:
            overlay = base_img.copy()
            draw = ImageDraw.Draw(overlay)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            # Simple text wrapping
            text = story_text
            max_width = overlay.width - 40
            lines = []
            if font is None:
                # naive split if no font metrics
                lines = [text[i:i+80] for i in range(0, len(text), 80)]
            else:
                words = text.split()
                cur_line = ""
                for word in words:
                    test_line = (cur_line + " " + word).strip()
                    try:
                        w, h = draw.textsize(test_line, font=font)
                    except Exception:
                        w = len(test_line) * 6
                    if w <= max_width:
                        cur_line = test_line
                    else:
                        lines.append(cur_line)
                        cur_line = word
                if cur_line:
                    lines.append(cur_line)

            # Draw semi-transparent box behind text
            try:
                box_height = 20 * max(1, len(lines)) + 20
                box_y = overlay.height - box_height - 20
                draw.rectangle([(20, box_y), (overlay.width - 20, overlay.height - 20)], fill=(0, 0, 0, 180))
            except Exception:
                pass

            # Draw lines
            y = overlay.height - (20 * len(lines) + 30)
            for ln in lines:
                draw.text((30, y), ln, fill=(255, 255, 255), font=font)
                y += 20

            # Convert to numpy array for moviepy
            frame = np.array(overlay)
        except Exception:
            warnings.warn("Failed to draw text overlay; using base image without overlay.")
            frame = np.array(base_img)

        # Create clip and write video (use non-verbose writing)
        img_clip = mp.ImageClip(frame).set_duration(30)
        video = mp.CompositeVideoClip([img_clip.set_audio(None)]).set_duration(30)
        video_path = f"outputs/video_motiv_{torch.randint(0, 10000, (1,)).item()}.mp4"
        try:
            video.write_videofile(video_path, fps=24, verbose=False, logger=None)
        except Exception as exc:
            warnings.warn(f"Video write failed: {exc}")
            return ""
        finally:
            try:
                video.close()
            except Exception:
                pass
        return video_path

# Global instance
multimodal_gen = MultimodalGenerator()
