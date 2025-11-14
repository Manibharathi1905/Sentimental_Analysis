from PIL import Image, ImageDraw, ImageFont
import io
from typing import Optional

def generate_emotion_thumbnail(emotion: Optional[str] = None, size=(480, 270)) -> bytes:
    """Create a thumbnail with an emotion label over a colored background."""
    colors = {
        'sadness': (70, 130, 180),  # Steel blue
        'anxiety': (123, 104, 238),  # Medium slate blue
        'anger': (220, 20, 60),      # Crimson
        'fear': (139, 69, 19),       # Saddle brown
        'neutral': (100, 100, 100),  # Gray
        'joy': (60, 179, 113)       # Medium sea green
    }
    bg_color = colors.get(emotion.lower() if emotion else 'neutral', (80, 80, 80))
    
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Load font with fallback
    try:
        font = ImageFont.truetype('arial.ttf', 32)
    except Exception:
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', 28)
        except Exception:
            font = ImageFont.load_default()

    # Emotion label
    label = (emotion or 'Motivation').title()
    bbox = draw.textbbox((0, 0), label, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = (size[0] - w) // 2, size[1] - h - 30
    
    # Draw semi-transparent background for text
    draw.rectangle([(0, y - 10), (size[0], y + h + 10)], fill=(0, 0, 0, 128))
    draw.text((x, y), label, font=font, fill=(255, 255, 255))

    # Save to bytes
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()

def generate_youtube_logo_thumbnail(emotion: Optional[str] = None, size=(480, 270)) -> bytes:
    """Generate a YouTube-style thumbnail with a play button and emotion label."""
    img = Image.new('RGB', size, color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Draw red YouTube-style bar
    bar_height = int(size[1] * 0.3)
    bar_y = size[1] - bar_height
    draw.rectangle([(0, bar_y), (size[0], size[1])], fill=(230, 33, 23))

    # Draw play button
    play_size = int(bar_height * 0.5)
    cx, cy = int(size[0] * 0.15), bar_y + bar_height // 2
    triangle = [
        (cx - play_size // 2, cy - play_size // 2),
        (cx - play_size // 2, cy + play_size // 2),
        (cx + play_size // 2, cy)
    ]
    draw.polygon(triangle, fill=(255, 255, 255))

    # Load font with fallback
    try:
        font = ImageFont.truetype('arial.ttf', 28)
    except Exception:
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', 24)
        except Exception:
            font = ImageFont.load_default()

    # Emotion label
    label = (emotion or 'Motivation').title()
    bbox = draw.textbbox((0, 0), label, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = int(size[0] * 0.25), bar_y + (bar_height - h) // 2
    draw.text((x, y), label, font=font, fill=(255, 255, 255))

    # Save to bytes
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()