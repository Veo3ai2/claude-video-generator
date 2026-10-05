from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import anthropic
import numpy as np
from dotenv import load_dotenv
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips

load_dotenv()


class ClaudeVideoGenerator:
    def __init__(self) -> None:
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.width = int(os.getenv("VIDEO_WIDTH", "1280"))
        self.height = int(os.getenv("VIDEO_HEIGHT", "720"))
        self.fps = int(os.getenv("VIDEO_FPS", "30"))
        self.use_voiceover = os.getenv("USE_VOICEOVER", "true").lower() == "true"
        self.scenes_count = int(os.getenv("SCENES_COUNT", "5"))
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
        self.tone = os.getenv("TONE", "معلوماتية")
        self.theme = os.getenv("THEME", "dark")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None

    def _safe_font(self) -> str | None:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _extract_json(self, text: str) -> Dict[str, Any]:
        clean = text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
            clean = re.sub(r"\s*```$", "", clean, flags=re.IGNORECASE)
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            clean = match.group(0)
        return json.loads(clean)

    def _default_storyboard(self, topic: str) -> Dict[str, Any]:
        return {
            "title": topic,
            "scenes": [
                {"title": "مقدمة", "text": f"لنبدأ في فهم {topic} بشكل بسيط ومباشر.", "duration": 4},
                {"title": "الفكرة الأساسية", "text": "الفكرة الرئيسية هي أن هذا الموضوع يساعدنا في اتخاذ قرارات أفضل.", "duration": 4},
                {"title": "الأسباب", "text": "الأسباب تكمن في السرعة، الفعالية، والقدرة على التكيف مع التغيرات.", "duration": 4},
                {"title": "التطبيق", "text": "يمكن تطبيق هذه الفكرة في الحياة اليومية والعمل والتعليم بشكل عملي.", "duration": 4},
                {"title": "الخاتمة", "text": "باختصار، هذا الموضوع يمنحنا قيمة كبيرة إذا استخدمناها بذكاء.", "duration": 4},
            ],
        }

    def generate_storyboard(self, topic: str, scenes_count: int | None = None) -> Dict[str, Any]:
        count = scenes_count or self.scenes_count
        if not self.client:
            return self._default_storyboard(topic)

        prompt = f"""
        أنت مساعد لإنشاء سيناريوهات فيديو احترافية باللغة العربية.

        الموضوع: {topic}
        عدد المشاهد: {count}
        النبرة: {self.tone}

        المطلوب: أعد جواباً بصيغة JSON فقط، بدون شرح، بهذا الشكل الدقيق:
        {
          "title": "عنوان الفيديو",
          "scenes": [
            {"title": "عنوان المشهد", "text": "نص المشهد من 1 إلى 3 جمل قصيرة وواضحة.", "duration": 4},
            {"title": "عنوان المشهد", "text": "نص المشهد...", "duration": 4}
          ]
        }

        قواعد مهمة:
        - كل نص مشهد 1 إلى 3 جمل فقط.
        - كل مدة من 3 إلى 6 ثوانٍ.
        - كل مشهد يضيف قيمة جديدة لكل من الفيديو.
        - استخدم نصوص عربية واضحة ومباشرة.
        - لا تضع أي علامات إضافية خارج JSON.
        """

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text
        data = self._extract_json(raw_text)

        if not isinstance(data, dict) or "scenes" not in data:
            return self._default_storyboard(topic)

        scenes = data.get("scenes") or []
        if not scenes:
            return self._default_storyboard(topic)

        for scene in scenes:
            scene["title"] = scene.get("title", "مشهد")
            scene["text"] = scene.get("text", "")
            scene["duration"] = max(3, min(6, int(scene.get("duration", 4))))

        data["title"] = data.get("title", topic)
        return data

    def _blend_colors(self, c1: tuple[int, int, int], c2: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
        return tuple(
            int(c1[i] + (c2[i] - c1[i]) * ratio)
            for i in range(3)
        )

    def _generate_scene_image(self, title: str, subtitle: str, output_path: Path, idx: int) -> None:
        colors = {
            "dark": ((18, 28, 52), (62, 84, 168)),
            "blue": ((9, 58, 99), (19, 147, 190)),
            "purple": ((38, 24, 75), (125, 80, 210)),
            "green": ((11, 57, 53), (22, 164, 121)),
        }
        bg1, bg2 = colors.get(self.theme, colors["dark"])

        img = Image.new("RGB", (self.width, self.height), bg1)
        pixels = np.array(img)
        for y in range(self.height):
            ratio = y / max(1, self.height)
            color = self._blend_colors(bg1, bg2, ratio)
            pixels[y, :, :] = color
        img = Image.fromarray(pixels)

        draw = ImageDraw.Draw(img)

        # Add glow or accent circle
        x_center = self.width - 200
        y_center = 140
        for r in range(200, 20, -20):
            alpha = max(0, 70 - (200 - r))
            overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            d = ImageDraw.Draw(overlay)
            d.ellipse((x_center - r, y_center - r, x_center + r, y_center + r), fill=(120, 160, 255, alpha))
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        # Accent bar
        bar = Image.new("RGBA", (self.width, 10), (255, 255, 255, 80))
        img = Image.alpha_composite(img.convert("RGBA"), bar).convert("RGB")

        font_path = self._safe_font()
        font_title = ImageFont.truetype(font_path or "DejaVuSans-Bold.ttf", 58) if font_path else ImageFont.load_default()
        font_sub = ImageFont.truetype(font_path or "DejaVuSans.ttf", 30) if font_path else ImageFont.load_default()

        title_lines = self._wrap_text(title, 18, font_title, self.width - 120)
        title_y = 130
        for line in title_lines:
            bbox = draw.textbbox((0, 0), line, font=font_title)
            text_w = bbox[2] - bbox[0]
            draw.text(((self.width - text_w) / 2, title_y), line, font=font_title, fill=(255, 255, 255))
            title_y += max(60, bbox[3] - bbox[1] + 8)

        subtitle_lines = self._wrap_text(subtitle, 35, font_sub, self.width - 200)
        subtitle_y = title_y + 18
        for line in subtitle_lines:
            bbox = draw.textbbox((0, 0), line, font=font_sub)
            text_w = bbox[2] - bbox[0]
            draw.text(((self.width - text_w) / 2, subtitle_y), line, font=font_sub, fill=(220, 224, 255))
            subtitle_y += max(32, bbox[3] - bbox[1] + 8)

        img.save(output_path)

    def _wrap_text(self, text: str, max_chars: int, font: ImageFont.ImageFont, max_width: int) -> List[str]:
        words = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            bbox = font.getbbox(candidate)
            if bbox[2] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        if not lines:
            return [text[:max_chars]]
        return lines

    def _create_voiceover(self, text: str, audio_path: Path) -> None:
        if not self.use_voiceover:
            return
        tts = gTTS(text=text, lang="ar", slow=False)
        tts.save(str(audio_path))

    def _finalize_output(self, title: str, storyboard: Dict[str, Any], export_path: str) -> str:
        clips = []
        for idx, scene in enumerate(storyboard["scenes"]):
            scene_title = scene.get("title", f"مشهد {idx + 1}")
            scene_text = scene.get("text", "")
            duration = int(scene.get("duration", 4))

            image_file = self.output_dir / f"scene_{idx + 1}.png"
            self._generate_scene_image(scene_title, scene_text, image_file, idx)

            audio_file = self.output_dir / f"scene_{idx + 1}.mp3"
            if self.use_voiceover and scene_text.strip():
                self._create_voiceover(scene_text, audio_file)

            clip = ImageClip(str(image_file), duration=duration)
            if audio_file.exists():
                audio = AudioFileClip(str(audio_file)).subclip(0, duration)
                clip = clip.set_audio(audio)
            clips.append(clip)

        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(export_path, fps=self.fps, codec="libx264", audio_codec="aac")
        return export_path

    def generate_video(self, topic: str, output_file: str = "output.mp4") -> str:
        storyboard = self.generate_storyboard(topic)
        export_path = str(Path(output_file))
        return self._finalize_output(storyboard.get("title", topic), storyboard, export_path)


def main() -> None:
    print("=" * 60)
    print("🎬 Claude AI Video Generator - نسخة احترافية")
    print("=" * 60)

    topic = input("\n📌 أدخل موضوع الفيديو: ").strip()
    if not topic:
        print("❌ يجب إدخال موضوع صحيح.")
        return

    generator = ClaudeVideoGenerator()
    output_name = f"{topic.replace(' ', '_')[:30]}_video.mp4"

    try:
        result = generator.generate_video(topic, output_name)
        print(f"\n✅ تم إنشاء الفيديو بنجاح: {result}")
    except Exception as exc:  # pragma: no cover
        print(f"❌ حدث خطأ أثناء إنشاء الفيديو: {exc}")


if __name__ == "__main__":
    main()

