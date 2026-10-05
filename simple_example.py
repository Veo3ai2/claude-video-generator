"""
مثال بسيط لاستخدام منشئ الفيديو
"""

from video_generator import VideoGenerator

def main():
    # إنشاء مثيل من منشئ الفيديو
    generator = VideoGenerator()
    
    # مواضيع مختلفة للاختبار
    topics = [
        "فوائد الذكاء الاصطناعي في التعليم",
        "كيفية تعلم البرمجة",
        "أهمية الصحة العقلية",
    ]
    
    print("🎬 سيتم إنشاء فيديوهات تجريبية...\n")
    
    for topic in topics:
        try:
            output_file = f"demo_{topic.replace(' ', '_')[:20]}.mp4"
            print(f"\n📹 جاري إنشاء فيديو عن: {topic}")
            generator.generate_video(topic, output_file, duration=15)
            print(f"✅ تم إنشاء: {output_file}\n")
        except Exception as e:
            print(f"❌ خطأ: {str(e)}\n")

if __name__ == "__main__":
    main()
