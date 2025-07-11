from moviepy import VideoFileClip, AudioFileClip
import time


def concat_video_audio(video_path: str, audio_path: str):
    timestamp = int(time.time())
    file_name = f"audio_video_{timestamp}.mp4"
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    # 设置视频的音频
    final_clip = video_clip.with_audio(audio_clip)

    # 输出合并后的文件
    final_clip.write_videofile(
        file_name,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
    )

    # 释放资源
    video_clip.close()
    audio_clip.close()
    final_clip.close()

    return file_name
