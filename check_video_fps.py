#!/usr/bin/env python3
"""
检查视频帧率的脚本
"""
import torchvision
import sys

def check_fps(video_path):
    """检查视频的帧率"""
    try:
        _, _, video_info = torchvision.io.read_video(video_path, pts_unit='sec')
        fps = video_info['video_fps']
        print(f"视频: {video_path}")
        print(f"帧率: {fps} fps")
        return fps
    except Exception as e:
        print(f"错误: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_video_fps.py <视频路径1> [视频路径2]")
        sys.exit(1)
    
    fps_list = []
    for video_path in sys.argv[1:]:
        fps = check_fps(video_path)
        if fps:
            fps_list.append(fps)
        print()
    
    if len(fps_list) == 2:
        print(f"帧率变化: {fps_list[0]} fps → {fps_list[1]} fps")
        print(f"倍数: {fps_list[1] / fps_list[0]:.2f}x")

