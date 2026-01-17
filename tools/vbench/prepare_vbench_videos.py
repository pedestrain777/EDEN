#!/usr/bin/env python3
"""
将生成的视频文件重命名/创建符号链接为 VBench 期望的格式。

VBench 期望格式: {prompt}-{index}.mp4
例如: "In a still frame, a stop sign-0.mp4"

实际格式: prompt_XXXX/{index}.mp4
例如: prompt_00000/0.mp4
"""

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict


def load_metadata(metadata_path: Path) -> Dict[str, str]:
    """加载 metadata.csv，返回 prompt_id -> prompt 的映射"""
    prompt_map = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_id = row["prompt_id"]
            prompt = row["prompt"]
            if prompt_id not in prompt_map:
                prompt_map[prompt_id] = prompt
    return prompt_map


def sanitize_filename(text: str) -> str:
    """清理文件名，移除不合法字符"""
    # 移除或替换不合法字符
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        text = text.replace(char, '_')
    # 移除前后空格
    text = text.strip()
    # 限制长度（避免文件名过长）
    if len(text) > 200:
        text = text[:200]
    return text


def prepare_videos(
    videos_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    use_symlink: bool = True,
    dry_run: bool = False,
) -> None:
    """
    准备视频文件为 VBench 期望的格式
    
    Args:
        videos_dir: 原始视频目录（包含 prompt_XXXX/ 子目录）
        metadata_path: metadata.csv 路径
        output_dir: 输出目录（VBench 期望的格式）
        use_symlink: 是否使用符号链接（True）还是复制文件（False）
        dry_run: 是否只显示将要执行的操作，不实际执行
    """
    prompt_map = load_metadata(metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 输入目录: {videos_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🔗 使用方式: {'符号链接' if use_symlink else '复制文件'}")
    print(f"🧪 模式: {'干运行（只显示）' if dry_run else '实际执行'}")
    print()
    
    total_videos = 0
    success_count = 0
    error_count = 0
    
    # 遍历所有 prompt 目录
    for prompt_dir in sorted(videos_dir.glob("prompt_*")):
        prompt_id = prompt_dir.name.replace("prompt_", "")
        
        if prompt_id not in prompt_map:
            print(f"⚠️  跳过 {prompt_dir.name}: 在 metadata.csv 中找不到对应的 prompt")
            continue
        
        prompt = prompt_map[prompt_id]
        sanitized_prompt = sanitize_filename(prompt)
        
        # 遍历该 prompt 的所有视频（0.mp4, 1.mp4, 2.mp4, 3.mp4, 4.mp4）
        for video_idx in range(5):
            source_video = prompt_dir / f"{video_idx}.mp4"
            
            if not source_video.exists():
                print(f"⚠️  跳过: {source_video} 不存在")
                continue
            
            # 目标文件名: {prompt}-{index}.mp4
            target_name = f"{sanitized_prompt}-{video_idx}.mp4"
            target_path = output_dir / target_name
            
            total_videos += 1
            
            if target_path.exists() and not dry_run:
                print(f"⚠️  已存在，跳过: {target_name}")
                continue
            
            if dry_run:
                print(f"  [{total_videos}] 将创建: {target_name}")
                print(f"      源文件: {source_video}")
                success_count += 1
            else:
                try:
                    if use_symlink:
                        # 创建符号链接（使用绝对路径）
                        target_path.symlink_to(source_video.resolve())
                    else:
                        # 复制文件
                        shutil.copy2(source_video, target_path)
                    success_count += 1
                    if total_videos % 10 == 0:
                        print(f"  ✓ 已处理 {total_videos} 个视频...")
                except Exception as e:
                    error_count += 1
                    print(f"  ❌ 错误: {target_name} - {e}")
    
    print()
    print("=" * 70)
    print("处理完成")
    print("=" * 70)
    print(f"总视频数: {total_videos}")
    print(f"成功: {success_count}")
    if error_count > 0:
        print(f"错误: {error_count}")
    print()
    print(f"✅ 输出目录: {output_dir}")
    print(f"   现在可以将此目录作为 --videos_path 传递给 VBench")


def main():
    parser = argparse.ArgumentParser(
        description="准备视频文件为 VBench 期望的格式"
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="原始视频目录（包含 prompt_XXXX/ 子目录）",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="metadata.csv 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录（VBench 期望的格式）",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="复制文件而不是创建符号链接",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将要执行的操作，不实际执行",
    )
    
    args = parser.parse_args()
    
    videos_dir = Path(args.videos_dir)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    
    if not videos_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {videos_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv 不存在: {metadata_path}")
    
    prepare_videos(
        videos_dir=videos_dir,
        metadata_path=metadata_path,
        output_dir=output_dir,
        use_symlink=not args.copy,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()


