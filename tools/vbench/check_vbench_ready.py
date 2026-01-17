#!/usr/bin/env python3
"""
检查 VBench 评估前的准备工作是否完成
"""

import argparse
from pathlib import Path
from typing import List, Tuple


def check_video_format(video_path: Path) -> Tuple[bool, str]:
    """检查视频文件命名格式是否正确"""
    name = video_path.name
    
    # 检查扩展名
    if not name.endswith(('.mp4', '.gif')):
        return False, f"不支持的格式: {name}"
    
    # 检查是否包含 "-{数字}.mp4" 格式
    if '-' not in name:
        return False, f"缺少分隔符 '-': {name}"
    
    parts = name.rsplit('-', 1)
    if len(parts) != 2:
        return False, f"格式不正确: {name}"
    
    index_part = parts[1].replace('.mp4', '').replace('.gif', '')
    if not index_part.isdigit():
        return False, f"索引不是数字: {name}"
    
    index = int(index_part)
    if index < 0 or index >= 5:
        return False, f"索引超出范围 (0-4): {name}"
    
    return True, "格式正确"


def check_vbench_ready(videos_dir: Path, verbose: bool = False) -> bool:
    """
    检查 VBench 评估前的准备工作
    
    Returns:
        bool: 是否准备好
    """
    print("=" * 70)
    print("VBench 评估前检查")
    print("=" * 70)
    print()
    
    all_ok = True
    
    # 1. 检查目录是否存在
    print("1. 检查视频目录...")
    if not videos_dir.exists():
        print(f"   ❌ 目录不存在: {videos_dir}")
        return False
    print(f"   ✓ 目录存在: {videos_dir}")
    
    # 2. 检查是否有子目录（不应该有）
    print()
    print("2. 检查目录结构...")
    subdirs = [d for d in videos_dir.iterdir() if d.is_dir()]
    if subdirs:
        print(f"   ❌ 发现子目录 {len(subdirs)} 个（VBench 要求所有视频在同一目录）")
        print(f"      子目录示例: {subdirs[0].name}")
        all_ok = False
    else:
        print("   ✓ 无子目录（正确）")
    
    # 3. 检查视频文件
    print()
    print("3. 检查视频文件...")
    videos = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.gif"))
    if not videos:
        print("   ❌ 没有找到视频文件")
        return False
    
    print(f"   ✓ 找到 {len(videos)} 个视频文件")
    
    # 4. 检查命名格式
    print()
    print("4. 检查文件命名格式...")
    format_errors = []
    for video in videos[:10]:  # 只检查前10个
        ok, msg = check_video_format(video)
        if not ok:
            format_errors.append((video.name, msg))
            all_ok = False
    
    if format_errors:
        print(f"   ❌ 发现 {len(format_errors)} 个格式错误:")
        for name, msg in format_errors[:5]:
            print(f"      - {name}: {msg}")
    else:
        print("   ✓ 命名格式正确")
        if verbose:
            print(f"      示例: {videos[0].name}")
    
    # 5. 检查每个 prompt 是否有 5 个视频
    print()
    print("5. 检查每个 prompt 的视频数量...")
    from collections import defaultdict
    prompt_counts = defaultdict(int)
    
    for video in videos:
        # 提取 prompt（文件名去掉 "-{index}.mp4"）
        name = video.name
        if '-' in name:
            prompt = name.rsplit('-', 1)[0]
            prompt_counts[prompt] += 1
    
    incomplete_prompts = []
    for prompt, count in prompt_counts.items():
        if count != 5:
            incomplete_prompts.append((prompt, count))
            all_ok = False
    
    if incomplete_prompts:
        print(f"   ❌ 发现 {len(incomplete_prompts)} 个 prompt 视频数量不正确:")
        for prompt, count in incomplete_prompts[:5]:
            print(f"      - '{prompt}': {count} 个视频（应该是 5 个）")
    else:
        print(f"   ✓ 所有 {len(prompt_counts)} 个 prompts 都有 5 个视频")
    
    # 6. 检查文件可读性
    print()
    print("6. 检查文件可读性...")
    unreadable = []
    for video in videos[:10]:  # 只检查前10个
        if not video.is_file():
            unreadable.append(video.name)
            all_ok = False
    
    if unreadable:
        print(f"   ❌ 发现 {len(unreadable)} 个不可读文件")
    else:
        print("   ✓ 文件可读")
    
    # 总结
    print()
    print("=" * 70)
    if all_ok:
        print("✅ 所有检查通过！可以运行 VBench 评估")
    else:
        print("❌ 发现问题，请先修复后再运行 VBench 评估")
    print("=" * 70)
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="检查 VBench 评估前的准备工作"
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="视频目录路径",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细信息",
    )
    
    args = parser.parse_args()
    
    videos_dir = Path(args.videos_dir)
    check_vbench_ready(videos_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()


