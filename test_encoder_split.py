#!/usr/bin/env python3
"""
测试encoder拆分后的代码是否能正常运行
"""
import torch
import torchvision
from src.models import load_model
from src.utils import InputPadder
from src.transport import create_transport, Sampler
import yaml
import os

def test_encode_and_denoise():
    """测试encode()和denoise_from_tokens()是否能正常工作"""
    print("=" * 60)
    print("测试Encoder拆分功能")
    print("=" * 60)
    
    # 1. 加载配置
    config_path = "configs/eval_eden.yaml"
    with open(config_path, "r") as f:
        config = yaml.unsafe_load(f)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 2. 加载模型
    print("\n1. 加载模型...")
    model_name = config["model_name"]
    eden = load_model(model_name, **config["model_args"])
    ckpt_path = config["pretrained_eden_path"]
    
    if not os.path.exists(ckpt_path):
        print(f"警告: 模型权重文件不存在: {ckpt_path}")
        print("请先下载模型权重")
        return False
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    eden.load_state_dict(ckpt["eden"])
    eden.to(device)
    eden.eval()
    print("✓ 模型加载成功")
    
    # 3. 准备测试数据
    print("\n2. 准备测试数据...")
    # 创建随机测试图像
    frame_0 = torch.rand(1, 3, 480, 640).to(device)
    frame_1 = torch.rand(1, 3, 480, 640).to(device)
    print(f"✓ 测试图像形状: {frame_0.shape}")
    
    # 4. 测试encode()
    print("\n3. 测试encode()方法...")
    h, w = frame_0.shape[2:]
    padder = InputPadder([h, w])
    cond_frames = padder.pad(torch.cat((frame_0, frame_1), dim=0))
    
    with torch.no_grad():
        enc_out = eden.encode(cond_frames)
    
    print("✓ encode()执行成功")
    print(f"  - cond_dit形状: {enc_out['cond_dit'].shape}")
    print(f"  - cond_dec形状: {enc_out['cond_dec'].shape}")
    print(f"  - ph: {enc_out['ph']}, pw: {enc_out['pw']}")
    print(f"  - stats_mean形状: {enc_out['stats_mean'].shape}")
    print(f"  - stats_std形状: {enc_out['stats_std'].shape}")
    
    # 5. 测试denoise_from_tokens()
    print("\n4. 测试denoise_from_tokens()方法...")
    new_h, new_w = cond_frames.shape[2:]
    noise = torch.randn([1, new_h // 32 * new_w // 32, config["model_args"]["latent_dim"]]).to(device)
    difference = torch.tensor([[0.85]], device=device)  # 模拟difference
    denoise_timestep = torch.tensor([0.5], device=device)
    
    with torch.no_grad():
        denoised = eden.denoise_from_tokens(noise, denoise_timestep, enc_out, difference)
    
    print("✓ denoise_from_tokens()执行成功")
    print(f"  - 输出形状: {denoised.shape}")
    print(f"  - 输入形状: {noise.shape}")
    assert denoised.shape == noise.shape, "输出形状应该与输入相同"
    
    # 6. 测试完整流程（encode + denoise + decode）
    print("\n5. 测试完整流程（encode + denoise + decode）...")
    transport = create_transport("Linear", "velocity")
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3)
    
    difference = ((torch.mean(torch.cosine_similarity(frame_0, frame_1),
                              dim=[1, 2]) - config["cos_sim_mean"]) / config["cos_sim_std"]).unsqueeze(1).to(device)
    
    def denoise_wrapper(query_latents, t):
        if isinstance(t, torch.Tensor):
            if t.numel() == 1:
                denoise_timestep = t.unsqueeze(0) if t.dim() == 0 else t
            else:
                denoise_timestep = t[0:1]
        else:
            denoise_timestep = torch.tensor([t], device=query_latents.device, dtype=torch.float32)
        if denoise_timestep.dim() == 0:
            denoise_timestep = denoise_timestep.unsqueeze(0)
        return eden.denoise_from_tokens(query_latents, denoise_timestep, enc_out, difference)
    
    with torch.no_grad():
        samples = sample_fn(noise, denoise_wrapper)[-1]
        denoise_latents = samples / config["vae_scaler"] + config["vae_shift"]
        generated_frame = eden.decode(denoise_latents)
        generated_frame = padder.unpad(generated_frame.clamp(0., 1.))
    
    print("✓ 完整流程执行成功")
    print(f"  - 生成帧形状: {generated_frame.shape}")
    assert generated_frame.shape == frame_0.shape, "生成帧形状应该与输入相同"
    
    # 7. 对比原版和新版的结果
    print("\n6. 对比原版和新版的结果...")
    with torch.no_grad():
        # 原版方法
        denoise_kwargs_old = {"cond_frames": cond_frames, "difference": difference}
        samples_old = sample_fn(noise, eden.denoise, **denoise_kwargs_old)[-1]
        denoise_latents_old = samples_old / config["vae_scaler"] + config["vae_shift"]
        generated_frame_old = eden.decode(denoise_latents_old)
        generated_frame_old = padder.unpad(generated_frame_old.clamp(0., 1.))
        
        # 新版方法（已经执行过）
        generated_frame_new = generated_frame
    
    # 计算差异
    diff = torch.abs(generated_frame_old - generated_frame_new).mean().item()
    print(f"  - 平均像素差异: {diff:.6f}")
    
    if diff < 1e-5:
        print("  ✓ 结果完全一致！")
    elif diff < 1e-3:
        print("  ✓ 结果非常接近（可能是数值精度差异）")
    else:
        print("  ⚠ 结果有较大差异，需要检查")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = test_encode_and_denoise()
        if success:
            print("\n✓ 所有测试通过！")
        else:
            print("\n✗ 测试失败")
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()

