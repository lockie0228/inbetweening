import os
import subprocess
import shutil

IN_ROOT = "videos_out"
OUT_ROOT = "videos_out_h264"
TARGET_FPS = 24
BITRATE = "4M"  # 可调：'2M' ~ '8M'

def has_encoder(enc_name: str) -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, text=True)
        return enc_name in out
    except Exception:
        return False

def pick_encoder():
    # 优先顺序：libx264 > h264_nvenc > libopenh264 > h264_videotoolbox
    if has_encoder("libx264"):
        return ["-c:v", "libx264"]
    if has_encoder("h264_nvenc"):
        return ["-c:v", "h264_nvenc", "-preset", "p5"]
    if has_encoder("libopenh264"):
        return ["-c:v", "libopenh264"]
    if has_encoder("h264_videotoolbox"):
        return ["-c:v", "h264_videotoolbox"]
    return None

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    enc = pick_encoder()
    if enc is None:
        print("[warn] 没有可用的 H.264 编码器（libx264/h264_nvenc/libopenh264 都不存在）。将直接复制文件。")
    for fname in os.listdir(IN_ROOT):
        if not fname.lower().endswith(".mp4"):
            continue
        in_path = os.path.join(IN_ROOT, fname)
        out_name = os.path.splitext(fname)[0] + ".mp4"
        out_path = os.path.join(OUT_ROOT, out_name)
        if enc is None:
            shutil.copy2(in_path, out_path)
            print(f"[copy] {in_path} -> {out_path}")
            continue
        cmd = [
            "ffmpeg", "-y", "-hide_banner",
            "-i", in_path,
            *enc,
            "-b:v", BITRATE,
            "-profile:v", "main",
            "-pix_fmt", "yuv420p",
            "-vf", f"fps={TARGET_FPS},format=yuv420p",
            "-movflags", "+faststart",
            out_path
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"[ok] {out_path}")

if __name__ == "__main__":
    main()
