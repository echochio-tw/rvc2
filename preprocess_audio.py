import os
import subprocess
import wave
import numpy as np

# ==========================================
#              參數配置區
# ==========================================
INPUT_DIR = "./raw_audio"        # 原始音檔資料夾
OUTPUT_DIR = "./dataset/me"      # 輸出資料夾
TARGET_SR = 40000                # 採樣率 (配合 SAMPLING_RATE = 40k)
SEGMENT_SEC = 10                 # 每段秒數
MIN_VOICE_SEC = 2.0              # 最短有效人聲秒數 (低於此值刪除)
MAX_SILENCE_RATIO = 0.5          # 最高靜音比例 (超過此比例刪除)
SILENCE_THRESHOLD = 0.01         # 靜音判斷門檻 (振幅 0~1)

# ==========================================
#              核心邏輯
# ==========================================
def analyze_wav(filepath):
    """分析 WAV 檔，回傳 (總秒數, 靜音秒數, 人聲秒數)"""
    with wave.open(filepath, 'r') as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    total_sec = len(audio) / sr
    frame_size = int(sr * 0.02)  # 20ms frame
    silence_frames = 0
    total_frames = 0

    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i+frame_size]
        rms = np.sqrt(np.mean(frame**2))
        if rms < SILENCE_THRESHOLD:
            silence_frames += 1
        total_frames += 1

    silence_sec = (silence_frames / total_frames) * total_sec if total_frames > 0 else total_sec
    voice_sec = total_sec - silence_sec
    return total_sec, silence_sec, voice_sec

def process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 收集所有音檔
    all_files = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
                all_files.append(os.path.join(root, f))

    if not all_files:
        print(f"❌ 在 {INPUT_DIR} 找不到音檔")
        return

    print(f"📁 找到 {len(all_files)} 個音檔，開始處理...")

    chunk_index = 0
    kept = 0
    deleted_silence = 0
    deleted_short = 0

    for src in all_files:
        base = os.path.splitext(os.path.basename(src))[0]
        tmp_wav = f"./tmp_{base}_full.wav"

        # 轉成目標格式
        subprocess.run([
            "ffmpeg", "-y", "-i", src,
            "-ar", str(TARGET_SR),
            "-ac", "1",
            "-sample_fmt", "s16",
            tmp_wav
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(tmp_wav):
            print(f"  ⚠️ 轉換失敗: {src}")
            continue

        # 切成小段
        tmp_seg_pattern = f"./tmp_{base}_%04d.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_wav,
            "-f", "segment",
            "-segment_time", str(SEGMENT_SEC),
            "-ar", str(TARGET_SR),
            "-ac", "1",
            "-sample_fmt", "s16",
            tmp_seg_pattern
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        os.remove(tmp_wav)

        # 分析每段並篩選
        seg_files = sorted([
            f for f in os.listdir(".")
            if f.startswith(f"tmp_{base}_") and f.endswith(".wav")
        ])

        for seg in seg_files:
            seg_path = os.path.join(".", seg)
            try:
                total_sec, silence_sec, voice_sec = analyze_wav(seg_path)
                silence_ratio = silence_sec / total_sec if total_sec > 0 else 1.0

                if voice_sec < MIN_VOICE_SEC:
                    print(f"  🗑️ 刪除 (人聲太短 {voice_sec:.1f}s): {seg}")
                    deleted_short += 1
                    os.remove(seg_path)
                elif silence_ratio > MAX_SILENCE_RATIO:
                    print(f"  🗑️ 刪除 (靜音太多 {silence_ratio*100:.0f}%): {seg}")
                    deleted_silence += 1
                    os.remove(seg_path)
                else:
                    out_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_index:04d}.wav")
                    os.rename(seg_path, out_path)
                    print(f"  ✅ 保留 chunk_{chunk_index:04d}.wav (人聲 {voice_sec:.1f}s, 靜音 {silence_ratio*100:.0f}%)")
                    chunk_index += 1
                    kept += 1
            except Exception as e:
                print(f"  ⚠️ 分析失敗: {seg} ({e})")
                if os.path.exists(seg_path):
                    os.remove(seg_path)

    print(f"""
========================================
✅ 完成！
   保留: {kept} 段
   刪除 (人聲太短): {deleted_short} 段
   刪除 (靜音太多): {deleted_silence} 段
   輸出位置: {OUTPUT_DIR}
========================================
""")

if __name__ == "__main__":
    process()
