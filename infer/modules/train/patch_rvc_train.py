"""
執行方式：python apply_patch.py
會把 train-org.py 修改後存成 train.py
"""

import re

with open("train-org.py", "r", encoding="utf-8") as f:
    src = f.read()

# ── 1. 在 global_step = 0 後面加入 _last_progress_time ──────────────────────
old_global = "global_step = 0"
new_global  = "global_step = 0\n_last_progress_time = 0.0  # 用於每 10 秒強制印出進度"
src = src.replace(old_global, new_global, 1)

# ── 2. 在 train_and_evaluate 函式頂端宣告使用該全域變數 ─────────────────────
old_global_step_decl = "    global global_step"
new_global_step_decl = "    global global_step, _last_progress_time"
src = src.replace(old_global_step_decl, new_global_step_decl, 1)

# ── 3. 在 epoch_recorder 初始化後，加入 batch 迴圈開頭的計時邏輯 ────────────
#    目標位置：for batch_idx, info in data_iterator: 的迴圈 body 最前面
#    找到 "# Data\n    ## Unpack" 並在它之前插入計時程式碼
old_loop_start = "        # Data\n        ## Unpack"
new_loop_start = """\
        # ── 每 10 秒強制輸出一次進度（不論 log_interval）──
        if rank == 0:
            _now = ttime()
            if _now - _last_progress_time >= 10.0:
                _last_progress_time = _now
                _pct = 100.0 * batch_idx / len(train_loader) if len(train_loader) > 0 else 0.0
                _ts  = datetime.datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{_ts}] Epoch {epoch} | "
                    f"batch {batch_idx}/{len(train_loader)} ({_pct:.1f}%) | "
                    f"global_step={global_step}",
                    flush=True,
                )

        # Data
        ## Unpack"""
src = src.replace(old_loop_start, new_loop_start, 1)

with open("train.py", "w", encoding="utf-8") as f:
    f.write(src)

print("完成！已輸出 train.py")