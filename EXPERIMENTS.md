# 實驗紀錄

依時間順序。每筆記錄：目的、設定、指令、結果、結論。失敗的也記，因為它們決定了現在的設計。
數字都來自 `outputs/` 底下的 log（不進 git）。硬體：單張 L40S。

---

## E0. 環境與儀器（2026-10-04 ~ 10-05）

### E0.1 F/T 校準
- 目的：確認 wrist F/T（hand link 的 incoming joint wrench）方向、座標系、大小都對。
- 設定：靜止時對 hand 施加已知外力。
- 結果：+10 N z → 9.97 N；+5 N x → 4.92 N；+5 N y → 5.08 N。靜止讀數 0，不漂移。
- 過程中修掉的問題：
  1. Isaac Lab OSC 不開 inertial decoupling 時 nullspace 投影不是 dynamically consistent，手會漂 → 改寫自己的 impedance controller（`guide_anything/control.py`）。
  2. 繞 z 軸的 task-space 慣量只有 0.0028 kg·m²，旋轉剛性 40 時顯式積分不穩，joint 7 衝到速度上限 → 旋轉剛性改成 (20, 20, 5)。

### E0.2 接觸曲線（`scripts/check_contact.py`）
- 兩個世界同一個盲壓動作，損壞前 F/T 差 0.003–0.05 N（GPU physics 雜訊）。
- FLOOR 在 10.4 N 損壞，LATCH 在 15 N 彈開並滑到孔底（12 mm）。

### E0.3 Oracle motor（`scripts/scripted_insert.py`）
- 告訴 controller 世界是哪個，兩個世界成功率都是 1.000（2048 episodes），損壞 0。
- 速度：256 / 1024 / 4096 envs → 1.7k / 6.6k / 22.8k env steps/s。

### E0.4 Aliasing certificate（`scripts/aliasing_certificate.py`）
- 目的：停在 8 N 之前，sensor history 分不出兩個世界（checklist 1.4）。
- 設定：2500 episodes 各半，盲壓到 8 N 停住，110 步完整 observation，一半訓練一半測試。
- 結果（最新，含 commit action、1 mm 間隙）：logistic 0.503、MLP 0.494（隨機的 s.e. 0.014）→ PASS。
- 對照組（壓到 18 N，世界真的不同）：兩者都 1.000，證明分類器抓得到外洩。
- 第一版（0.5 mm 間隙、12 維 observation）：logistic 0.516、MLP 0.502；換 5 種切法平均 0.497。

---

## E1. Checklist 1.2：學習器學得到「p > 0.8 才壓」嗎

共同設定：belief p ~ U(0,1)，世界依 p 抽。只在最後一步給結果（成功 +1、損壞 −3），所有 episode 等長，加 potential-based shaping。解析最佳：p > 0.8 才壓，完美 motor 的 return 0.600。

參考線（`eval_decision.py --scripted`）：scripted 門檻 0.8 → 門檻 0.800、return 0.606；永遠停 0.50；永遠壓 −1.0。

### E1.1 PPO 從零開始（失敗）
| 設定 | 結果 |
|---|---|
| noise 1.0, γ 0.99, λ 0.95 | 第 200 iter 門檻 0.786 但停下的 episode 不靠近（return 0.09）；第 299 iter 崩成幾乎都壓（門檻 0.06, return −0.79） |
| noise 0.3 | 塌成永遠壓（pressed 0.99） |
| γ 0.998, λ 0.99, noise 0.3 | 塌成不靠近不碰（success 0） |
| γ 0.998, λ 0.99, noise 1.0 | 塌成永遠壓 |
- 結論：接觸時 action 等於力，探索雜訊一碰就超過 10 N；PPO 學不會 motor 和決策同時進行。
- 另外確認 reward 沒寫錯：scripted 門檻 0.8 的折扣 return 0.672 > 永遠停 0.651 > 永遠壓 0.335。

### E1.2 一連串修正（每項都是看到失敗模式後加的）
1. Commit action + 8 N 安全上限：沒 commit 不可能壓壞。
2. DAgger warm start，teacher 永遠不壓（只教 motor）。
3. Motor 凍結：PPO 梯度會毀掉 0.25 mm 精度的插孔（固定 lr 1e-4 時第二批 episode 成功率就歸 0；1e-5 不會）。
4. Actor 的 observation normalizer 凍結：PPO 資料讓統計漂移，凍結的 motor 輸入被重新縮放。
5. 上一步動作只放 motor 3 維，不放 commit。
6. Asymmetric critic：壞掉的 FLOOR 零件 sensor 看不出來，critic 只看到 LATCH 的成功 → PPO 偏好壓。critic 額外看世界、損壞、彈開。
7. Guarded commit：碰到阻力才全力推，否則太早 commit 會撞孔口。

### E1.3 每步取樣的 commit（`run_1_2.sh`，TAG=_ent / _cover）
- `_ent`（entropy 0.01）：取樣評估 4/5 在 0.8±0.05（0.803, 1.000, 0.814, 0.782, 0.793），但 return 0.42–0.57；平均值評估 0/5（永遠不壓）。原因：決策寫在每步 commit 的取樣機率裡，平均值沒越過 0；壓過之後 motor 沒看過那些狀態。
- `_cover`（warm start 一半 episode 強制 commit，讓 motor 學壓過之後怎麼插到底）：return 改善（取樣 0.49–0.58），但門檻取樣 2/5、平均值 1/5。原因：每步擲骰子讓門檻在取樣時偏早、平均值時偏晚。

### E1.4 一次性決定（commit 只在第一次接觸讀一次）+ 1 mm 間隙 → PPO 通過
- 改動：commit 在第一次接觸時鎖定；間隙 0.5 → 1.0 mm（BC 的 xy 誤差 0.33 mm 會卡孔口）；DAgger 6 輪；commit std 0.3，NO_COMMIT −0.2；entropy 0.003。
- 驗證：warm start 後強制壓 → LATCH 成功 1.000；強制停 → FLOOR 成功 1.000。
- 指令：`TAG=_latch scripts/run_1_2.sh`（5 seeds × 1024 envs × 300 iter，約 2 小時）

| Seed | 門檻（平均值） | Return（平均值） | 門檻（取樣） | Return（取樣） |
|---|---|---|---|---|
| 0 | 0.812 ✅ | 0.597 | 0.826 ✅ | 0.586 |
| 1 | 0.788 ✅ | 0.597 | 0.809 ✅ | 0.588 |
| 2 | 0.941 ❌ | 0.543 | 0.997 ❌ | 0.536 |
| 3 | 0.810 ✅ | 0.593 | 0.895 ❌ | 0.547 |
| 4 | 0.773 ✅ | 0.591 | 0.776 ✅ | 0.578 |

- 結論：平均值評估 4/5 通過，return 距 Bayes 上限 < 0.01。s0 的按壓率：p ≤ 0.65 為 0、0.75 為 3%、0.85 為 89%、0.95 為 100%。

### E1.5 Q 學習器（contextual bandit）→ 5/5 通過
- 設定：同一個 env、同一個 motor（scripted teacher；PPO 那條路的 motor 就是模仿它，兩條路徑驗證過都 1.000）。第一次接觸時 Q(o, {停, 壓}) 選大的；ε 依序 1.0, 0.5, 0.3, 0.2, 0.1, 0.1，每輪 2048 episodes，對結果做回歸。
- 指令：`scripts/run_1_2_q.sh`

| Seed | 門檻 | Return |
|---|---|---|
| 0 | 0.819 ✅ | 0.606 |
| 1 | 0.802 ✅ | 0.597 |
| 2 | 0.788 ✅ | 0.599 |
| 3 | 0.795 ✅ | 0.606 |
| 4 | 0.772 ✅ | 0.602 |

- 資料量：每 seed 1.2 萬 episodes（約 180 萬步），PPO 每 seed 約 1000 萬步。

### 1.2 結論
- 架構表示得出校準過的「信任 / 拒絕」門檻。
- PPO 可以學到（4/5），但需要大量工程才穩，而且有一個 seed 停在偏保守的 0.94。
- Q 學習器 5/5、return 貼上限、資料量少一個量級，跟玩具版結論一致（fitted Q 5/5，PPO 10/10 塌）。
- 建議：決策層用 Q 學習器；motor 用 DAgger 模仿、凍結。

---

## 待做
- 1.3：加入 probe 動作，檢驗三個區域（停 / 試探 / 壓）。
- 2.0 identification 曲線、2.1 oracle gap（需要加可以摸得出來的 factor）。
- V-force 小版本（Claim 2 目前沒資料）。
- F/T 觀測 noise。
