# maidata「看谱面推谱师」模型 —— 从 `&inote_5=`→`E` 区间识别 `&des_5=` 区间

> v2：已剔除 `-` 与无署名谱；新增 1D-CNN 时序模型与本地交互界面（输入 zip 路径即可识别）。

## 1. 任务定义

每个 `maidata.txt` 的难度块结构固定为：

```
&lv_5=13.9
&des_5=Jack            ← 目标区间：谱师署名
&inote_5=(180){1},,    ← 输入区间起点
{16}4h[8:7]/5,,6,,,    …
E                      ← 输入区间终点（END 标记）
```

- **输入区间**：`&inote_5=` 之后第一个字符 → `E` 标记行（6477/6477 条样本全部以 `E` 正常收尾）。
- **输出目标**：`&des_5=` 字段所在区间 + 其取值（谱师）。
- 模型**只看音符区间原文**（不含曲名 / 版本 / 星数 / 目录名）。

## 2. 数据清洗（本版新增）

按要求剔除两类「平凡标签」：

| | 全量 | 清洗后（仅真名） |
|---|---|---|
| 样本 | 6477 | **3267** |
| 类别 | 204 | **202**（`∅` 与 `-` 已移除） |
| 留出集（按歌曲分组 12%） | 791 | **392**（187 首歌） |

其余统计不变：1651 个谱包、全部 UTF-8-SIG、0 坏包；每难度一条样本。

> 清洗后准确率数字**不能再和 v1 混看**：v1 的 59% 有一半是「猜出 ∅ / `-`」，v2 是纯粹的「认人」难度。

## 3. 两个模型（纯 numpy，无第三方依赖）

```
                    ┌─ 词袋分支 model_bag.npz ─────────────────────────────┐
音符区间原文 ───────┤   EmbeddingBag(音符格 token + 相邻二元组, 64 维)      │
                    │   EmbeddingBag(字符 2/3-gram, 48 维)                  │
                    │   123 维结构化数值特征                                │
                    │        → FC256+ReLU+Dropout → FC(202)                │
                    │                                                   ├→ 加权平均
                    └─ 时序分支 model_cnn.npz ─────────────────────────────┘   → + α·log先验
                        Embedding(音符格, 24)                               → softmax
                        conv(k3,s2)+ReLU → maxpool2 → conv → ReLU → maxpool2
                        → conv → ReLU → 掩码全局 max‖mean 池化 (96 维)
                        + EmbeddingBag(字符 n-gram) + 数值特征 → FC256 → FC(202)
```

- 反向传播**手写**（im2col 卷积 / 最大池化 / 掩码池化 / bincount 加速的嵌入 scatter）。
- 用 `test_cnn_grad.py` 做**有限差分梯度自检**：12 组参数全部通过（相对误差 ≤1.4e-2，第一层卷积因梯度量级小用绝对容差 3e-3 判定）。这一步抓到了一个真实 bug：`np.take_along_axis` 因形状广播错位导致全局 max 池化取错通道，已修正。
- 训练：`Adam + 余弦退火`，类权重 `1/√频次` 平衡长尾，验证集上搜索先验权重 α。

## 4. 效果（清洗后留出集，n=392）

| 模型 | Top-1 | Top-3 | Top-5 | Macro-F1 |
|---|---|---|---|---|
| 词袋 model_bag | 40.3% | 60.2% | 73.7% | 0.384 |
| 1D-CNN model_cnn | 33.2% | 56.1% | 68.6% | 0.351 |
| **集成（词袋×1 + CNN×0.2，α=0.5，默认）** | **39.8%** | **60.2%** | **73.5%** | — |
| 众数基线 | 20.6% | — | — | — |

分难度 Top-1（`metrics_bag.json` 的 `val_by_diff`，带先验 prior@1）：

| 难度 | N | 留出集 n | plain@1 | prior@1 | plain@3 | plain@5 |
|---|---|---|---|---|---|---|
| EXPERT | 4 | 187 | 46.5% | 46.0% | 66.3% | 79.7% |
| MASTER | 5 | 187 | 26.2% | 36.4% | 55.1% | 67.9% |
| **Re:MASTER** | **6** | **18** | **16.7%** | **22.2%** | **33.3%** | **50.0%** |

> **难度编号对照**：2 BASIC / 3 ADVANCED / 4 EXPERT / 5 MASTER / **6 Re:MASTER（remaster）** / **7 宴会場**。
> 星数带 `?` 后缀（`14?`、`13+?`）是**宴会場（N=7）**的标志；**Re:MASTER 是 N=6**，星数是普通数字（`13.7` 等）。
> Re:MASTER 全档案 149 条、`&des_6=` **全部有署名**，完整参与训练与验证；宴会場 92 条、`&des_7=` **全部为空**，不参与训练也无法验证。

**诚实的结论：1D-CNN 并没有打败词袋。** 原因有二：

1. 数据太少——3267 条样本 × 202 个类别，卷积的时序建模能力过剩，训练 loss 很快压到 0.001 就开始过拟合（即使加了 dropout 0.5 + 权重衰减 3e-4 仍在 ep20 后回落）。
2. 早期不加正则时 CNN 明显更差（29.3%），正则后才追平到 33%。

CNN 唯一稳定的贡献是把 Top-1 从 40.3% 抬到 39.8~40.8% 区间（网格最优 w=0.1 时 40.8%），提升在噪声范围内。**默认仍以词袋为主、CNN 权重 0.2**，你可以直接改：

```powershell
--models "model_bag.npz:1"                  # 纯词袋（最快）
--models "model_bag.npz:1,model_cnn.npz:0.2" # 默认
```

另有一个反直觉的发现：**MASTER 专用模型反而更差**（30.5% vs 通用模型在 diff5 上的 36.4%）。v1 里专用模型更强，是因为它还能利用「难度→大概率是无署名/某谱师」的捷径；清洗掉 ∅ / `-` 之后这个捷径没了，专训只剩数据量劣势。所以本版没有采用它。

## 5. 交互界面（输入 zip 路径即可识别）

**双击 `start.bat` 即可**：自动定位 Python、选择空闲端口（默认 8765，被占用时自动顺延）、启动服务并弹出浏览器；关闭那个窗口（或 `Ctrl+C`）即停止，不留后台进程。也支持 `start.bat --no-browser`（只起服务不开浏览器）与 `set DES5_PORT=9000` 自定义端口。

等价的命令行方式：

```powershell
& $py server.py --port 8765      # 打开 http://127.0.0.1:8765
```

- 纯标准库 `http.server`，无 Flask/Node 依赖；界面在 `static/index.html`。
- 输入框贴服务器本地绝对路径（`.zip` 或 `maidata.txt`，也可贴文件夹 → 自动取前 20 个包），选难度 + 候选数，点「识别」。
- 结果卡片左右并排显示 **输入区间**（字符偏移 + 行号 + 音符行数 + 是否以 `E` 收尾）与 **输出区间**（`&des_5=` 字符偏移 + 取值区间 + 实际值），下方是候选谱师的概率条与命中标记。
- 对 `-` / 无署名的谱会显式提示「不在模型标签空间内」，此时给出的是「谁可能打这首歌」的候选，不计入命中率。
- 接口：`GET /api/health`、`POST /api/predict {"path": "...", "diffs": "5", "top": 6, "limit": 20}`。
- 实测：`ORANGE` 整个目录批量跑（12 个包 × 2 个难度 = 24 个谱面），Top-1 命中 19 个（79%）。

命令行同样可用：

```powershell
& $py predict.py "..\PRiSM\Jack\[DX] Empire of Winter.zip" --diff 5
& $py predict.py "..\UNiVERSE" --diff 5 --limit 20 --json out.json
```

## 6. 关于「predict.py 有 239 个错误」

那是 IDE 的 **basedpyright 严格模式类型告警**，不是运行错误（脚本从未报过运行时异常）。239 条里：

- **约 200 条是同一个根因的连锁反应**：`parse_maidata()` 没有返回类型注解 → pyright 把返回值推断成 `str | dict | list` 联合类型 → 后面每一行 `rec["diff"]` 都报「无法赋值给 `__getitem__` 的 key」。
- 其余是 numpy 相关的高级 `Any` 报告与「缺少参数注解」类的提示，对数值脚本没有实际收益。

已做的处理：

1. 给 `decode()` / `parse_maidata()` / `locate_des_region()` / `analyze_file()` 等补上精确类型注解 → **239 条降到 3 条**。
2. 新增 `pyrightconfig.json`，把 `reportImplicitRelativeImport` 等对「可直接 `python xxx.py` 运行的脚本目录」无意义的规则关掉，并注明这是有意为之。
3. 剩下 3 条是 `import features / cnn / extract_dataset` 的同目录导入提示。这些脚本是按「进目录直接跑」设计的（`python predict.py ...`），改成相对导入就必须变成包并用 `python -m des5_ai.predict` 运行，所以保留现状。

## 7. 复现全流程

```powershell
$py = "C:\Users\LENOVO\AppData\Local\Programs\Python\Python314\python.exe"
cd c:\MaiArchive-全版本.zip-nobga.df310f7b0af67fbd\des5_ai

& $py extract_dataset.py                 # 扫描 1651 个包 → data/dataset.jsonl（含两个区间偏移）
& $py build_features.py --named-only     # 特征缓存 data/features.npz（剔除 ∅ / -）
& $py train.py --epochs 25 --out model_bag.npz --metrics metrics_bag.json
& $py train_cnn.py --epochs 30 --dropout 0.5 --wd 3e-4 --out model_cnn.npz --metrics metrics_cnn.json
& $py test_cnn_grad.py                   # 梯度自检（改 cnn.py 后必跑）
& $py eval_ensemble.py                    # 词袋 / CNN / 集成的对比与权重网格
& $py server.py --port 8765              # 交互界面
```

## 8. 文件清单

| 文件 | 说明 |
|---|---|
| `extract_dataset.py` | 全档案扫描，抽取音符区间与 des 区间（含字符偏移） |
| `features.py` | 音符格 / 字符 n-gram / 时序 id / 123 维结构化特征 |
| `build_features.py` | 生成 `data/features.npz`（`--named-only` 剔除 ∅ 与 `-`） |
| `train.py` | 词袋模型训练 + 留出集评估 |
| `cnn.py` | 1D-CNN 前向 / 反向（手写卷积、池化、掩码） |
| `train_cnn.py` | CNN 训练（dropout / 权重衰减 / 先验搜索） |
| `test_cnn_grad.py` | 有限差分梯度自检（12 组参数） |
| `eval_ensemble.py` | 词袋 / CNN / 集成在留出集上的对比与权重网格 |
| `predict.py` | 推理：双架构 + 任意加权集成 + 区间定位 |
| `server.py` / `static/index.html` | 零依赖本地交互界面 |
| `start.bat` | 双击启动器（找 Python、避让端口、自动开浏览器） |
| `README.md` | 包内快速上手说明 |
| `make_package.py` | 重新打包为 zip |
| `pyrightconfig.json` | 类型检查配置（说明同目录导入为有意写法） |
| `model_bag.npz` / `model_cnn.npz` | 模型权重；`metrics_*.json` 为验证指标 |
| `data/` | `dataset.jsonl`、`features.npz`、`summary.json` |
| `report.md` | 本文件 |

## 9. 数据观察与局限

1. **高难度最难，且 Re:MASTER 最差**：EXPERT Top-1 46.5% → MASTER 26.2%（带先验 36.4%）→ **Re:MASTER 仅 16.7%（带先验 22.2%）**。Re:MASTER 留出集只有 18 条，且谱师高度集中（はっぴー 28/149、Jack 16、サファ太 14），长尾谱师的 Re:MASTER 谱风格与 MASTER 差异大。
2. **长尾极重**：202 个类别里最多 60 条、最少 1 条，单 MASTER 难度可用训练样本仅约 1400 条——这是当前瓶颈。
3. **严格只用音符区间**：一旦引入 `&version=`，「版本 ≈ 某位谱师」的先验会大幅抬高分数（PRiSM → サファ太 / 7.3Hz＋Jack 等），但那就不是「看谱面认人」了。
4. **目录名往往等于谱师全名**（全档案 1651 个包实测：目录名 == 最高难度署名 **85.5%**，不符 8.9%，全部无署名 5.6%），且文件内 `&des_N=` 可能被截断。若只是要补全署名，读目录名更准；模型的价值在**目录名缺失 / 被改名 / 谱包被二次分发**时的推断。注意宴会場的 92 个包直接放在版本根目录、没有谱师子目录（因为无署名），这条规则对它们不适用。
5. 下一步若要继续提升：把 1651 首的 `&version=`、星数作为**可选开关**接入（会大幅涨点但改变了任务性质）；或引入更多标注数据（现有数据里 6477 条样本只覆盖 202 个真名标签）。
