# maidata 谱师识别模型（inote → des）

从 maimai 谱面文件中，**只看 `&inote_N=` 到 `E` 的音符区间**，预测 `&des_N=` 区间里的谱师姓名，
并给出该字段在原文中的字符区间。

## 环境要求

- Python 3.14（3.11+ 应可用）
- `numpy`（唯一第三方依赖；反向传播全部手写，无需 torch）

## 快速开始

### 方式一：双击 `start.bat`（推荐）

直接双击本目录下的 **`start.bat`**，会自动完成 4 步：

```
[1/4] 定位 Python        依次尝试 %PYTHON% → Python314/313/312/311 → PATH 里的 python → py -3
[2/4] 检测 numpy         有就直接用；没有则自动 pip install（见下）
[3/4] 检查模型文件        缺 model_bag.npz 时给出训练命令提示
[4/4] 检查端口           占用则自动往后顺延；占用者正是本服务则直接开页面
```

关掉那个黑色窗口（或按 `Ctrl+C`）即停止服务，不会残留后台进程。

**numpy 自动安装**

`numpy` 是唯一的第三方依赖（反向传播全部手写，不需要 torch）。`start.bat` 启动时会检测：

- 已安装 → 打印 `[2/4] numpy 2.5.1 (已安装)`
- 未安装 → 自动执行 `python -m pip install "numpy>=1.17"`（需要联网，约 10~90 秒），完成后打印 `[2/4] numpy 2.5.1 (刚刚自动安装完成)`
- 常规安装因权限失败 → 自动改用 `--user` 重试
- 仍失败 → 打印排查提示（网络 / 代理 / 磁盘空间 / pip 被禁用）与手动命令，并附清华镜像命令，然后退出

代码用到 `np.random.default_rng`，所以版本要求是 **numpy ≥ 1.17**。

- 端口被占用时：若占用者正是本服务，直接打开页面；若被别的程序占用，会自动往后找空闲端口。
- 换端口：`set DES5_PORT=9000 && start.bat`，或改脚本顶部的 `DES5_PORT`。
- 不开浏览器（仅启动服务）：`start.bat --no-browser`

### 方式二：命令行

```powershell
# 1) 单个谱包识别（默认 MASTER 难度，词袋+CNN 集成）
python predict.py "D:\MaiArchive\PRiSM\Jack\[DX] Empire of Winter.zip" --diff 5

# 2) 交互界面：贴路径 → 出结果
python server.py --port 8765      # 浏览器打开 http://127.0.0.1:8765

# 3) 批量（贴目录，自动取前 20 个包）
python predict.py "D:\MaiArchive\UNiVERSE" --diff 5 --limit 20

# 4) 桌面版入口（等价于 exe 里的行为：自动选端口 + 自动开浏览器）
python des5_app.py
```

界面支持的输入：`.zip` 谱包（含 `maidata.txt`）、`maidata.txt` 本身、或装着谱包的文件夹。

### 方式三：打包成 exe（目标机器无需装 Python）

```bat
python -m pip install pyinstaller
python build_exe.py
```

产物：`_build\dist\des5_ai.exe`，约 **25 MB**，自带 Python 运行时 + numpy + 模型 + 网页界面。

- **双击即用**：自动选一个空闲端口（默认从 8765 起，占用则顺延）、加载模型、拉起服务，等服务真的响应了再自动打开浏览器。
- **停止服务**：关闭黑窗口，或按 `Ctrl+C`。
- **重复双击不会起两个服务**：第二次启动会认出「服务已在运行」，直接打开已有页面后退出。
- 模型与界面从包内读取，因此 exe 放在桌面、U 盘、开始菜单都能正常工作，不依赖当前工作目录。

其他打包模式：

```bat
python build_exe.py --onedir     :: 文件夹版，启动更快（免去每次解压到 %TEMP%）
python build_exe.py --windowed   :: 不弹黑窗口，但只能用任务管理器结束进程
python build_exe.py --no-icon    :: 不设置 exe 图标
python build_exe.py --clean      :: 先清理构建产物再打包
```

**换 exe 图标**：Windows 的 exe 图标必须是 `.ico`（不能直接用 PNG），
用 `make_icon.py` 从任意 PNG 生成多分辨率图标：

```bat
python make_icon.py                                  :: 默认源图 -> des5_ai.ico
python make_icon.py "D:\某目录\image.png" -o des5_ai.ico
```

生成的 `des5_ai.ico` 会写入 16 / 24 / 32 / 48 / 64 / 128 / 256 七档尺寸，
Windows 按显示环境自动挑合适的（大图缩小不糊、小图放大不毛边）。
源图不是正方形时会居中补透明边，不会被拉伸变形。
`make_icon.py` 需要 Pillow（仅生成图标时需要，打包与运行时都不需要）。

> 默认保留黑窗口（`--console`）：既能看到模型加载日志与报错，也能用 `Ctrl+C` 正常停止。
> 若改用 `--windowed`，程序出错时不会有任何提示，且无法用 Ctrl+C 退出。
>
> 想让 exe 用自己重训的模型，把 `.npz` 放到同一目录并设环境变量 `DES5_HOME` 指向该目录即可。

### 网页三种输入方式

| 方式 | 操作 | 说明 |
|---|---|---|
| **① 浏览 / 拖拽** | 点「浏览…」多选文件，或把文件**直接拖到页面上** | **不用填路径**。浏览器出于安全限制拿不到本地绝对路径，所以前端读取文件字节上传、后端在内存里解析 zip / txt |
| ② 目录批量 | 贴文件夹路径 | 自动取前 20 个包，输出分难度命中率汇总 |
| ③ 单文件 | 贴单个文件路径 | 走磁盘路径，最快 |

### 多谱面融合：给出「综合谱师印象」

选多个谱包（或贴一个目录）时，除了逐张判断，还会把**全部谱面一次性前向传播**，
再用同一次推理得到的概率矩阵做跨谱面融合，给出一个综合结论。

**融合方式（幂平均，`q` 可调）**

| q | 含义 |
|---|---|
| **0（默认）** | **几何平均（对数几率池）** —— 只认「所有谱面共同支持」的人 |
| 1 | 算术平均 —— 只看总概率，某个谱面的一票不会消失 |
| 越大 | 越强调一致性 |

**留出集实测（35 位有 ≥2 张谱的谱师，判对几位）**

| 策略 | 命中 |
|---|---|
| 多数票（Top-1 投票） | 11/35 = 31.4% |
| 算术平均 q=1 | 11/35 = 31.4% |
| 幂平均 q=1.5 | 11/35 = 31.4% |
| **几何平均 q=0（默认）** | **13/35 = 37.1%** |

几何平均胜出，说明「**所有谱面都指向同一人**」比「总概率高」更可靠，所以默认 q=0。

**⚠ 融合效果几乎完全由谱面数量决定**

| 谱面数 | 实测融合命中率 |
|---|---|
| 2 张 | **0/9（完全无效）** |
| 3~4 张 | 2/7 ≈ 29% |
| 5~8 张 | 3/8 ≈ 25~38% |
| 9~15 张 | 3/5 = 40~60% |
| **≥16 张** | **6/6 ≈ 83~100%** |

所以界面会按谱面数给出可靠性提示：**谱面 <16 张时明确警告「仅供参考，请直接看单张判断」**。

> 注意：「2 张谱融合命中率 0%」不代表模型变差 —— 单张 Top-1 本身就有约 39%，
> 两张谱做平均反而会把本来判对的那张稀释掉。**融合是为「手上有很多谱、想知道整体像谁」设计的，
> 不是为「只有两三张谱」设计的。**

融合结果附带三项诊断：

- **票数一致性** —— 有多少张谱的单张 Top-1 就是这个人
- **与次名差距 / 归一化熵** —— 分布有多尖锐
- **已有署名对照** —— 若这批谱面本身带 `&des_N=` 真值，会单独列出并标注是否与综合结论一致（真值**不参与**融合，避免「用答案证明答案」）

实测举例（`FiNALE\はっぴー`，MASTER，9 个包）：

```
综合谱师印象 · 基于 9 张谱面 / 9 个谱包
可靠性：[mid] 谱面较多（9~15 张），实测融合命中率约 40~60%
Top1 = はっぴー (0.985)   票数一致性 8/9 (89%)   与次名差距 0.975
其中 9 张谱面在文件里已有署名（はっぴー）— 与综合结论一致 ✔
```

上传限制：单文件 64 MB、单次请求 256 MB、一次最多 40 个文件；每个文件独立容错，一个坏包不影响整批。

```powershell
# 等价的命令行上传测试
& $py predict.py "D:\MaiArchive\FiNALE" --diff 6 --top 5
```

接口：`GET /api/health`、`POST /api/predict {"path","diffs","top","limit"}`、`POST /api/upload {"files":[{"name","b64"}],"diffs","top"}`。

## 从头复现（需要原始谱包档案）

```powershell
python extract_dataset.py                 # 扫描全档案 → data/dataset.jsonl（含两个区间偏移）
python build_features.py --named-only     # → data/features.npz（剔除 "-" 与无署名谱）
python train.py --epochs 25 --out model_bag.npz --metrics metrics_bag.json
python train_cnn.py --epochs 30 --dropout 0.5 --wd 3e-4 --out model_cnn.npz --metrics metrics_cnn.json
python test_cnn_grad.py                   # 梯度自检（改过 cnn.py 后务必跑）
python eval_ensemble.py                    # 词袋 / CNN / 集成 对比与权重网格
```

## 目录结构

| 路径 | 说明 |
|---|---|
| `predict.py` | 推理：双架构 + 任意加权集成 + 区间定位（也可当库 import） |
| `paths.py` | 资源定位：源码形态用脚本目录，打成exe 后用 PyInstaller 解包目录 |
| `server.py`、`static/index.html` | 零依赖本地交互界面（标准库 http.server） |
| `des5_app.py` | 桌面版入口：自动选端口、查模型、等服务就绪再开浏览器（exe 的入口） |
| `build_exe.py` | 用 PyInstaller 打包成 Windows exe |
| `make_icon.py` | 从 PNG 生成 exe 用的多分辨率 `.ico` |
| `des5_ai.ico` | exe 图标（16~256 七档尺寸） |
| `start.bat` | 双击启动：找 Python、避让端口、拉起服务并自动开浏览器 |
| `extract_dataset.py` | 扫描 *.zip，抽取音符区间与 des 区间（含字符偏移） |
| `features.py` | 音符格 / 字符 n-gram / 时序 id / 123 维结构化特征 |
| `build_features.py` | 生成 `data/features.npz` |
| `train.py` | 词袋模型训练 + 留出集评估 |
| `cnn.py` | 1D-CNN 前向 / 反向（手写卷积、池化、掩码） |
| `train_cnn.py` | CNN 训练（dropout / 权重衰减 / 先验权重搜索） |
| `test_cnn_grad.py` | 有限差分梯度自检 |
| `eval_ensemble.py` | 模型对比与集成权重网格 |
| `make_package.py` | 重新打包本目录为 zip |
| `model_bag.npz`、`model_cnn.npz` | 模型权重 |
| `metrics_*.json` | 验证指标 |
| `data/dataset.jsonl` | 全量样本（每条含音符区间与 des 区间的字符偏移） |
| `data/features.npz` | 特征缓存（由 build_features.py 生成，可删除后重建） |
| `report.md` | 完整报告：方法、效果对比、局限、复现步骤 |
| `pyrightconfig.json` | 类型检查配置（同目录导入为有意写法） |

## 难度编号对照（重要）

`&des_N=` 里的 `N` 是**难度编号**，共 6 种：

| N | 难度 | 全档案条数 | `&lv_N=` 形态 | `&des_N=` 署名 | 能否验证 |
|---|---|---|---|---|---|
| 2 | BASIC | 1559 | `6.0` | 有 | 可 |
| 3 | ADVANCED | 1559 | `8.0` | 有 | 可 |
| 4 | EXPERT | 1559 | `12.7` | 有 | 可 |
| 5 | MASTER | 1559 | `13.2` | 有 | 可 |
| **6** | **Re:MASTER（remaster）** | **149** | `13.7` 普通星数，**不带 `?`** | **149/149 全部有署名** | 可 |
| **7** | **宴会場** | **92** | **`14?` / `13+?`，星数带 `?`** | **92/92 全部为空** | **不可** |

两点容易踩坑：

- **`?` 后缀是「宴会場」（N=7）的标志，不是 Re:MASTER 的。** Re:MASTER 是 N=6，星数是普通数字。
- **Re:MASTER（N=6）署名完整**，已参与训练与验证；**宴会場（N=7）全档案零署名**，无法验证，模型只能给"谁可能打这首歌"的候选。

```powershell
python predict.py "D:\MaiArchive\FiNALE" --diff 6    # 只看 Re:MASTER
python predict.py "D:\MaiArchive" --diff all        # 含 Re:MASTER 与宴会場
```

界面里同样可选：`Re:MASTER (6)`、`宴会場 (7)`，或 `全部常规（含 Re:MASTER）` / `全部难度`。结果卡片上 Re:MASTER 会显示紫色徽标，宴会場显示橙色徽标，批量汇总有分难度命中率。

## 效果（清洗后留出集，392 条 / 202 类，按歌曲分组）

| 模型 | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| 词袋 | 40.3% | 60.2% | 73.7% |
| 1D-CNN | 33.2% | 56.1% | 68.6% |
| 集成（默认） | 39.8% | 60.2% | 73.5% |

**分难度 Top-1（留出集）**：EXPERT（`--diff 4`）46.5% → MASTER（`--diff 5`）26.2% → **Re:MASTER（`--diff 6`）16.7%**。
Re:MASTER 明显最难：留出集仅 18 条，且谱师高度集中（はっぴー 28/149、Jack 16、サファ太 14），长尾谱师的 Re:MASTER 谱风格与 MASTER 差异大。详见 `report.md`。

## 注意事项

- 标签空间**只含真实署名**（202 位谱师）；`&des_N=` 为 `-` 或空的文件不在标签空间内，界面会显式提示且不计命中率。宴会場（N=7）92 条全部属于这一类。
- 模型只使用音符区间文本，**不含**曲名 / 版本 / 星数 / 目录名。
- 多数谱包的目录名就等于谱师全名（85.5%）；若只是补全署名，直接读目录名更准，模型用于目录名缺失或改名的情况。
- Re:MASTER（N=6）已完整参与训练；宴会場（N=7）因全档案无署名，从未进入训练集，其预测结果无法验证对错。
