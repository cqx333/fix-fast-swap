# RNG 修复后最小必要重算包（不含相图）

日期：2026-10-20。本包只更新这次 rank-local RNG 修改影响的并行 `fast/swap` 证据。无需覆盖旧 bundle，也无需再编译一次 LAMMPS；使用已经通过运行标识检查的新版 `build-fast/lmp_mpi`。

## 1. 计算哪些，不计算哪些

| 类别 | 本包新计算 | 保持不变的对应证据 |
|---|---:|---|
| 静态并行一致性 | 45 条 `fast/swap` 轨迹 | 原版相同条件下的轨迹 |
| 双核 MD/MC | 6 条长轨迹，每条 6 个 100-sweep 分段，共 36 段 | 原版双核、两种实现单核及 dt/2 控制、纯 MD 制备态 |
| 原性能条件下的并行测试 | 75 次正式计时 | 原版计时、单核计时、公共初态 |
| 三维 LJ 尺寸扫描 | 10 尺寸 × 5 次重复，共 50 次正式计时 | 50 次原版计时和 10 个单核制备态 |
| 已完成 RNG 回归 | 不重算，直接归档 | 本次 9 个回归任务，包括解析零能量控制和三维 LJ 检验 |

合计 **206 个新生产任务**，全部是 **2--24 核 `fast/swap`**。其中 36 段 MD/MC 是 6 条链的检查点分段，**不是 36 个独立副本**。125 次计时另外有自动 pilot 和必要的计时长度校准；这些不计入正式重复次数。

**明确不重算：**任何 `atom/swap`、任何单核模拟、能量闭合、势参数/`pair_modify` 选项测试、拒绝测试、已被三维扫描取代的纯扁平 LJ 尺寸扫描，以及全部相图应用。现有 636 条历史记录随包保留来源信息和需要的日志，不冒充新版程序生成的结果。

源码快照 `src/MC/fix_fast_swap.cpp`、`.h` 只是已部署 RNG 补丁的归档，不包含第二次算法修改。不要因为上传本包再次覆盖/编译。

### 静态并行任务

- 11 个模型各 3 seed、24 核，共 33 条：LJ/cut、unequal-cutoff LJ/cut、ternary LJ/cut、Morse、tabulated LJ、Buckingham、Buckingham+Coul/cut、Coul/DSF、LJ+Yukawa overlay、Yukawa、ZBL。
- LJ/cut 和 Morse 各追加 2 核与 4 核、各 3 seed，共 12 条。
- 默认每条 600 sweeps；24 核 ternary/ZBL 各 3 条直接从原始初态跑到 1200 sweeps。
- 静态盒子保持原验证条件 `[4,4,96]`，24 核网格 `1×1×24`；实际完整三维回归已经在 `reused_rng_regression/` 中，不再重复。
- 每 sweep 输出能量、成分和实际 MC 计数；每 25 sweep 输出逐原子类型，另存最终坐标/类型。

旧 ternary/ZBL 的前 600 sweeps 或其并行末态也可能受旧随机流影响，**本包不从这些旧末态继续，不把新样本接在旧 fast 样本后面统计**。新的完整 1200-sweep 链与旧原版的完整链进行统计对照，不要求轨迹一致。

### 性能与初态

- 原性能任务保持原模型、盒子、MPI 网格、重复种子和公共制备构型；只更新受影响的 fast 一侧。
- 保留 LJ 固定 `8×8×96` 盒子的 2/4/8/12/16/24 核 strong scaling、11 模型的 24 核比较，以及 Morse/Buckingham+Coul/cut/Yukawa/ZBL 的 2/8/24 核比较。相同条件在多张图中出现时只算一次。
- 三维尺寸扫描保持 `2×3×4` 网格和之前 10 个三维盒子，6144--32928 原子；同一尺寸的 5 个正式重复使用相同公共初态。
- 所有新计时的生产 Loop 至少 30 秒，避免短计时干扰。原性能组仍为 3 次重复，三维尺寸组仍为 5 次重复。旧原版较短的计时区间如实保留，不伪称也满足新 30 秒标准。
- 归一化依据 **实际 `f_mc[1]` 增量/N**，不是凭 `run` 步数猜测。pilot、warmup、启动和写盘不计入正式 Loop。
- 性能数据需要与原版在同一节点比较，因此本包强制计时在 **node05** 执行；默认全部顺序运行，最多同时 24 核。
- 默认复用各组原 MPI 启动参数：旧性能/轨迹使用 `mpirun`，三维扫描使用 `mpirun -bind-to core`。显式 `--mpi-launcher` 会覆盖这些参数，通常不要设置，也应先 `unset MPI_LAUNCHER`，避免环境变量无意覆盖。
- 本包不能替你独占节点或阻止其他用户使用 CPU；每个计时任务开始前保存节点负载和进程占用，便于事后审查。请尽量在低负载时段运行。

## 2. 上传后的一条全量命令

把整个文件夹或上传 ZIP 放在 node05，例如 `/home/chenqx/CPC-test_rng_recompute_20261020`。**不需要把旧 results 再复制进来，也不需要传旧目录参数。**必要的原版日志、公共初态、MD restart、tabulated 势文件已经归档在包内。

```bash
conda activate lmp-fast
cd /home/chenqx/CPC-test_rng_recompute_20261020
LMP=/home/chenqx/LAMMPS-opt/build-fast/lmp_mpi
export LD_LIBRARY_PATH="${LMP%/*}:${LD_LIBRARY_PATH:-}"
unset MPI_LAUNCHER
python -m pip install -r requirements.txt
bash run_all.sh "$LMP"
```

这是 CLI 工作流，**不需要 `mpi4py` 或 LAMMPS Python 接口**。只需要当前环境中的 NumPy、SciPy，以及与已编译 LAMMPS 匹配的 MPI 启动器。不要混用 Intel MPI 和此 `lmp-fast` 的 conda MPI 库。

已验证新动态库 SHA-256 为：

```text
5fa13058c965a2f0ffec776343acce4bc021a5f6988ad2937a82502b8cf2156d
```

可以将核查进一步锁定到该文件：

```bash
bash run_all.sh "$LMP" \
  --expected-library-sha256 5fa13058c965a2f0ffec776343acce4bc021a5f6988ad2937a82502b8cf2156d \
  --cmake-cache /home/chenqx/LAMMPS-opt/build-fast/CMakeCache.txt
```

不提供 hash 时也会核查实际动态库路径、拒绝已知旧库，并要求**每个新任务**打印 RNG 修复运行标识。不会再把 ELF 中找不到连续字符串误判为旧程序。没有额外单核 smoke 模拟。

开始前会按各组 MPI 参数运行轻量的 Python rank/CPU 亲和性探针，确认计算没有跨节点、可用逻辑 CPU 数不少于 rank 数。探针不运行 LAMMPS，不是额外 MC 算例，也不能代替节点独占。

## 3. 先核查或分批运行

不启动模拟的完整任务检查：

```bash
bash run_all.sh "$LMP" --dry-run
```

先获得一致性结果，再补性能：

```bash
bash run_all.sh "$LMP" --suite parallel,mdmc
bash run_all.sh "$LMP" --suite timing,size3d
```

所有结果默认写入本包 `results/`，上述命令可依次执行。不在同一输出目录同时开两个任务；这里没有自动并行跑多例，也不包含多节点分片。

只补指定模型/盒子，例如：

```bash
bash run_all.sh "$LMP" --suite parallel --ids morse,ternary_lj_cut
bash run_all.sh "$LMP" --suite size3d --ids 12x18x24
bash run_all.sh "$LMP" --suite mdmc --ids morse_dynamic
```

`--ids` 是包含匹配，不是正则表达式。MD/MC 后段被选中时，会自动加入其前面的**新版**分段依赖，不会使用旧版 MC checkpoint。

## 4. 断点重算

在**相同节点、环境、文件夹和输出目录**重跑原命令即可。

- 已完成任务只有在输入、库、脚本、依赖和输出哈希一致时才跳过。
- 静态轨迹和单次计时按任务恢复；中断的当前任务从头执行，不伪称恢复其 RNG 状态。
- MD/MC 每 100 sweeps 保存 restart；已完成的新分段保留，只补后续新分段。
- 更换二进制/脚本/节点后不会自动把新旧结果混入同一输出。需要新建 `--output results_new`。
- 文件完整性失败不会自动覆盖已有成功结果。确认需要重做时，对明确选中的任务加 `--force`；不要无意对整包使用该选项。
- 遗留 `results/workflow.lock` 只能在确认相关 Python 和 MPI/LAMMPS 都退出后删除。正常结束或 Ctrl+C 会清理；突然断电时可能遗留。

示例：

```bash
bash run_all.sh "$LMP" --suite size3d --ids 14x21x28 --force
```

## 5. 输出、分析与下载

计算结束后自动分析，也可以不启动模拟重新分析：

```bash
python scripts/workflow.py --analyze-only
bash pack_results.sh
```

下载上一级生成的 `CPC-test_rng_recompute_20261020_results.zip`。不要只下载均值表：需要保留原始日志、逐 snapshot 数据和程序来源，才能更新正文/SI图表。

主要输出：

- `results/run_status.json`：206 项的状态、真实尝试数、实际网格、实际 ghost cutoff、运行标识和哈希。
- `results/inputs/`、`logs/`、`raw/`：输入、完整日志、类型快照、最终构型和 MD restart；pilot/各次校准也保留。
- `results/metadata/`：实际加载的库及 hash、MPI 参数、CPU 信息、计时前负载/进程记录。
- `results/analysis/timeseries/`：新 fast 与复用 atom 的能量/各组元成分、原子计数、接受计数；MD/MC 按真实 MC 时刻对齐，分段只拼接**同一新版本链**。
- `statistical_comparisons.csv`：舍弃前 25%/50%后的均值差、相关性修正区间、成分差对应原子数，以及方差比的 block-8/block-32 bootstrap 区间。
- `replica_diagnostics.csv`：各 seed 的 ESS、方差、漂移与区间，不把 MC snapshot 当作独立重复。
- `timing_raw.csv`、`timing_summary.csv`：新 fast 和复用 atom 的真实 sweep 耗时、均值/SD/CV/区间、ratio-of-means 加速比。
- `summary.json`：缺失、失败、哈希异常和统计审阅提示。

**重要：现有旧 24 核 atom 参考与独立 Gibbs 对照之间仍有方差疑点。因此本包保留原版数据，但不把“原版”名称视作无条件分布金标准。**自动分析不只比较均值，也比较方差；出现差异会保留 `REVIEW`，不自动判新 fast 错误，也不放宽阈值强行通过。即使所有任务完成，最终 ensemble 可靠性结论仍需结合本次解析控制、Gibbs 对照与新多势数据审阅。

本包尊重本次“不要重算原版和串行”的范围，不悄悄添加此前建议的单核 6144 原子对照。当前 9 项新 RNG 回归保存在 `reused_rng_regression/`，不再次计算。

## 6. 预计成本与本地验证

125 次正式性能计时的 30 秒下限合计为 **62.5 分钟**；此外还有校准、warmup、轨迹、MPI启动和输出。实际运行约需小时量级，不能用旧锁步 fast 的耗时承诺新版速度；负载或 unequal-cutoff 开销可能使总时间增加。全部逐项执行，不同时启动 206 例。

本包本地验证命令：

```bash
python -m unittest discover -s tests -v
```

本地检查覆盖任务范围/数量、几何余量、原版与串行排除、旧初态 hash、MD依赖、计时替换、原子计数和拒绝 fallback 的逻辑；Windows 本机没有匹配的集群 MPI/LAMMPS，**没有声称已在本机执行这 206 个新生产任务**。
