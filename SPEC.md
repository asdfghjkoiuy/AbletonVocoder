# AbletonVocoder v2 规格

目标：REAPER JSFX 声码器，行为与 Ableton Live Vocoder（Live 12 手册 29.42 节）一致，音质对标 Ableton / Bitwig。
`AbletonVocoder_v1.jsfx` 是上游原版，只作参照，不要修改。新版写在 `AbletonVocoder.jsfx`。

## 1. 引脚

```
in_pin:Modulator L
in_pin:Modulator R
in_pin:Carrier L (sidechain)
in_pin:Carrier R (sidechain)
out_pin:Output L
out_pin:Output R
```

## 2. 滑块（索引和名字固定，测试脚本按索引设置）

| # | 名字 | 范围 / 选项 | 默认 |
| :-- | :-- | :-- | :-- |
| 1 | Carrier | {Noise, External, Modulator, Pitch Tracking} | 0 |
| 2 | Bands | {4,8,12,16,20,24,28,32,36,40} | 4（=20 带） |
| 3 | Range Low (Hz) | 20..2000 | 80 |
| 4 | Range High (Hz) | 1000..20000 | 18000 |
| 5 | BW (%) | 10..200 | 100 |
| 6 | Filter Mode | {Precise, Retro} | 0 |
| 7 | Gate (dB) | -96..0（-96 = 关） | -96 |
| 8 | Level (dB) | -24..24 | 0 |
| 9 | Depth (%) | 0..200 | 100 |
| 10 | Attack (ms) | 0.1..100 | 5 |
| 11 | Release (ms) | 1..1000 | 80 |
| 12 | Formant (st) | -12..12 | 0 |
| 13 | Dry/Wet (%) | 0..100 | 100 |
| 14 | Channels | {Mono, Stereo, L/R} | 1 |
| 15 | Enhance | 0/1 | 0 |
| 16 | Unvoiced (dB) | -96..0（-96 = 关） | -96 |
| 17 | Sens (%) | 0..100 | 50 |
| 18 | Unvoiced Speed | {Slow, Fast} | 0 |
| 19 | Noise Rate | 0..1（1 = 满采样率） | 1 |
| 20 | Noise Crackle | 0..1（1 = 连续白噪声） | 1 |
| 21 | Osc Wave | {Saw, Pulse 50%, Pulse 25%, Pulse 12.5%} | 0 |
| 22 | Osc Pitch (st) | -24..24 整数 | 0 |
| 23 | Track Low (Hz) | 20..1000 | 60 |
| 24 | Track High (Hz) | 100..2000 | 1000 |
| 25 | Device On | 0/1 | 1 |

逐带衰减（手册：点击显示区可衰减各带）：40 个增益存在内存里，用 `@serialize` 持久化，默认全 0 dB。

## 3. 手册规定的行为（必须满足）

- Carrier
  - Noise：内置噪声。Rate 横轴 = 降采样（采样保持），越小采样率越低；Crackle 纵轴 = 密度，越小越稀疏（随机脉冲）。
  - External：载波来自 3/4 声道。
  - Modulator：载波就是调制信号本身（配合 Enhance + Depth 100% = 共振峰移位器）。
  - Pitch Tracking：单音振荡器跟随调制信号音高。Track Low/High 限定跟踪范围；Osc Wave 选锯齿或三种脉冲；Osc Pitch 粗调。**只在检测到清晰音高时更新频率，否则保持上一音高。**
- Enhance：对载波的**频谱和动态做归一化**，声音更亮、音量更稳。
- Unvoiced：额外噪声发生器的音量，用来重建无音高部分（f、s）。Sens：100% 噪声一直开，0% 只用主载波。Fast/Slow：清浊判断切换的快慢。
- Bands：滤波器个数。Range：频带覆盖范围。BW：100% 最准确；越低越接近单频，越高重叠越多。
- Precise：所有带增益、带宽相同。Retro：越高的带越窄、越响。
- Gate：低于阈值的频带静音。
- Depth：0% 丢弃调制包络；100% 经典；200% 只保留高峰。
- Attack / Release：调制包络响应速度。很快的值保留瞬态，但可能失真。
- Mono：载波和调制都当单声道。Stereo：调制单声道，载波立体声。L/R：载波和调制都立体声（左右各一台声码器）。
- Formant：载波滤波器组的中心频率整体上移或下移。
- Dry/Wet、Level：常规。

## 4. DSP 实现要求（最好实践）

- 参数派生量放在 `@block` 里做变化检测再重算。`@gfx` 里改变量不会触发 `@slider`，不能依赖 `@slider`。
- 频带中心按对数等距。Q 由相邻带间距（八度）× BW 推出，保证 BW=100% 时相邻带在 -3 dB 处交接，Precise 模式总响应平坦。
- 滤波器用 TPT/SVF（Zavalishin/Cytomic），两级串联（4 阶带通）。合成侧系数可随 Formant 平滑变化而不爆音。
- 包络检测：推荐利用 SVF 的带通与低通输出构成近似正交对求瞬时包络（或平方律 + 平滑），压低 2 倍频纹波。Attack/Release 按界面数值如实生效，不要乘隐藏系数。
- Enhance：每个载波频带各自做功率跟随，再除以该包络（带底噪下限防止放大噪声）。
- Unvoiced：用高频/低频能量比（可加过零率）判断，不用绝对电平。噪声注入合成滤波器组的输入，由调制包络整形。
- Pitch Tracking：YIN（或 MPM），在 `@block` 中对降采样后的单声道调制信号运算，跳步 ≤ 5 ms。带置信度阈值 + 能量门限，不清晰就保持。频率滑动在对数域做。振荡器用 PolyBLEP 抗混叠。
- Gate：作用在包络（dB）上，带约 3 dB 回差和几毫秒的增益平滑，不能在波形上逐采样清零。
- 增益标定：Carrier=Modulator + Enhance 开 + Depth 100% 时，输出响度和长期频谱应接近输入（重合成）。频带数从 4 到 40 切换，输出响度变化小。
- 安全：无 NaN / Inf，防 denormal，静音进静音出，参数快速自动化不爆音。Device On 关闭时约 10 ms 交叉淡化到干声。
- 性能：40 带 L/R 模式在 48 kHz 下 CPU 可接受（离线渲染 ≥ 10 倍实时）。

## 5. 界面

保留 Ableton 风格界面，覆盖全部滑块：Noise 时显示 Rate/Crackle 的 X-Y 区；Pitch Tracking 时显示 Wave、Pitch、Low/High；中央显示各带电平，可点击/拖动衰减单带；窗口可缩放。

## 6. 验证

`tools/` 下是离线测试框架（ysfx 无头渲染 + Python 指标）。`python3 tools/suite.py` 输出各项指标并与通过线比较。
