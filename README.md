# 标注数据质量检查工具（label-qa）

一行命令查出标注数据里的常见错误，训练模型之前先把脏数据挑出来。

支持 **YOLO**（`.txt`，0~1 归一化坐标）与 **Pascal VOC**（`.xml`，绝对像素坐标）两种格式。

```
python check_labels.py --demo
```

```
  图片        : 19 张
  标注文件    : 19 个
  标注框      : 44 个
  问题总数    : 8 处
--------------------------------------------------------------
  级别    次数   规则
  警告      2   COORD_OUT_OF_BOUNDS
  错误      1   LABEL_WITHOUT_IMAGE
  警告      1   IMAGE_WITHOUT_LABEL
  错误      1   COORD_NOT_NORMALIZED
  错误      1   BOX_DEGENERATE
  警告      1   DUP_BOX
  错误      1   CLS_OUT_OF_RANGE
```

---

## 为什么做这个

标注质量决定模型效果的上限。人工拉框最容易出的错就那么几类：框偏了、漏标、
同一个目标标了两个框、类别 id 写错、把像素坐标当成归一化坐标写进 txt……
这些错误在训练前用脚本几十秒就能查完，靠人翻图要几个小时，而且会漏。

做过标注的人最清楚人会在哪里出错——这个工具就是把这些"人容易错的地方"写成规则。

## 能查什么

| 规则码 | 级别 | 含义 |
| --- | --- | --- |
| `CLS_OUT_OF_RANGE` | 错误 | 类别 id 超出 `classes.txt` 的范围（例如 4 类却写了 `id=9`） |
| `CLS_DUP_NAME` / `CLS_CASE_CONFLICT` | 警告 | 类别名重复，或只有大小写不同（`helmet` / `Helmet`） |
| `COORD_NOT_NORMALIZED` | 错误 | YOLO 坐标出现 >1 或 <0 的值，通常是误把像素坐标写进去了 |
| `COORD_OUT_OF_BOUNDS` | 警告 | 框越出图像边界（x2 > 1 或 y2 > 1） |
| `BOX_DEGENERATE` | 错误 | 宽或高 ≤ 0，框退化成一条线 |
| `BOX_TOO_SMALL` | 警告 | 框面积接近 0 |
| `DUP_BOX` | 警告 | 同图同类、两个框 IoU > 阈值（默认 0.9），疑似重复标注 |
| `LABEL_WITHOUT_IMAGE` | 错误 | 有标注文件但没有同名图片 |
| `IMAGE_WITHOUT_LABEL` | 警告 | 有图片但没有标注文件（负样本可忽略） |
| `IMAGE_UNREADABLE` | 错误 | 图片打不开/损坏 |
| `EMPTY_LABEL_FILE` | 警告 | 标注文件是空的 |
| `FMT_FIELD_COUNT` / `FMT_NOT_NUMBER` | 错误 | 行字段数不对、含非数字 |
| `SIZE_MISMATCH` | 警告 | VOC：XML 里写的 width/height 与图片实际尺寸不一致 |
| `BBOX_ORDER` | 错误 | VOC：xmin ≥ xmax 或 ymin ≥ ymax |
| `BBOX_NEGATIVE` | 错误 | VOC：坐标出现负数 |
| `BBOX_OUT_OF_IMAGE` | 警告 | VOC：框越出图片 |
| `CLS_UNKNOWN_NAME` | 警告 | VOC：类别名不在 `classes.txt` 里（拼写/大小写问题） |
| `CLASS_WITHOUT_BOX` | 警告 | 某个类别在整个数据集里一个框都没有 |

红色框 = 有问题的框，绿色框 = 正常框，抽查图直接能截图说明问题。

## 快速开始

```bash
pip install pillow
```

**1）先跑一遍内置演示**（会造一份故意埋了 6 类错误的假数据集，用来验证脚本能查出来）

```bash
python check_labels.py --demo
```

**2）检查自己的 YOLO 数据集**

```bash
python check_labels.py --format yolo \
    --images dataset/images \
    --labels dataset/labels \
    --classes dataset/classes.txt \
    --viz 20 --out report
```

`labels/` 里是 `xxx.txt`，与 `images/xxx.jpg` 同名；`classes.txt` 一行一个类别名，行号即类别 id。

**3）检查 Pascal VOC 数据集**

```bash
python check_labels.py --format voc \
    --images dataset/JPEGImages \
    --labels dataset/Annotations \
    --classes dataset/classes.txt \
    --viz 20 --out report
```

## 输出

| 文件 | 内容 |
| --- | --- |
| `report/report.json` | 完整报告：总数、按规则统计、按类别统计、每条问题明细 |
| `report/issues.csv` | 每条问题一行（Excel 直接打开，可筛选后派给标注员返修） |
| `report/stats_by_class.csv` | 每类框数 / 图片数 |
| `report/viz/*.png` | 抽查图，文件名带问题类型，例：`demo_007__dup-box.png` |

## 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--demo` | 关 | 生成演示数据集并检查 |
| `--format` | `yolo` | `yolo` 或 `voc` |
| `--images` / `--labels` | — | 图片目录 / 标注目录 |
| `--classes` | 无 | 类别表；不传则跳过类别白名单检查 |
| `--iou` | `0.9` | 判为重复框的 IoU 阈值 |
| `--viz` | `12` | 生成多少张可视化抽查图，`0` 表示不生成 |
| `--out` | `report` | 报告输出目录 |
| `--seed` | `0` | 抽查随机种子 |
| `--fail-on-error` | 关 | 存在 error 级问题时以退出码 1 结束，可接 CI / 训练前置校验 |

## 面试怎么讲（30 秒版）

> 我做过图像标注，知道人最容易在哪出错——漏标、框偏、同一目标标两个框、
> 类别 id 写错。所以我写了个脚本，把这些错误按规则自动查出来：重复框用 IoU 判，
> 未归一化的坐标用阈值判，有标注没图片这种直接比对文件名；最后输出统计报表和抽查图。
> 以前靠人翻图两小时能查完的，现在一分钟出报告。

## 目录结构

```
label-qa/
├── check_labels.py      # 主程序（单文件，无第三方依赖除了 Pillow）
├── requirements.txt
├── README.md
├── demo_data/           # --demo 生成，可删
└── demo_output/         # --demo 的报告与抽查图，可删
```

## 还可以继续加

- 导出一份 COCO 格式的体检报告
- 多人标注一致性检查（同一张图两个人标的框做 IoU 比对）
- 把 `--fail-on-error` 接进训练脚本，脏数据直接拦住不让开训

---

作者：陈力嘉　｜　Python + Pillow，单文件，无第三方依赖（除 Pillow）
