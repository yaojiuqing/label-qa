#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注数据质量检查工具（label QA）
--------------------------------
支持 YOLO（.txt，归一化坐标）与 Pascal VOC（.xml，绝对像素坐标）两种标注格式，
自动检查 6 类常见标注错误，输出统计报表与可视化抽查图。

示例：
    # 1) 先跑一遍内置演示数据（会造出一份带错误的假数据集）
    python check_labels.py --demo

    # 2) 检查自己的 YOLO 数据集
    python check_labels.py --format yolo \
        --images dataset/images --labels dataset/labels \
        --classes dataset/classes.txt --viz 20 --out report

    # 3) 检查 Pascal VOC 数据集
    python check_labels.py --format voc \
        --images dataset/JPEGImages --labels dataset/Annotations \
        --classes dataset/classes.txt --viz 20 --out report

输出（--out 目录）：
    report.json        完整问题清单
    issues.csv         每条问题一行，方便筛
    stats_by_class.csv 每类框数 / 图片数
    viz/*.png          抽查图（红框 = 有问题的框，绿框 = 正常）

依赖：Python 3.8+、Pillow（pip install pillow）
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    sys.exit("缺少 Pillow：请先执行  pip install pillow")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

ERROR = "error"
WARN = "warn"
INFO = "info"


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def iou(a, b):
    """两个 xyxy 框的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def read_classes(path):
    """读取 classes.txt；返回 [(name, 是否重复/大小写冲突说明)] 与白名单小写集合。"""
    if not path:
        return [], set()
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"找不到类别文件：{p}")
    names = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return names, {n.lower() for n in names}


def list_images(folder):
    folder = Path(folder)
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
    ) if folder.is_dir() else []


def list_labels(folder, suffix):
    folder = Path(folder)
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() == suffix
    ) if folder.is_dir() else []


def add_issue(issues, level, code, file, detail, extra=None):
    item = {"level": level, "code": code, "file": str(file), "detail": detail}
    if extra:
        item.update(extra)
    issues.append(item)


# --------------------------------------------------------------------------
# YOLO 解析
# --------------------------------------------------------------------------
def parse_yolo_label(path, n_classes):
    """返回 (boxes, problems)。boxes: [{cls, cx, cy, w, h, line}]"""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    boxes, problems = [], []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append(("FMT_FIELD_COUNT", f"第 {lineno} 行字段数 {len(parts)}，应为 5（class cx cy w h）"))
            continue
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            problems.append(("FMT_NOT_NUMBER", f"第 {lineno} 行含非数字：{line}"))
            continue
        cls = vals[0]
        if abs(cls - round(cls)) > 1e-6:
            problems.append(("CLS_NOT_INT", f"第 {lineno} 行类别 {cls} 不是整数"))
        cls = int(round(cls))
        cx, cy, w, h = vals[1:5]
        boxes.append({"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h, "line": lineno})
        if cls < 0 or (n_classes and cls >= n_classes):
            problems.append(("CLS_OUT_OF_RANGE", f"第 {lineno} 行类别 id={cls}，超出 0~{n_classes - 1}"))
    return boxes, problems


def yolo_to_xyxy(box):
    cx, cy, w, h = box["cx"], box["cy"], box["w"], box["h"]
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


# --------------------------------------------------------------------------
# VOC 解析
# --------------------------------------------------------------------------
def parse_voc_xml(path):
    """返回 (size, objects, problems)。size: (w,h) 或 None"""
    import xml.etree.ElementTree as ET

    problems = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return None, [], [("XML_PARSE_ERROR", f"XML 解析失败：{exc}")]

    size = None
    for node in root.findall("size"):
        try:
            size = (int(float(node.findtext("width", "0"))),
                    int(float(node.findtext("height", "0"))))
        except ValueError:
            problems.append(("SIZE_INVALID", "size 里的 width/height 不是数字"))
    objects = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bnd = obj.find("bndbox")
        if bnd is None:
            problems.append(("BBOX_MISSING", f"类别 {name} 缺少 bndbox"))
            continue
        try:
            box = [float(bnd.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax")]
        except ValueError:
            problems.append(("BBOX_INVALID", f"类别 {name} 的坐标不是数字"))
            continue
        if any(math.isnan(v) for v in box):
            problems.append(("BBOX_INVALID", f"类别 {name} 的坐标缺失"))
            continue
        objects.append({"name": name, "xyxy": box})
    return size, objects, problems


# --------------------------------------------------------------------------
# 检查：YOLO
# --------------------------------------------------------------------------
def check_yolo(images_dir, labels_dir, classes_file, iou_thr, issues, tol=1e-3):
    class_names, class_lower = read_classes(classes_file)
    n_classes = len(class_names)

    # classes.txt 自身问题
    seen, seen_lower = {}, {}
    for idx, name in enumerate(class_names):
        if name in seen:
            add_issue(issues, WARN, "CLS_DUP_NAME", classes_file,
                      f"类别名重复：{name}（第 {seen[name] + 1} 与第 {idx + 1} 行）")
        seen.setdefault(name, idx)
        low = name.lower()
        if low in seen_lower:
            add_issue(issues, WARN, "CLS_CASE_CONFLICT", classes_file,
                      f"类别名仅大小写不同：{class_names[seen_lower[low]]} / {name}")
        seen_lower.setdefault(low, idx)

    images = list_images(images_dir)
    labels = list_labels(labels_dir, ".txt")
    image_stems = {f.stem for f in images}
    label_stems = {f.stem for f in labels}

    for stem in sorted(label_stems - image_stems):
        add_issue(issues, ERROR, "LABEL_WITHOUT_IMAGE", f"{labels_dir}/{stem}.txt",
                  "有标注文件但找不到同名图片")
    for stem in sorted(image_stems - label_stems):
        add_issue(issues, WARN, "IMAGE_WITHOUT_LABEL", f"{images_dir}/{stem}.*",
                  "有图片但没有标注文件（负样本可忽略）")

    boxes_by_image = {}
    per_class = {i: {"boxes": 0, "images": 0} for i in range(max(n_classes, 0))}
    unknown_class = {}
    total_boxes = 0
    image_problems = {}

    for img in images:
        label_path = Path(labels_dir) / f"{img.stem}.txt"
        try:
            with Image.open(img) as im:
                im.verify()
        except Exception as exc:
            add_issue(issues, ERROR, "IMAGE_UNREADABLE", img, f"图片无法打开：{exc}")
            continue
        if not label_path.exists():
            continue

        text = label_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            add_issue(issues, WARN, "EMPTY_LABEL_FILE", label_path, "标注文件为空（负样本可忽略）")
            continue

        boxes, problems = parse_yolo_label(label_path, n_classes)
        for code, detail in problems:
            add_issue(issues, ERROR if code.startswith(("CLS_", "FMT_")) else WARN, code, label_path, detail)
            image_problems.setdefault(img.name, []).append(code)

        named = []
        for box in boxes:
            total_boxes += 1
            cid = box["cls"]
            if 0 <= cid < n_classes:
                per_class[cid]["boxes"] += 1
            else:
                unknown_class[cid] = unknown_class.get(cid, 0) + 1

            x1, y1, x2, y2 = yolo_to_xyxy(box)
            tag = f"第 {box['line']} 行"
            bad = not (0 <= cid < n_classes)

            # 未归一化 / 越界 / 退化
            if max(box["cx"], box["cy"], box["w"], box["h"]) > 1 + tol or \
               min(box["cx"], box["cy"], box["w"], box["h"]) < -tol:
                add_issue(issues, ERROR, "COORD_NOT_NORMALIZED", label_path,
                          f"{tag} 坐标疑似不是 0~1 归一化值：cx={box['cx']}, cy={box['cy']}, "
                          f"w={box['w']}, h={box['h']}")
                image_problems.setdefault(img.name, []).append("COORD_NOT_NORMALIZED")
                bad = True
            if box["w"] <= 0 or box["h"] <= 0:
                add_issue(issues, ERROR, "BOX_DEGENERATE", label_path,
                          f"{tag} 宽或高 <= 0（w={box['w']}, h={box['h']}）")
                image_problems.setdefault(img.name, []).append("BOX_DEGENERATE")
                bad = True
            elif box["w"] * box["h"] < 1e-6:
                add_issue(issues, WARN, "BOX_TOO_SMALL", label_path,
                          f"{tag} 框面积过小（w={box['w']}, h={box['h']}）")
                bad = True
            if x1 < -tol or y1 < -tol or x2 > 1 + tol or y2 > 1 + tol:
                add_issue(issues, WARN, "COORD_OUT_OF_BOUNDS", label_path,
                          f"{tag} 框越出图像边界：[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
                image_problems.setdefault(img.name, []).append("COORD_OUT_OF_BOUNDS")
                bad = True

            named.append({"cls": cid, "xyxy": (x1, y1, x2, y2), "line": box["line"], "box": box, "bad": bad})

        # 重复框（同类，IoU > 阈值）
        for i in range(len(named)):
            for j in range(i + 1, len(named)):
                a, b = named[i], named[j]
                if a["cls"] != b["cls"]:
                    continue
                score = iou(a["xyxy"], b["xyxy"])
                if score > iou_thr:
                    for k in (i, j):
                        named[k]["bad"] = True
                    add_issue(issues, WARN, "DUP_BOX", label_path,
                              f"第 {a['line']} 行与第 {b['line']} 行疑似重复框"
                              f"（同类 {a['cls']}，IoU={score:.2f} > {iou_thr}）")
                    image_problems.setdefault(img.name, []).append("DUP_BOX")

        if named:
            for cid in {n["cls"] for n in named if 0 <= n["cls"] < n_classes}:
                per_class[cid]["images"] += 1

        boxes_by_image[img.name] = named

    return {
        "images": len(images),
        "labels": len(labels),
        "boxes": total_boxes,
        "per_class": per_class,
        "class_names": class_names,
        "unknown_class": unknown_class,
        "image_problems": image_problems,
        "boxes_by_image": boxes_by_image,
    }


# --------------------------------------------------------------------------
# 检查：VOC
# --------------------------------------------------------------------------
def voc_label_key(path):
    """a.jpg.xml / a.xml 都归一到 a，便于与图片同名匹配。"""
    stem = Path(path).stem
    low = stem.lower()
    for suf in IMAGE_SUFFIXES:
        if low.endswith(suf):
            return stem[: -len(suf)]
    return stem


def check_voc(images_dir, labels_dir, classes_file, iou_thr, issues, tol=1.0):
    class_names, class_lower = read_classes(classes_file)
    images = list_images(images_dir)
    labels = list_labels(labels_dir, ".xml")
    label_by_key = {}
    for p in labels:
        label_by_key.setdefault(voc_label_key(p), p)
    image_stems = {f.stem for f in images}
    label_stems = set(label_by_key)

    for stem in sorted(label_stems - image_stems):
        add_issue(issues, ERROR, "LABEL_WITHOUT_IMAGE", label_by_key[stem],
                  "有标注文件但找不到同名图片")
    for stem in sorted(image_stems - label_stems):
        add_issue(issues, WARN, "IMAGE_WITHOUT_LABEL", f"{images_dir}/{stem}.*",
                  "有图片但没有标注文件（负样本可忽略）")

    boxes_by_image = {}
    per_class, unknown_name = {}, {}
    total_boxes = 0
    image_problems = {}

    for img in images:
        xml_path = label_by_key.get(img.stem)
        if xml_path is None or not xml_path.exists():
            continue
        try:
            with Image.open(img) as im:
                real_size = im.size
        except Exception as exc:
            add_issue(issues, ERROR, "IMAGE_UNREADABLE", img, f"图片无法打开：{exc}")
            continue

        size, objects, problems = parse_voc_xml(xml_path)
        for code, detail in problems:
            add_issue(issues, ERROR, code, xml_path, detail)

        if size and size != real_size:
            add_issue(issues, WARN, "SIZE_MISMATCH", xml_path,
                      f"XML 里 width/height={size}，图片实际={real_size}")
            image_problems.setdefault(img.name, []).append("SIZE_MISMATCH")

        w_img, h_img = size if size and size[0] and size[1] else real_size
        named = []
        for obj in objects:
            total_boxes += 1
            name = obj["name"]
            x1, y1, x2, y2 = obj["xyxy"]
            bad = False
            if class_lower and name.lower() not in class_lower:
                unknown_name[name] = unknown_name.get(name, 0) + 1
                add_issue(issues, WARN, "CLS_UNKNOWN_NAME", xml_path,
                          f"类别名“{name}”不在 classes.txt 里（检查拼写/大小写）")
                image_problems.setdefault(img.name, []).append("CLS_UNKNOWN_NAME")
                bad = True
            per_class[name] = per_class.get(name, {"boxes": 0, "images": 0})
            per_class[name]["boxes"] += 1

            if x2 <= x1 or y2 <= y1:
                add_issue(issues, ERROR, "BBOX_ORDER", xml_path,
                          f"类别 {name} 的框 xmin/xmax 或 ymin/ymax 顺序错误：{[x1, y1, x2, y2]}")
                image_problems.setdefault(img.name, []).append("BBOX_ORDER")
                bad = True
            if x1 < 0 or y1 < 0:
                add_issue(issues, ERROR, "BBOX_NEGATIVE", xml_path,
                          f"类别 {name} 的坐标出现负数：{[x1, y1, x2, y2]}")
                image_problems.setdefault(img.name, []).append("BBOX_NEGATIVE")
                bad = True
            if x1 < -tol or y1 < -tol or x2 > w_img + tol or y2 > h_img + tol:
                add_issue(issues, WARN, "BBOX_OUT_OF_IMAGE", xml_path,
                          f"类别 {name} 的框越出图片 {w_img}x{h_img}：{[x1, y1, x2, y2]}")
                image_problems.setdefault(img.name, []).append("BBOX_OUT_OF_IMAGE")
                bad = True

            named.append({"cls": name, "xyxy": (x1, y1, x2, y2), "bad": bad})

        for i in range(len(named)):
            for j in range(i + 1, len(named)):
                a, b = named[i], named[j]
                if a["cls"] != b["cls"]:
                    continue
                score = iou(a["xyxy"], b["xyxy"])
                if score > iou_thr:
                    for k in (i, j):
                        named[k]["bad"] = True
                    add_issue(issues, WARN, "DUP_BOX", xml_path,
                              f"同类 {a['cls']} 疑似重复框（IoU={score:.2f} > {iou_thr}）")
                    image_problems.setdefault(img.name, []).append("DUP_BOX")

        if objects:
            for name in {o["name"] for o in objects}:
                per_class.setdefault(name, {"boxes": 0, "images": 0})["images"] += 1

        boxes_by_image[img.name] = named

    per_class_indexed = {str(k): v for k, v in per_class.items()}
    return {
        "images": len(images),
        "labels": len(labels),
        "boxes": total_boxes,
        "per_class": per_class_indexed,
        "class_names": class_names,
        "unknown_class": unknown_name,
        "image_problems": image_problems,
        "boxes_by_image": boxes_by_image,
    }


# --------------------------------------------------------------------------
# 可视化抽查
# --------------------------------------------------------------------------
def color_for(bad):
    return (214, 48, 49) if bad else (46, 160, 67)


def draw_boxes(img_path, boxes, out_path, scale_hint=None):
    with Image.open(img_path) as im:
        canvas = im.convert("RGB").copy()
    w, h = canvas.size
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("msyh.ttc", max(12, int(h * 0.035)))
    except Exception:
        font = ImageFont.load_default()

    for item in boxes:
        x1, y1, x2, y2 = item["xyxy"]
        if scale_hint == "yolo":
            # 归一化坐标乘图片尺寸；若出现明显大于 1 的值（典型错误：把像素坐标直接写进 txt），
            # 按像素解释，这样越界的框仍能画在图上，方便截图说明问题
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
                pass
            else:
                x1, x2 = x1 * w, x2 * w
                y1, y2 = y1 * h, y2 * h
        # 容错：坐标顺序反了先交换，越界先裁到图内，退化成一条线的框补齐 1px，
        # 否则 PIL 会直接抛异常（这类本身就是待修的标注错误）
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1, x2 = max(0.0, min(x1, w - 1)), max(0.0, min(x2, w - 1))
        y1, y2 = max(0.0, min(y1, h - 1)), max(0.0, min(y2, h - 1))
        if x2 - x1 < 1:
            x2 = min(w - 1, x1 + 1)
        if y2 - y1 < 1:
            y2 = min(h - 1, y1 + 1)

        color = color_for(item["bad"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, int(min(w, h) * 0.006)))
        label = str(item["cls"])
        try:
            tw = draw.textlength(label, font=font)
        except Exception:
            tw = 8 * len(label)
        th = max(12, int(h * 0.035))
        ty = max(0, y1 - th)
        draw.rectangle([x1, ty, min(w - 1, x1 + tw + 6), ty + th], fill=color)
        draw.text((x1 + 3, ty), label, fill=(255, 255, 255), font=font)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def pick_viz_targets(images, image_problems, limit, seed=0):
    """优先挑有问题的图，再随机补齐。"""
    if limit <= 0:
        return []
    bad = [im for im in images if image_problems.get(im.name)]
    rest = [im for im in images if not image_problems.get(im.name)]
    random.Random(seed).shuffle(bad)
    random.Random(seed + 1).shuffle(rest)
    return (bad + rest)[:limit]


# --------------------------------------------------------------------------
# 报表
# --------------------------------------------------------------------------
def write_reports(out_dir, meta, issues, stats, classes_summary):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    counts = {}
    for item in issues:
        counts[item["code"]] = counts.get(item["code"], 0) + 1

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "format": meta["format"],
        "images": meta["images"],
        "label_files": meta["labels"],
        "boxes": meta["boxes"],
        "issues_total": len(issues),
        "issue_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "classes": classes_summary,
        "issues": issues,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out / "issues.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["level", "code", "file", "detail"])
        writer.writeheader()
        for item in issues:
            writer.writerow({k: item.get(k, "") for k in writer.fieldnames})

    with (out / "stats_by_class.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class", "boxes", "images"])
        writer.writeheader()
        for row in classes_summary:
            writer.writerow(row)
    return report


def print_summary(report, out_dir, viz_count):
    counts = report["issue_counts"]
    print("\n" + "=" * 62)
    print("  标注质量检查报告")
    print("=" * 62)
    print(f"  格式        : {report['format'].upper()}")
    print(f"  图片        : {report['images']} 张")
    print(f"  标注文件    : {report['label_files']} 个")
    print(f"  标注框      : {report['boxes']} 个")
    print(f"  问题总数    : {report['issues_total']} 处")
    print("-" * 62)
    if not counts:
        print("  未发现问题 ✅")
    else:
        level_name = {"error": "错误", "warn": "警告", "info": "提示"}
        code_level = {}
        for item in report["issues"]:
            code_level.setdefault(item["code"], item["level"])
        print(f"  {'级别':<3}{'次数':>5}   规则")
        for code, cnt in counts.items():
            print(f"  {level_name.get(code_level.get(code), '提示'):<4}{cnt:>5}   {code}")
    print("-" * 62)
    print("  类别分布")
    for row in report["classes"]:
        print(f"    {row['class']:<14}框 {row['boxes']:>5}   图片 {row['images']:>4}")
    print("-" * 62)
    print(f"  报告目录    : {out_dir}")
    print(f"  可视化抽查  : {viz_count} 张（viz/）")
    print("=" * 62 + "\n")


# --------------------------------------------------------------------------
# 演示数据：造一份带已知错误的数据集
# --------------------------------------------------------------------------
DEMO_CLASSES = ["person", "car", "helmet", "mask"]
DEMO_ERRORS = {
    "COORD_NOT_NORMALIZED": 0,
    "BOX_DEGENERATE": 0,
    "DUP_BOX": 0,
    "CLS_OUT_OF_RANGE": 0,
    "LABEL_WITHOUT_IMAGE": 0,
    "IMAGE_WITHOUT_LABEL": 0,
}


def build_demo(root):
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (root / "classes.txt").write_text("\n".join(DEMO_CLASSES) + "\n", encoding="utf-8")

    rng = random.Random(2024)
    total = 18
    for i in range(1, total + 1):
        w, h = 320, 240
        img = Image.new("RGB", (w, h), (232, 236, 241))
        draw = ImageDraw.Draw(img)
        boxes = []
        for _ in range(rng.randint(1, 3)):
            cls = rng.randint(0, 3)
            bw, bh = rng.randint(40, 90), rng.randint(35, 80)
            x1 = rng.randint(5, w - bw - 5)
            y1 = rng.randint(5, h - bh - 5)
            fill = [(52, 101, 164), (117, 163, 66), (196, 121, 43), (150, 78, 143)][cls]
            draw.rectangle([x1, y1, x1 + bw, y1 + bh], fill=fill)
            boxes.append((cls, (x1 + bw / 2) / w, (y1 + bh / 2) / h, bw / w, bh / h))

        lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for c, cx, cy, bw, bh in boxes]

        # 人为埋错：让演示报告有东西可看
        if i == 3:      # 未归一化坐标（把像素当归一化）
            lines[0] = "1 120.0 90.0 60.0 50.0"
            DEMO_ERRORS["COORD_NOT_NORMALIZED"] += 1
        if i == 5:      # 宽为 0
            lines[-1] = f"2 0.5 0.5 0.0 0.2"
            DEMO_ERRORS["BOX_DEGENERATE"] += 1
        if i == 7:      # 重复框
            lines.append(lines[0])
            DEMO_ERRORS["DUP_BOX"] += 1
        if i == 9:      # 类别越界
            lines.append("9 0.4 0.4 0.2 0.2")
            DEMO_ERRORS["CLS_OUT_OF_RANGE"] += 1
        if i == 11:     # 越界框
            lines.append("2 0.98 0.5 0.3 0.3")

        img.save(images_dir / f"demo_{i:03d}.jpg", quality=92)
        (labels_dir / f"demo_{i:03d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 有标注无图片
    (labels_dir / "demo_999.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    DEMO_ERRORS["LABEL_WITHOUT_IMAGE"] += 1
    # 有图片无标注（负样本）
    Image.new("RGB", (320, 240), (210, 215, 220)).save(images_dir / "demo_888.jpg", quality=92)
    DEMO_ERRORS["IMAGE_WITHOUT_LABEL"] += 1

    return images_dir, labels_dir, root / "classes.txt"


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="标注数据质量检查工具（YOLO / VOC）")
    ap.add_argument("--demo", action="store_true", help="生成并检查一份演示数据集")
    ap.add_argument("--format", choices=["yolo", "voc"], default="yolo", help="标注格式")
    ap.add_argument("--images", help="图片目录")
    ap.add_argument("--labels", help="标注目录（YOLO 为 .txt，VOC 为 .xml）")
    ap.add_argument("--classes", help="classes.txt（类别表）")
    ap.add_argument("--iou", type=float, default=0.9, help="判为重复框的 IoU 阈值，默认 0.9")
    ap.add_argument("--viz", type=int, default=12, help="生成多少张可视化抽查图，默认 12，0 为不生成")
    ap.add_argument("--out", default="report", help="报告输出目录，默认 ./report")
    ap.add_argument("--seed", type=int, default=0, help="抽查随机种子")
    ap.add_argument("--fail-on-error", action="store_true", help="有 error 级问题时以退出码 1 结束（可接 CI）")

    args = ap.parse_args(argv)

    if args.demo:
        images_dir, labels_dir, classes = build_demo("demo_data")
        fmt = "yolo"
        print(f"[demo] 已生成演示数据集：{Path('demo_data').resolve()}")
        print("[demo] 其中故意埋了 6 类错误，用于验证脚本能不能查出来\n")
        args.images, args.labels, args.classes = str(images_dir), str(labels_dir), str(classes)
    else:
        if not (args.images and args.labels):
            ap.error("请提供 --images 与 --labels（或使用 --demo）")
        fmt = args.format

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    if not images_dir.is_dir():
        raise SystemExit(f"图片目录不存在：{images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"标注目录不存在：{labels_dir}")

    issues = []
    if fmt == "yolo":
        stats = check_yolo(str(images_dir), str(labels_dir), args.classes, args.iou, issues)
    else:
        stats = check_voc(str(images_dir), str(labels_dir), args.classes, args.iou, issues)

    classes_summary = []
    if isinstance(stats["per_class"], dict):
        for key in sorted(stats["per_class"], key=lambda k: (isinstance(k, str), k)):
            row = stats["per_class"][key]
            name = stats["class_names"][key] if isinstance(key, int) and key < len(stats["class_names"]) else key
            classes_summary.append({"class": name, "boxes": row["boxes"], "images": row["images"]})
    for row in classes_summary:
        if stats["images"] and row["boxes"] == 0:
            add_issue(issues, WARN, "CLASS_WITHOUT_BOX", args.classes or "-",
                      f"类别“{row['class']}”在数据集里一个框都没有")

    images = list_images(images_dir)
    boxes_by_image = stats["boxes_by_image"]
    viz_targets = pick_viz_targets(images, stats["image_problems"], args.viz, args.seed)
    viz_done = 0
    for img in viz_targets:
        boxes = boxes_by_image.get(img.name, [])
        if not boxes:
            continue
        codes = stats["image_problems"].get(img.name, [])
        suffix = codes[0].lower().replace("_", "-") if codes else "ok"
        out_name = f"{Path(args.out) / 'viz' / (img.stem + '__' + suffix + '.png')}"
        try:
            draw_boxes(img, boxes, out_name, scale_hint="yolo" if fmt == "yolo" else None)
            viz_done += 1
        except Exception as exc:  # 单张画不出来不影响整份报告
            print(f"[warn] 可视化失败 {img.name}: {exc}")

    meta = {"format": fmt, "images": stats["images"], "labels": stats["labels"], "boxes": stats["boxes"]}
    report = write_reports(args.out, meta, issues, stats, classes_summary)
    print_summary(report, args.out, viz_done)

    if args.demo:
        print("埋入的错误 vs 检出结果：")
        for code, expected in DEMO_ERRORS.items():
            found = report["issue_counts"].get(code, 0)
            flag = "OK " if found >= expected else "缺 "
            print(f"  {flag}{code:<22} 埋入 {expected}  检出 {found}")
        print()

    if args.fail_on_error and any(i["level"] == ERROR for i in issues):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
