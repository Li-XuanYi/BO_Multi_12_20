from __future__ import annotations
"""
export_to_xlsx.py — LLAMBO-MO / ParEGO 结果导出工具
=====================================================
从检查点 JSON 文件读取数据，导出供绘图脚本直接使用的 XLSX 文件。

输出文件：
  hv_curves.xlsx      — HV 收敛 + Pareto count 曲线（供 plot_hv.py 使用）
  pareto_front.xlsx   — EIMO Pareto 前沿 3D 数据（供 plot_pareto3d.py 使用）

不依赖任何项目内部模块（纯 json/pandas/openpyxl），可在无 PyBaMM 环境运行。

用法：
  # 自动查找检查点（默认路径）
  python export_to_xlsx.py

  # 手动指定文件
  python export_to_xlsx.py \
      --eimo-db checkpoints/db_t0034.json \
      --parego-summary results_parego_300/parego_final_summary.json

  # 多 seed 模式（unified_runner.py 调用）
  python export_to_xlsx.py \
      --eimo-db-list results/eimo/seed_0/db_final.json results/eimo/seed_1/db_final.json \
      --parego-summary-list results/parego/seed_0/parego_final_summary.json \
      --out-dir results/xlsx/
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# pandas/openpyxl 为可选依赖；若不可用则使用内置 XLSX 写入器
try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False

try:
    import openpyxl as _openpyxl  # noqa: F401
    _OPENPYXL_OK = True
except ImportError:
    _OPENPYXL_OK = False


# ═══════════════════════════════════════════════════════════════════════════
# §Z  内置最小 XLSX 写入器（无需 openpyxl）
# ═══════════════════════════════════════════════════════════════════════════

def _esc(v: str) -> str:
    """转义 XML 特殊字符。"""
    return (str(v)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _cell_ref(row: int, col: int) -> str:
    """(1-based row, 1-based col) → 'A1' 形式单元格引用。"""
    col_str = ""
    c = col
    while c > 0:
        c, rem = divmod(c - 1, 26)
        col_str = chr(65 + rem) + col_str
    return f"{col_str}{row}"


def _xl_rows_xml(rows: List[List]) -> Tuple[str, List[str]]:
    """
    将 [[val, ...], ...] 转为 <sheetData>...</sheetData> XML 字符串
    以及共享字符串列表（按出现顺序）。
    """
    shared: List[str]    = []
    shared_idx: Dict     = {}
    row_tags: List[str]  = []

    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, val in enumerate(row, start=1):
            ref = _cell_ref(r_idx, c_idx)
            if val is None or (isinstance(val, float) and val != val):  # NaN
                cells.append(f'<c r="{ref}"/>')
            elif isinstance(val, (int, float, np.integer, np.floating)):
                cells.append(f'<c r="{ref}" t="n"><v>{val}</v></c>')
            else:
                s = _esc(str(val))
                if s not in shared_idx:
                    shared_idx[s] = len(shared)
                    shared.append(s)
                idx = shared_idx[s]
                cells.append(f'<c r="{ref}" t="s"><v>{idx}</v></c>')
        row_tags.append(f'<row r="{r_idx}">{"".join(cells)}</row>')

    return "<sheetData>" + "".join(row_tags) + "</sheetData>", shared


def _sheet_xml(rows: List[List]) -> Tuple[str, List[str]]:
    """返回 worksheet XML 字符串 + 共享字符串列表。"""
    data_xml, shared = _xl_rows_xml(rows)
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        + data_xml
        + "</worksheet>"
    )
    return xml, shared


def _shared_strings_xml(strings: List[str]) -> str:
    items = "".join(f"<si><t>{s}</t></si>" for s in strings)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
        f' count="{len(strings)}" uniqueCount="{len(strings)}">'
        + items + "</sst>"
    )


def _workbook_xml(sheet_names: List[str]) -> str:
    sheets = "".join(
        f'<sheet name="{_esc(n)}" sheetId="{i+1}" r:id="rId{i+1}"/>'
        for i, n in enumerate(sheet_names)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>" + sheets + "</sheets></workbook>"
    )


def _workbook_rels_xml(sheet_names: List[str]) -> str:
    rels = "".join(
        f'<Relationship Id="rId{i+1}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{i+1}.xml"/>'
        for i in range(len(sheet_names))
    )
    rels += (
        '<Relationship Id="rIdSS" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" '
        'Target="sharedStrings.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + rels + "</Relationships>"
    )


def _content_types_xml(n_sheets: int) -> str:
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i+1}.xml" '
        f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(n_sheets)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/sharedStrings.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        + overrides + "</Types>"
    )


def write_xlsx(path: str, sheets: Dict[str, List[List]]) -> None:
    """
    将 {sheet_name: [[row_values], ...]} 写入 XLSX 文件。
    不依赖 openpyxl / xlsxwriter，纯 stdlib 实现。
    """
    sheet_names = list(sheets.keys())

    # 收集所有共享字符串（跨 sheet 合并）
    all_shared: List[str]    = []
    shared_idx: Dict[str, int] = {}
    sheet_xmls: List[str]    = []

    for name in sheet_names:
        rows = sheets[name]
        # 重新构建：直接传入 rows，收集字符串
        data_xml, sheet_shared = _xl_rows_xml(rows)
        # 合并到全局共享字符串，并重写 sheet XML 中的字符串索引
        remap: Dict[int, int] = {}
        for local_idx, s in enumerate(sheet_shared):
            if s not in shared_idx:
                shared_idx[s] = len(all_shared)
                all_shared.append(s)
            remap[local_idx] = shared_idx[s]

        # 用正则替换 t="s"><v>N</v> 中的 N 为全局索引
        def _replace_idx(m: re.Match) -> str:
            return f't="s"><v>{remap[int(m.group(1))]}</v>'
        data_xml = re.sub(r't="s"><v>(\d+)</v>', _replace_idx, data_xml)

        xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            + data_xml + "</worksheet>"
        )
        sheet_xmls.append(xml)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml",   _content_types_xml(len(sheet_names)))
        zf.writestr("_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/></Relationships>'
        )
        zf.writestr("xl/workbook.xml",        _workbook_xml(sheet_names))
        zf.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(sheet_names))
        zf.writestr("xl/sharedStrings.xml",   _shared_strings_xml(all_shared))
        for i, xml in enumerate(sheet_xmls):
            zf.writestr(f"xl/worksheets/sheet{i+1}.xml", xml)


def _df_to_rows(df: "pd.DataFrame") -> List[List]:
    """pd.DataFrame → [[header...], [row...], ...] for write_xlsx。"""
    rows = [list(df.columns)]
    for _, row in df.iterrows():
        rows.append([None if (isinstance(v, float) and v != v) else v
                     for v in row])
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# §A  JSON 读取
# ═══════════════════════════════════════════════════════════════════════════

def _find_latest_eimo_db(checkpoint_dir: str) -> Optional[str]:
    """在 checkpoint_dir 中找最新的 db_t*.json。"""
    ckpt = Path(checkpoint_dir)
    if not ckpt.exists():
        return None
    files = sorted(ckpt.glob("db_t*.json"))
    return str(files[-1]) if files else None


def load_eimo_stats(db_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    读取 LLAMBO-MO 数据库 JSON，返回：
      iteration_stats : [{n_total, hypervolume, pareto_size}, ...]
      pareto_obs      : [{theta, objectives}, ...]（Pareto 前沿观测）
    """
    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = data.get("iteration_stats", [])

    observations   = data.get("observations", [])
    pareto_indices = data.get("pareto_indices", [])
    pareto_obs = [observations[i] for i in pareto_indices if i < len(observations)]

    return stats, pareto_obs


def load_parego_stats(summary_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    读取 ParEGO 汇总 JSON，返回：
      iteration_stats : [{n_total, hypervolume, pareto_size}, ...]
      pareto_obs      : [{theta, objectives}, ...]
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats      = data.get("iteration_stats", [])
    pareto_obs = data.get("pareto_front", [])
    return stats, pareto_obs


# ═══════════════════════════════════════════════════════════════════════════
# §B  HV / Pareto Count 曲线对齐
# ═══════════════════════════════════════════════════════════════════════════

def _stats_to_series(stats: List[Dict], value_key: str) -> Dict[int, float]:
    """iteration_stats → {n_total: value} 字典（以累计评估数为 x 轴）。"""
    series: Dict[int, float] = {}
    for s in stats:
        n = s.get("n_total")
        v = s.get(value_key)
        if n is not None and v is not None:
            series[int(n)] = float(v)
    return series


def build_aligned_df(
    method_series_list: List[Tuple[str, Dict[int, float]]],
    ffill: bool = True,
):
    """
    将多个 {n_total: value} 时间序列对齐到公共 x 轴（union），前向填充缺失点。

    Returns
    -------
    pd.DataFrame — 首列 "Evaluation"，其余列各为一个方法/run
    """
    all_x: set = set()
    for _, series in method_series_list:
        all_x.update(series.keys())
    x_sorted = sorted(all_x)

    # 构建简单字典列表（不依赖 pandas）
    col_names = [name for name, _ in method_series_list]
    data_rows: List[List] = [["Evaluation"] + col_names]  # header

    prev = {name: float("nan") for name in col_names}
    for x in x_sorted:
        row: List = [x]
        for name, series in method_series_list:
            v = series.get(x, float("nan"))
            if ffill and (v != v):  # NaN
                v = prev[name]
            prev[name] = v
            row.append(v)
        data_rows.append(row)

    return data_rows


# ═══════════════════════════════════════════════════════════════════════════
# §C  Pareto 前沿关键点标注
# ═══════════════════════════════════════════════════════════════════════════

def _label_pareto_key_points(pareto_obs: List[Dict]) -> List[str]:
    """
    为 Pareto 前沿标注关键点：
      A — 最短充电时间
      B — 最低峰值温度
      C — 最小老化程度
      D/E/F — 离归一化质心最近的 3 个折衷点（Pareto ≥ 7 时才标注）
    返回与 pareto_obs 等长的标签列表（无标注点为空字符串）。
    """
    n = len(pareto_obs)
    labels = [""] * n
    if n == 0:
        return labels

    objs_arr = np.array(
        [obs.get("objectives", [0.0, 0.0, 0.0]) for obs in pareto_obs],
        dtype=float,
    )  # shape (n, 3)

    def _assign(target_label: str, col: int) -> None:
        """将 col 维度最小值分配给 target_label，跳过已有标注的点。"""
        for idx in np.argsort(objs_arr[:, col]):
            if labels[idx] == "":
                labels[idx] = target_label
                return

    _assign("A", 0)  # A: 最短充电时间
    _assign("B", 1)  # B: 最低温度
    _assign("C", 2)  # C: 最小老化

    # D/E/F: Pareto 点 >= 7 时，标注离归一化质心最近的 3 个折衷点
    if n >= 7:
        mins  = objs_arr.min(axis=0)
        maxs  = objs_arr.max(axis=0)
        denom = np.where(maxs - mins < 1e-10, 1.0, maxs - mins)
        norm  = (objs_arr - mins) / denom
        centroid = norm.mean(axis=0)
        dists    = np.linalg.norm(norm - centroid, axis=1)
        extras   = 0
        for label_char in ["D", "E", "F"]:
            if extras >= 3:
                break
            for idx in np.argsort(dists):
                if labels[idx] == "":
                    labels[idx] = label_char
                    extras += 1
                    break

    return labels


# ═══════════════════════════════════════════════════════════════════════════
# §D  主导出函数
# ═══════════════════════════════════════════════════════════════════════════

def export_hv_curves(
    eimo_stats_list:   List[List[Dict]],
    parego_stats_list: List[List[Dict]],
    out_path: str = "hv_curves.xlsx",
) -> str:
    """
    导出 HV 收敛曲线 + Pareto Count 曲线到 XLSX。

    Sheet "HV"           : Evaluation | EIMO_run1 | ... | ParEGO_run1 | ...
    Sheet "Pareto Count" : 同格式，值为 pareto_size
    """
    hv_series: List[Tuple[str, Dict[int, float]]] = []
    for i, stats in enumerate(eimo_stats_list):
        hv_series.append((f"EIMO_run{i + 1}", _stats_to_series(stats, "hypervolume")))
    for i, stats in enumerate(parego_stats_list):
        hv_series.append((f"ParEGO_run{i + 1}", _stats_to_series(stats, "hypervolume")))

    count_series: List[Tuple[str, Dict[int, float]]] = []
    for i, stats in enumerate(eimo_stats_list):
        count_series.append((f"EIMO_run{i + 1}", _stats_to_series(stats, "pareto_size")))
    for i, stats in enumerate(parego_stats_list):
        count_series.append((f"ParEGO_run{i + 1}", _stats_to_series(stats, "pareto_size")))

    rows_hv    = build_aligned_df(hv_series)
    rows_count = build_aligned_df(count_series)

    write_xlsx(out_path, {"HV": rows_hv, "Pareto Count": rows_count})

    print(f"[export] HV 曲线已导出: {out_path}  "
          f"({len(eimo_stats_list)} EIMO + {len(parego_stats_list)} ParEGO runs)")
    return out_path


def _filter_nondominated(obs_list: List[Dict]) -> List[Dict]:
    """从观测列表中提取非支配解（用于合并多 seed Pareto 前沿）。"""
    if not obs_list:
        return []
    objs = np.array(
        [obs.get("objectives", [0.0, 0.0, 0.0]) for obs in obs_list], dtype=float
    )
    n = len(objs)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                is_dominated[i] = True
                break
    return [obs_list[i] for i in range(n) if not is_dominated[i]]


def export_pareto_front(
    eimo_pareto_list: List[List[Dict]],
    out_path: str = "pareto_front.xlsx",
    use_last_seed_only: bool = True,
) -> str:
    """
    导出 EIMO Pareto 前沿 3D 数据到 XLSX。
    列：Charging Time | Temperature | Aging | Method | Label

    Parameters
    ----------
    eimo_pareto_list     : 每个元素为一次 seed 的 Pareto 前沿观测列表
    out_path             : 输出路径
    use_last_seed_only   : True → 仅用最后一次 seed；False → 合并所有 seed 后取非支配解
    """
    if use_last_seed_only or len(eimo_pareto_list) == 1:
        pareto_obs = eimo_pareto_list[-1]
    else:
        all_obs: List[Dict] = []
        for obs_list in eimo_pareto_list:
            all_obs.extend(obs_list)
        pareto_obs = _filter_nondominated(all_obs)

    if not pareto_obs:
        print("[export] 警告：EIMO Pareto 前沿为空，跳过导出。")
        return out_path

    labels = _label_pareto_key_points(pareto_obs)

    rows = []
    for obs, lbl in zip(pareto_obs, labels):
        o = obs.get("objectives", [0.0, 0.0, 0.0])
        rows.append({
            "Charging Time": float(o[0]),
            "Temperature":   float(o[1]),
            "Aging":         float(o[2]),
            "Method":        "EIMO",
            "Label":         lbl,
        })

    header = ["Charging Time", "Temperature", "Aging", "Method", "Label"]
    xlsx_rows: List[List] = [header]
    for r in rows:
        xlsx_rows.append([r["Charging Time"], r["Temperature"], r["Aging"],
                          r["Method"], r["Label"]])

    write_xlsx(out_path, {"Pareto Front": xlsx_rows})
    n_labeled = sum(1 for r in rows if r["Label"])
    print(f"[export] Pareto 前沿已导出: {out_path}  "
          f"({len(rows)} 个非支配解, {n_labeled} 个标注点)")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# §E  便捷入口（供 unified_runner.py 调用）
# ═══════════════════════════════════════════════════════════════════════════

def export_single_run(
    eimo_db_path:        str,
    parego_summary_path: str,
    out_dir:             str = ".",
    hv_filename:         str = "hv_curves.xlsx",
    pareto_filename:     str = "pareto_front.xlsx",
) -> Tuple[str, str]:
    """单次运行导出：各一个 JSON → 两个 XLSX。"""
    eimo_stats,   eimo_pareto = load_eimo_stats(eimo_db_path)
    parego_stats, _           = load_parego_stats(parego_summary_path)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hv_path     = export_hv_curves(
        [eimo_stats], [parego_stats],
        str(out / hv_filename),
    )
    pareto_path = export_pareto_front(
        [eimo_pareto],
        str(out / pareto_filename),
    )
    return hv_path, pareto_path


def export_multi_run(
    eimo_db_paths:        List[str],
    parego_summary_paths: List[str],
    out_dir:              str = ".",
    hv_filename:          str = "hv_curves.xlsx",
    pareto_filename:      str = "pareto_front.xlsx",
) -> Tuple[str, str]:
    """多 seed 导出：多个 JSON → 多列 XLSX。"""
    eimo_stats_list:  List[List[Dict]] = []
    eimo_pareto_list: List[List[Dict]] = []
    for p in eimo_db_paths:
        s, pf = load_eimo_stats(p)
        eimo_stats_list.append(s)
        eimo_pareto_list.append(pf)

    parego_stats_list: List[List[Dict]] = []
    for p in parego_summary_paths:
        s, _ = load_parego_stats(p)
        parego_stats_list.append(s)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hv_path     = export_hv_curves(
        eimo_stats_list, parego_stats_list,
        str(out / hv_filename),
    )
    pareto_path = export_pareto_front(
        eimo_pareto_list,
        str(out / pareto_filename),
        use_last_seed_only=False,
    )
    return hv_path, pareto_path


# ═══════════════════════════════════════════════════════════════════════════
# §F  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 LLAMBO-MO / ParEGO 检查点 JSON 导出为绘图用 XLSX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动查找检查点（默认路径）
  python export_to_xlsx.py

  # 手动指定文件
  python export_to_xlsx.py \\
      --eimo-db checkpoints/db_t0034.json \\
      --parego-summary results_parego_300/parego_final_summary.json

  # 多 seed 模式
  python export_to_xlsx.py \\
      --eimo-db-list results/eimo/seed_0/db_final.json results/eimo/seed_1/db_final.json \\
      --parego-summary-list results/parego/seed_0/parego_final_summary.json \\
      --out-dir results/xlsx/
        """,
    )
    parser.add_argument(
        "--eimo-db", type=str, default=None,
        help="LLAMBO-MO 数据库 JSON 路径（单次模式）",
    )
    parser.add_argument(
        "--parego-summary", type=str, default=None,
        help="ParEGO 汇总 JSON 路径（单次模式）",
    )
    parser.add_argument(
        "--eimo-db-list", nargs="+", default=None,
        help="LLAMBO-MO 数据库 JSON 路径列表（多 seed 模式）",
    )
    parser.add_argument(
        "--parego-summary-list", nargs="+", default=None,
        help="ParEGO 汇总 JSON 路径列表（多 seed 模式）",
    )
    parser.add_argument(
        "--eimo-checkpoint-dir", type=str, default="checkpoints",
        help="LLAMBO-MO 检查点目录（自动查找最新，默认：checkpoints）",
    )
    parser.add_argument(
        "--parego-results-dir", type=str, default="results_parego_300",
        help="ParEGO 结果目录（默认：results_parego_300）",
    )
    parser.add_argument(
        "--out-dir", type=str, default=".",
        help="XLSX 输出目录（默认：当前目录）",
    )
    parser.add_argument(
        "--hv-file", type=str, default="hv_curves.xlsx",
        help="HV 曲线输出文件名（默认：hv_curves.xlsx）",
    )
    parser.add_argument(
        "--pareto-file", type=str, default="pareto_front.xlsx",
        help="Pareto 前沿输出文件名（默认：pareto_front.xlsx）",
    )
    return parser


def main() -> None:
    parser = _create_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # ── 多 seed 模式 ──────────────────────────────────────────────────────
    if args.eimo_db_list or args.parego_summary_list:
        eimo_paths   = args.eimo_db_list   or []
        parego_paths = args.parego_summary_list or []
        if not eimo_paths:
            print("[错误] --eimo-db-list 为空", file=sys.stderr)
            sys.exit(1)
        if not parego_paths:
            print("[错误] --parego-summary-list 为空", file=sys.stderr)
            sys.exit(1)
        export_multi_run(
            eimo_db_paths=eimo_paths,
            parego_summary_paths=parego_paths,
            out_dir=args.out_dir,
            hv_filename=args.hv_file,
            pareto_filename=args.pareto_file,
        )
        return

    # ── 单次模式 — 自动查找 EIMO ─────────────────────────────────────────
    eimo_db = args.eimo_db
    if eimo_db is None:
        ckpt_dir = script_dir / args.eimo_checkpoint_dir
        eimo_db  = _find_latest_eimo_db(str(ckpt_dir))
        if eimo_db is None:
            print(
                f"[错误] 在 {ckpt_dir} 中未找到 db_t*.json\n"
                "请用 --eimo-db 手动指定路径",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[自动] 使用 EIMO 检查点: {eimo_db}")

    # ── 自动查找 ParEGO ───────────────────────────────────────────────────
    parego_summary = args.parego_summary
    if parego_summary is None:
        for candidate in [
            script_dir / args.parego_results_dir / "parego_final_summary.json",
            script_dir / args.parego_results_dir / "pareto_summary.json",
        ]:
            if candidate.exists():
                parego_summary = str(candidate)
                print(f"[自动] 使用 ParEGO 汇总: {parego_summary}")
                break
        if parego_summary is None:
            print(
                "[错误] 未找到 ParEGO 汇总文件\n"
                "请用 --parego-summary 手动指定路径",
                file=sys.stderr,
            )
            sys.exit(1)

    export_single_run(
        eimo_db_path=eimo_db,
        parego_summary_path=parego_summary,
        out_dir=args.out_dir,
        hv_filename=args.hv_file,
        pareto_filename=args.pareto_file,
    )


if __name__ == "__main__":
    main()