"""
xlsx_io.py — 纯 stdlib XLSX 读写辅助模块
=========================================
当 openpyxl 不可用时提供 read_excel / write_excel 替代实现。
依赖：zipfile（stdlib）、xml.etree.ElementTree（stdlib）、pandas（仅用于构造 DataFrame）

对外接口（与 pandas 兼容）：
  read_excel(path, sheet_name=0)
      → 单个 sheet_name（str/int/None）: 返回 pd.DataFrame
      → sheet_name=None: 返回 {sheet_name: pd.DataFrame}

  ExcelFile(path)
      .sheet_names     → List[str]
      .parse(sheet)    → pd.DataFrame
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# XML 命名空间
_NS_SS  = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


def _tag(ns: str, local: str) -> str:
    return f"{{{ns}}}{local}"


def _read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(data)
    result: List[str] = []
    for si in root.findall(_tag(_NS_SS, "si")):
        # 优先取 <t>，再取所有 <r><t> 的拼接（富文本）
        t = si.find(_tag(_NS_SS, "t"))
        if t is not None and t.text:
            result.append(t.text)
        else:
            parts = [r.text or "" for r in si.iter(_tag(_NS_SS, "t"))]
            result.append("".join(parts))
    return result


def _read_sheet_names(zf: zipfile.ZipFile) -> List[str]:
    """读取 xl/workbook.xml 中的 sheet 顺序。"""
    data = zf.read("xl/workbook.xml")
    root = ET.fromstring(data)
    names: List[str] = []
    for sheet in root.iter(_tag(_NS_SS, "sheet")):
        names.append(sheet.attrib.get("name", ""))
    return names


def _col_index(col_str: str) -> int:
    """列字母 → 0-based 索引：'A' → 0, 'Z' → 25, 'AA' → 26。"""
    idx = 0
    for ch in col_str.upper():
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _parse_cell_ref(ref: str):
    """'A1' → (col_0based=0, row_1based=1)。"""
    col_str = ""
    row_str = ""
    for ch in ref:
        if ch.isalpha():
            col_str += ch
        else:
            row_str += ch
    return _col_index(col_str), int(row_str)


def _read_sheet(zf: zipfile.ZipFile, path: str, shared: List[str]) -> pd.DataFrame:
    """
    读取单个 worksheet XML → pd.DataFrame（第一行为列名）。
    """
    data  = zf.read(path)
    root  = ET.fromstring(data)
    sheet_data = root.find(_tag(_NS_SS, "sheetData"))
    if sheet_data is None:
        return pd.DataFrame()

    raw_rows: Dict[int, Dict[int, Any]] = {}   # {row_1based: {col_0based: value}}
    max_col = 0

    for row_el in sheet_data:
        r_idx = int(row_el.attrib.get("r", 0))
        for c_el in row_el:
            ref = c_el.attrib.get("r", "")
            if not ref:
                continue
            col_0, _ = _parse_cell_ref(ref)
            max_col = max(max_col, col_0)

            cell_type = c_el.attrib.get("t", "n")
            v_el = c_el.find(_tag(_NS_SS, "v"))
            val: Any = None
            if v_el is not None and v_el.text is not None:
                if cell_type == "s":           # shared string
                    try:
                        val = shared[int(v_el.text)]
                    except (IndexError, ValueError):
                        val = v_el.text
                elif cell_type in ("b",):       # boolean
                    val = bool(int(v_el.text))
                else:
                    try:
                        val = int(v_el.text) if "." not in v_el.text else float(v_el.text)
                    except ValueError:
                        val = v_el.text

            raw_rows.setdefault(r_idx, {})[col_0] = val

    if not raw_rows:
        return pd.DataFrame()

    all_rows_sorted = sorted(raw_rows.keys())
    n_cols = max_col + 1

    # 第一行作为列名
    header_dict = raw_rows.get(all_rows_sorted[0], {})
    cols = [header_dict.get(c, f"col_{c}") for c in range(n_cols)]

    rows: List[List] = []
    for r_idx in all_rows_sorted[1:]:
        row_dict = raw_rows[r_idx]
        rows.append([row_dict.get(c, None) for c in range(n_cols)])

    return pd.DataFrame(rows, columns=cols)


def _sheet_zip_path(zf: zipfile.ZipFile, sheet_idx: int) -> str:
    """根据 workbook.xml.rels 解析第 sheet_idx+1 张 sheet 的 zip 内路径。"""
    rels_data = zf.read("xl/_rels/workbook.xml.rels")
    root = ET.fromstring(rels_data)
    rId = f"rId{sheet_idx + 1}"
    for rel in root:
        if rel.attrib.get("Id") == rId:
            target = rel.attrib.get("Target", "")
            if not target.startswith("xl/"):
                target = "xl/" + target
            return target
    # fallback
    return f"xl/worksheets/sheet{sheet_idx + 1}.xml"


# ─── 公开 API ──────────────────────────────────────────────────────────────

class ExcelFile:
    """pd.ExcelFile 的轻量替代（无需 openpyxl）。"""

    def __init__(self, path: str):
        self._path   = path
        self._zf     = zipfile.ZipFile(path, "r")
        self._shared = _read_shared_strings(self._zf)
        self._names  = _read_sheet_names(self._zf)

    @property
    def sheet_names(self) -> List[str]:
        return self._names

    def parse(self, sheet: Union[str, int] = 0) -> pd.DataFrame:
        if isinstance(sheet, int):
            idx  = sheet
        else:
            try:
                idx = self._names.index(sheet)
            except ValueError:
                available = self._names
                raise KeyError(f"Sheet '{sheet}' not found. Available: {available}")
        zip_path = _sheet_zip_path(self._zf, idx)
        return _read_sheet(self._zf, zip_path, self._shared)

    def close(self):
        self._zf.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_excel(
    path: str,
    sheet_name: Union[str, int, None] = 0,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    pd.read_excel 的轻量替代（无需 openpyxl）。

    Parameters
    ----------
    path        : xlsx 文件路径
    sheet_name  : str/int → 返回单个 DataFrame；None → 返回 {name: DataFrame}

    Returns
    -------
    pd.DataFrame  或  {sheet_name: pd.DataFrame}
    """
    with ExcelFile(path) as xls:
        if sheet_name is None:
            return {name: xls.parse(name) for name in xls.sheet_names}
        return xls.parse(sheet_name)


# ─── 让 pandas 的 read_excel / ExcelFile 自动回退 ─────────────────────────

def patch_pandas_if_needed() -> None:
    """
    若 openpyxl 不可用，将 pd.read_excel 和 pd.ExcelFile 替换为本模块实现。
    在 plot_hv.py / plot_pareto3d.py 开头调用一次即可。
    """
    try:
        import openpyxl  # noqa: F401
        return  # openpyxl 可用，无需 patch
    except ImportError:
        pass

    pd.read_excel = read_excel  # type: ignore[assignment]
    pd.ExcelFile  = ExcelFile   # type: ignore[assignment]
