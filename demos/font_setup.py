# -*- coding: utf-8 -*-
"""统一 matplotlib 中文字体配置 — 在所有 demo 脚本开头调用"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

_CJK_FONT = None

def setup_cjk_font():
    """寻找系统中可用的中文字体并全局设置"""
    global _CJK_FONT
    if _CJK_FONT:
        return _CJK_FONT

    # 按优先级查找
    candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong',
                  'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'PingFang SC']

    for name in candidates:
        for f in fm.fontManager.ttflist:
            if f.name == name:
                plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                _CJK_FONT = name
                print(f"[Font] Using CJK font: {name}")
                return name

    # 兜底: 直接查 Windows 字体目录
    win_font_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
    for fn in os.listdir(win_font_dir):
        if fn.lower() in ['msyh.ttc', 'msyh.ttf', 'simhei.ttf', 'simsun.ttc']:
            font_path = os.path.join(win_font_dir, fn)
            fm.fontManager.addfont(font_path)
            prop = fm.FontProperties(fname=font_path)
            font_name = prop.get_name()
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            _CJK_FONT = font_name
            print(f"[Font] Using CJK font (fallback): {font_name} from {fn}")
            return font_name

    print("[Font] WARNING: No CJK font found, Chinese may not display correctly")
    _CJK_FONT = 'DejaVu Sans'
    return 'DejaVu Sans'


if __name__ == '__main__':
    setup_cjk_font()
