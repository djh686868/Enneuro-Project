# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from font_setup import setup_cjk_font
f = setup_cjk_font()
print(f"FOUND: {f}")
