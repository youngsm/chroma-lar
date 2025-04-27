"""
Geometry module for chroma-lar.
"""

from .build_larcube import build_detector, add_cathode
from .pmt import generate_pmt_positions, build_r5912_pmt
from .wireplane import add_wires
from .config_loader import (
    load_config_from_file,
    build_detector_from_config,
    build_detector_from_dict
)

__all__ = [
    'build_detector',
    'add_cathode',
    'generate_pmt_positions',
    'build_r5912_pmt',
    'add_wires',
    'load_config_from_file',
    'build_detector_from_config',
    'build_detector_from_dict'
]

print(r"""
 d                 u
---->----•----->----
         \              
       W /             ____  ____   __   __ _  ____   __   ____  ____  
         \      e     / ___)(_  _) / _\ (  ( \(  __) /  \ (  _ \(    \
         •----->----  \___ \  )(  /    \/    / ) _) (  O ) )   / ) D ( 
         |            (____/ (__) \_/\_/\_)__)(__)   \__/ (__\_)(____/
       v X             __ _  ____  _  _  ____  ____   __   __ _   __   ____ 
         |            (  ( \(  __)/ )( \(_  _)(  _ \ (  ) (  ( \ /  \ / ___)
         •----->----  /    / ) _) ) \/ (  )(   )   /  )(  /    /(  O )\___ \
         \      e     \_)__)(____)\____/ (__) (__\_) (__) \_)__) \__/ (____/
       W /            
         \              
---->----•----->----
 d                 u   
""")