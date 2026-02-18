#!/usr/bin/env python3
"""Render inhibitor-GTPase complex using PyMOL for the architecture diagram."""

import pymol
from pymol import cmd

pymol.finish_launching(['pymol', '-qc'])  # quiet, no GUI

CIF = '/cwork/hsb26/GEF_inhibition/data/af3_server_outputs/itsn_rfd1_0_7_cdc42/fold_itsn_rfd1_0_7_cdc42_model_0.cif'
OUT = '/cwork/hsb26/GEF_inhibition/results/figures/structure_render.png'

cmd.load(CIF, 'complex')

# Chain A = inhibitor, Chain B = GTPase
cmd.remove('solvent')

# Background
cmd.bg_color('white')
cmd.set('ray_opaque_background', 1)

# Show as cartoon
cmd.hide('everything')
cmd.show('cartoon', 'all')

# Color: inhibitor (chain A) in blue tones, GTPase (chain B) in green/gray
cmd.color('marine', 'chain A')       # inhibitor - blue
cmd.color('gray70', 'chain B')       # GTPase - gray

# Highlight interface contacts as sticks
cmd.select('iface_A', 'chain A within 5.5 of chain B')
cmd.select('iface_B', 'chain B within 5.5 of chain A')
cmd.show('sticks', 'iface_A')
cmd.show('sticks', 'iface_B')
cmd.color('tv_blue', 'iface_A and elem C')
cmd.color('palegreen', 'iface_B and elem C')
cmd.color('red', 'elem O')
cmd.color('blue', 'elem N')

# Highlight hotspot residues on GTPase (281, 288, 295, 303) in red
cmd.select('hotspots', 'chain B and resi 281+288+295+303')
cmd.color('firebrick', 'hotspots and elem C')
cmd.show('sticks', 'hotspots')
cmd.set('stick_radius', 0.15)

# Switch regions
cmd.select('switch_I', 'chain B and resi 28-40')
cmd.select('switch_II', 'chain B and resi 57-74')
cmd.color('tv_yellow', 'switch_I')
cmd.color('lightorange', 'switch_II')

# Style
cmd.set('cartoon_fancy_helices', 1)
cmd.set('cartoon_smooth_loops', 1)
cmd.set('cartoon_oval_width', 0.25)
cmd.set('cartoon_oval_length', 1.2)
cmd.set('cartoon_tube_radius', 0.15)
cmd.set('antialias', 2)
cmd.set('ray_shadows', 0)
cmd.set('specular', 0.2)
cmd.set('ambient', 0.4)
cmd.set('direct', 0.5)

# Orient to show interface
cmd.orient('all')
cmd.turn('y', 20)
cmd.turn('x', -10)

# Render at high resolution
cmd.set('ray_trace_mode', 1)  # quantized colors - clean for diagrams
cmd.ray(2400, 1800)
cmd.png(OUT, dpi=300)

print(f'Saved structure render to {OUT}')
cmd.quit()
