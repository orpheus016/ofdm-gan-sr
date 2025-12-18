# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Architecture Visualization using Graphviz
# MINI ARCHITECTURE VERSION (matches RTL implementation)
# =============================================================================
"""
ARCHITECTURE VISUALIZATION - MINI VERSION
==========================================

This module generates professional block diagrams of the CWGAN-GP mini architecture
designed for FPGA deployment using Graphviz.

Architecture Specs (Mini):
    Generator:    2 → 4 → 8 → 4 → 2 channels, 16 samples
    Discriminator: 4 → 8 → 16 → 1, 16 samples
    Total params: ~800

Generated diagrams:
1. Full GAN architecture (Generator + Discriminator)
2. Generator Mini U-Net detailed structure
3. Discriminator Mini CNN structure
4. Data flow diagram
5. FPGA implementation overview
6. RTL module hierarchy

Requirements:
    pip install graphviz
    
    Also install Graphviz system tool:
    - Windows: Download from https://graphviz.org/download/
    - Linux: apt-get install graphviz
    - Mac: brew install graphviz
"""

import graphviz
from pathlib import Path
from typing import Optional


def draw_full_architecture(
    output_path: str = 'gan_architecture',
    view: bool = True,
    format: str = 'png'
) -> str:
    """
    Draw complete CWGAN-GP mini architecture diagram.
    
    Shows:
    - Generator (Mini U-Net) with encoder-decoder structure
    - Skip connections (additive)
    - Conditional Discriminator
    - Data flow between components
    """
    dot = graphviz.Digraph('CWGAN_GP_OFDM_Mini', comment='Mini CWGAN-GP Architecture for FPGA')
    
    dot.attr(rankdir='LR', splines='polyline', ranksep='1.0', nodesep='0.5')
    dot.attr('node', fontname='Helvetica', fontsize='11')
    dot.attr('edge', fontname='Helvetica', fontsize='9')

    # Input/Output nodes
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgrey', color='black')
    dot.node('Noisy_Input', 'Noisy Input\n(I, Q)\n[2×16]')
    dot.node('Clean_Target', 'Clean Target\n(I, Q)\n[2×16]')
    dot.node('Generated', 'Generated Output\n(Enhanced I, Q)\n[2×16]')
    dot.node('Score', 'Validity Score\n[Scalar]')

    # Generator Encoder
    dot.attr('node', shape='box', style='filled', fillcolor='#B0C4DE', color='#000080')
    dot.node('E1', 'Enc1\n2→4\nL:16→8')
    
    # Bottleneck
    dot.attr('node', fillcolor='#4169E1', fontcolor='white')
    dot.node('BN', 'Bottleneck\n4→8\nL:8→4')

    # Generator Decoder
    dot.attr('node', fillcolor='#90EE90', color='#006400', fontcolor='black')
    dot.node('D1', 'Dec1\n8→4\nL:4→8')
    
    # Final projection
    dot.attr('node', fillcolor='#228B22', fontcolor='white')
    dot.node('Final', 'Final Conv\n4→2\ntanh\nL:8→16')

    # Discriminator blocks
    dot.attr('node', fillcolor='#FFB6C1', color='#8B0000', fontcolor='black')
    dot.node('D_C1', 'D-Conv1\n4→8\nL:16→8')
    dot.node('D_C2', 'D-Conv2\n8→16\nL:8→4')
    
    dot.attr('node', fillcolor='#DC143C', fontcolor='white')
    dot.node('D_Pool', 'Sum Pool + Dense\n16→1')

    # Generator path
    dot.edge('Noisy_Input', 'E1', label='')
    dot.edge('E1', 'BN')
    dot.edge('BN', 'D1')
    dot.edge('D1', 'Final')
    dot.edge('Final', 'Generated')

    # Skip connection
    skip_style = {'style': 'dashed', 'color': '#4682B4', 'fontcolor': '#4682B4', 
                  'label': '+', 'constraint': 'false'}
    dot.edge('E1', 'D1', **skip_style)

    # Discriminator path
    dot.edge('Noisy_Input', 'D_C1', style='dotted', label='Condition', color='#FF4500')
    dot.edge('Generated', 'D_C1', label='Fake', color='#FF6347')
    dot.edge('Clean_Target', 'D_C1', label='Real', color='#228B22')
    dot.edge('D_C1', 'D_C2')
    dot.edge('D_C2', 'D_Pool')
    dot.edge('D_Pool', 'Score')

    # Subgraphs
    with dot.subgraph(name='cluster_generator') as g:
        g.attr(label='Generator (Mini U-Net)\n~258 params', color='blue', style='dashed')
        g.node('E1'); g.node('BN'); g.node('D1'); g.node('Final')
        
    with dot.subgraph(name='cluster_discriminator') as d:
        d.attr(label='Discriminator (Critic)\n~521 params', color='red', style='dashed')
        d.node('D_C1'); d.node('D_C2'); d.node('D_Pool')

    try:
        return dot.render(output_path, view=view, format=format, cleanup=True)
    except Exception as e:
        dot.save(f"{output_path}.gv")
        return f"{output_path}.gv"


def draw_generator_detailed(
    output_path: str = 'generator_detailed',
    view: bool = True,
    format: str = 'png'
) -> str:
    """Draw detailed mini generator (U-Net) architecture."""
    dot = graphviz.Digraph('Generator_Detailed', comment='Mini U-Net Generator Details')
    
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.3', ranksep='0.4')
    dot.attr('node', fontname='Courier', fontsize='10', shape='record')
    
    dot.attr('node', shape='plaintext', fontsize='16', fontname='Helvetica-Bold')
    dot.node('title', 'Mini U-Net Generator Architecture\n(~258 parameters)')
    
    layers = [
        ('input', 'Input [2×16]', '16', '#E6E6E6'),
        ('enc1', 'Conv1D(2→4, k=3, s=2) + LeakyReLU', '16→8', '#B0C4DE'),
        ('bottle', 'Conv1D(4→8, k=3, s=2) + LeakyReLU', '8→4', '#4169E1'),
        ('up1', 'Upsample(×2) Nearest Neighbor', '4→8', '#90EE90'),
        ('dec1', 'Conv1D(8→4, k=3, s=1) + LeakyReLU', '8', '#7CCD7C'),
        ('skip_add', '+ Skip Connection (Enc1)', '8', '#66CD66'),
        ('up2', 'Upsample(×2) Nearest Neighbor', '8→16', '#4CBB4C'),
        ('out', 'Conv1D(4→2, k=3, s=1) + Tanh', '16', '#228B22'),
        ('output', 'Output [2×16]', '16', '#E6E6E6'),
    ]
    
    dot.attr('node', shape='box', style='filled')
    
    prev_node = 'title'
    for name, desc, length, color in layers:
        fontcolor = 'white' if color in ['#4169E1', '#228B22'] else 'black'
        dot.node(name, f"{desc}\\nL: {length}", fillcolor=color, fontcolor=fontcolor)
        dot.edge(prev_node, name)
        prev_node = name
    
    dot.edge('enc1', 'skip_add', style='dashed', color='#4682B4', constraint='false')
        
    try:
        return dot.render(output_path, view=view, format=format, cleanup=True)
    except Exception as e:
        dot.save(f"{output_path}.gv")
        return f"{output_path}.gv"


def draw_discriminator_detailed(
    output_path: str = 'discriminator_detailed',
    view: bool = True,
    format: str = 'png'
) -> str:
    """Draw detailed mini discriminator (critic) architecture."""
    dot = graphviz.Digraph('Discriminator_Detailed', comment='Mini Discriminator Details')
    
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.3', ranksep='0.4')
    dot.attr('node', fontname='Courier', fontsize='10', shape='record')
    
    dot.attr('node', shape='plaintext', fontsize='16', fontname='Helvetica-Bold')
    dot.node('title', 'Mini Discriminator (Critic)\n(~521 parameters)')
    
    dot.attr('node', shape='box', style='filled', fillcolor='#FFE4E1')
    dot.node('input_explain', 
             'Input: concat(Candidate[2×16], Condition[2×16])\\n= [4×16]')
    
    layers = [
        ('concat', 'Concatenate', '[4, 16]', '#FFE4E1'),
        ('c1', 'Conv1D(4→8, k=3, s=2) + LeakyReLU', '[8, 8]', '#FFB6C1'),
        ('c2', 'Conv1D(8→16, k=3, s=2) + LeakyReLU', '[16, 4]', '#FF9999'),
        ('pool', 'Global Sum Pooling', '[16]', '#DC143C'),
        ('dense', 'Dense(16→1)', '[1]', '#8B0000'),
    ]
    
    dot.attr('node', shape='box', style='filled')
    dot.edge('title', 'input_explain')
    
    prev_node = 'input_explain'
    for name, desc, shape, color in layers:
        fontcolor = 'white' if color in ['#DC143C', '#8B0000'] else 'black'
        dot.node(name, f"{desc}\\n→ Shape: {shape}", fillcolor=color, fontcolor=fontcolor)
        dot.edge(prev_node, name)
        prev_node = name
        
    try:
        return dot.render(output_path, view=view, format=format, cleanup=True)
    except Exception as e:
        dot.save(f"{output_path}.gv")
        return f"{output_path}.gv"


def draw_training_flow(
    output_path: str = 'training_flow',
    view: bool = True,
    format: str = 'png'
) -> str:
    """Draw the CWGAN-GP training data flow diagram."""
    dot = graphviz.Digraph('Training_Flow', comment='CWGAN-GP Training Flow')
    
    dot.attr(rankdir='TB', splines='curved', nodesep='0.5')
    dot.attr('node', fontname='Helvetica', fontsize='11')
    
    dot.attr('node', shape='cylinder', style='filled', fillcolor='#E8E8E8')
    dot.node('data', 'Training Data\n(Synthetic OFDM)')
    
    dot.attr('node', shape='box', fillcolor='#FFFACD')
    dot.node('ofdm_gen', 'OFDM Generator\n(QPSK, 8 subcarriers)')
    dot.node('channel', 'AWGN Channel\n(SNR: 5-20 dB)')
    
    dot.attr('node', shape='ellipse', fillcolor='#98FB98')
    dot.node('clean', 'Clean OFDM [2×16]')
    dot.node('noisy', 'Noisy OFDM [2×16]')
    
    dot.attr('node', shape='box', fillcolor='#ADD8E6')
    dot.node('gen', 'Mini Generator\n(258 params)')
    
    dot.attr('node', shape='ellipse', fillcolor='#87CEEB')
    dot.node('fake', 'Enhanced OFDM [2×16]')
    
    dot.attr('node', shape='box', fillcolor='#FFB6C1')
    dot.node('disc', 'Mini Discriminator\n(521 params)')
    
    dot.attr('node', shape='diamond', fillcolor='#DDA0DD')
    dot.node('d_loss', 'Critic Loss\nWasserstein + GP')
    dot.node('g_loss', 'Generator Loss\nAdversarial + L1')
    
    dot.edge('data', 'ofdm_gen')
    dot.edge('ofdm_gen', 'clean')
    dot.edge('clean', 'channel')
    dot.edge('channel', 'noisy')
    dot.edge('noisy', 'gen')
    dot.edge('gen', 'fake')
    dot.edge('noisy', 'disc', label='condition', style='dashed')
    dot.edge('clean', 'disc', label='real')
    dot.edge('fake', 'disc', label='fake')
    dot.edge('disc', 'd_loss')
    dot.edge('disc', 'g_loss')
    
    try:
        return dot.render(output_path, view=view, format=format, cleanup=True)
    except Exception as e:
        dot.save(f"{output_path}.gv")
        return f"{output_path}.gv"


def draw_fpga_overview(
    output_path: str = 'fpga_overview',
    view: bool = True,
    format: str = 'png'
) -> str:
    """Draw FPGA implementation overview for mini architecture."""
    dot = graphviz.Digraph('FPGA_Overview', comment='FPGA Mini Generator Implementation')
    
    dot.attr(rankdir='LR', splines='polyline', nodesep='0.4')
    dot.attr('node', fontname='Helvetica', fontsize='10')
    
    dot.attr('node', shape='box3d', style='filled', fillcolor='#E6E6FA')
    dot.node('weight_rom', 'Weight ROM\n(~800 bytes)\n(Q1.7 INT8)')
    dot.node('input_buffer', 'Input Buffer\n[2×16] I/Q\n(Q8.8 INT16)')
    dot.node('output_buffer', 'Output Buffer\n[2×16] I/Q\n(Q8.8 INT16)')
    
    dot.attr('node', shape='box', fillcolor='#87CEEB')
    dot.node('conv_engine', 'Conv1D Engine\n(Parallel k=3)\n~4-8 DSPs')
    dot.node('skip_ram', 'Skip Buffer\n(32 INT16)\n64 bytes')
    dot.node('upsample', 'Upsample ×2')
    dot.node('lrelu', 'LeakyReLU')
    dot.node('tanh_lut', 'Tanh LUT\n(256 entries)')
    
    dot.attr('node', shape='hexagon', fillcolor='#98FB98')
    dot.node('ctrl', 'FSM Controller\n(10 states)')
    
    dot.edge('weight_rom', 'conv_engine', label='weights')
    dot.edge('input_buffer', 'conv_engine', label='input')
    dot.edge('conv_engine', 'lrelu')
    dot.edge('lrelu', 'skip_ram', style='dashed', label='enc')
    dot.edge('skip_ram', 'conv_engine', style='dashed', label='skip')
    dot.edge('conv_engine', 'upsample')
    dot.edge('upsample', 'conv_engine')
    dot.edge('conv_engine', 'tanh_lut', label='final')
    dot.edge('tanh_lut', 'output_buffer')
    dot.edge('ctrl', 'conv_engine', style='dotted')
    
    with dot.subgraph(name='cluster_fpga') as f:
        f.attr(label='FPGA (Mini Architecture)\n~2K LUTs, 4-8 DSPs', color='black', style='bold')
        f.node('conv_engine'); f.node('skip_ram'); f.node('upsample')
        f.node('lrelu'); f.node('tanh_lut'); f.node('ctrl')
        
    try:
        return dot.render(output_path, view=view, format=format, cleanup=True)
    except Exception as e:
        dot.save(f"{output_path}.gv")
        return f"{output_path}.gv"


def draw_rtl_modules(
    output_path: str = 'rtl_modules',
    view: bool = True,
    format: str = 'png'
) -> str:
    """Draw RTL module hierarchy diagram."""
    dot = graphviz.Digraph('RTL_Modules', comment='RTL Module Hierarchy')
    
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.3')
    dot.attr('node', fontname='Courier', fontsize='9')
    
    dot.attr('node', shape='box', style='filled', fillcolor='#FFD700')
    dot.node('top', 'cwgan_gp_top.v')
    
    dot.attr('node', fillcolor='#87CEEB')
    dot.node('gen', 'generator_mini.v')
    dot.node('disc', 'discriminator_mini.v')
    
    dot.attr('node', fillcolor='#98FB98')
    dot.node('conv', 'conv1d_engine.v')
    dot.node('upsample', 'upsample_nn.v')
    dot.node('lrelu', 'activation_lrelu.v')
    dot.node('tanh', 'activation_tanh.v')
    dot.node('pool', 'sum_pool.v')
    dot.node('rom', 'weight_rom.v')
    
    dot.edge('top', 'gen')
    dot.edge('top', 'disc')
    dot.edge('gen', 'conv'); dot.edge('gen', 'upsample')
    dot.edge('gen', 'lrelu'); dot.edge('gen', 'tanh'); dot.edge('gen', 'rom')
    dot.edge('disc', 'conv'); dot.edge('disc', 'lrelu')
    dot.edge('disc', 'pool'); dot.edge('disc', 'rom')
    
    try:
        return dot.render(output_path, view=view, format=format, cleanup=True)
    except Exception as e:
        dot.save(f"{output_path}.gv")
        return f"{output_path}.gv"


def generate_all_diagrams(
    output_dir: str = './diagrams',
    view: bool = False,
    format: str = 'png'
):
    """Generate all architecture diagrams."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating Mini Architecture Diagrams...")
    print("=" * 50)
    
    diagrams = [
        ('Full GAN Architecture', draw_full_architecture, 'gan_architecture'),
        ('Generator Detailed', draw_generator_detailed, 'generator_detailed'),
        ('Discriminator Detailed', draw_discriminator_detailed, 'discriminator_detailed'),
        ('Training Flow', draw_training_flow, 'training_flow'),
        ('FPGA Overview', draw_fpga_overview, 'fpga_overview'),
        ('RTL Modules', draw_rtl_modules, 'rtl_modules'),
    ]
    
    for name, func, filename in diagrams:
        print(f"\n{name}...")
        try:
            func(str(output_path / filename), view=view, format=format)
        except Exception as e:
            print(f"  Error: {e}")
            
    print("\n" + "=" * 50)
    print(f"Diagrams saved to: {output_path.absolute()}")


if __name__ == "__main__":
    print("=" * 60)
    print("CWGAN-GP Mini Architecture Visualization")
    print("=" * 60)
    generate_all_diagrams(view=False)
    print("\n✓ Diagram generation complete!")
