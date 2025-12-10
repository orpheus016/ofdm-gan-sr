# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Architecture Visualization using Graphviz
# =============================================================================
"""
ARCHITECTURE VISUALIZATION
==========================

This module generates professional block diagrams of the CWGAN-GP architecture
using Graphviz.

Generated diagrams:
1. Full GAN architecture (Generator + Discriminator)
2. Generator U-Net detailed structure
3. Discriminator CNN structure
4. Data flow diagram
5. FPGA implementation overview

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
    Draw complete CWGAN-GP architecture diagram.
    
    Shows:
    - Generator (U-Net) with encoder-decoder structure
    - Skip connections (additive)
    - Conditional Discriminator
    - Data flow between components
    
    Args:
        output_path: Base path for output file (without extension)
        view: Whether to open the rendered image
        format: Output format ('png', 'pdf', 'svg')
        
    Returns:
        Path to generated file
    """
    dot = graphviz.Digraph('CWGAN_GP_OFDM', comment='1D U-Net CWGAN-GP Architecture')
    
    # --- Global Graph Attributes ---
    dot.attr(rankdir='LR', splines='polyline', ranksep='1.0', nodesep='0.5')
    dot.attr('node', fontname='Helvetica', fontsize='11')
    dot.attr('edge', fontname='Helvetica', fontsize='9')

    # --- Define Node Styles ---
    
    # Input/Output nodes
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgrey', color='black')
    dot.node('Noisy_Input', 'Noisy Input\n(I, Q)\n[2×1024]')
    dot.node('Clean_Target', 'Clean Target\n(I, Q)\n[2×1024]')
    dot.node('Generated', 'Generated Output\n(Enhanced I, Q)\n[2×1024]')
    dot.node('Score', 'Validity Score\n[Scalar]')

    # Generator Encoder blocks (Blue gradient)
    dot.attr('node', shape='box', style='filled', fillcolor='#B0C4DE', color='#000080')
    dot.node('E1', 'Enc1\n2→32\nL:1024→512')
    dot.node('E2', 'Enc2\n32→64\nL:512→256')
    dot.node('E3', 'Enc3\n64→128\nL:256→128')
    dot.node('E4', 'Enc4\n128→256\nL:128→64')
    dot.node('E5', 'Enc5\n256→512\nL:64→32')
    
    # Bottleneck (Darker Blue)
    dot.attr('node', fillcolor='#4169E1', fontcolor='white')
    dot.node('BN', 'Bottleneck\n512→512\nL:32')

    # Generator Decoder blocks (Green gradient)
    dot.attr('node', fillcolor='#90EE90', color='#006400', fontcolor='black')
    dot.node('D5', 'Dec5\n512+512→512\nL:32→64')
    dot.node('D4', 'Dec4\n512+256→256\nL:64→128')
    dot.node('D3', 'Dec3\n256+128→128\nL:128→256')
    dot.node('D2', 'Dec2\n128+64→64\nL:256→512')
    dot.node('D1', 'Dec1\n64+32→32\nL:512→1024')
    
    # Final projection
    dot.attr('node', fillcolor='#228B22', fontcolor='white')
    dot.node('Final', 'Final Conv\n32→2\ntanh')

    # Discriminator blocks (Red gradient)
    dot.attr('node', fillcolor='#FFB6C1', color='#8B0000', fontcolor='black')
    dot.node('D_C1', 'D-Conv1\n4→32\nL:1024→512')
    dot.node('D_C2', 'D-Conv2\n32→64\nL:512→256')
    dot.node('D_C3', 'D-Conv3\n64→128\nL:256→128')
    dot.node('D_C4', 'D-Conv4\n128→256\nL:128→64')
    dot.node('D_C5', 'D-Conv5\n256→512\nL:64→32')
    dot.node('D_C6', 'D-Conv6\n512→512\nL:32→16')
    
    dot.attr('node', fillcolor='#DC143C', fontcolor='white')
    dot.node('D_Pool', 'Global Sum Pool\n+\nDense→1')

    # --- Define Edges ---
    
    # Generator main path (encoder)
    dot.edge('Noisy_Input', 'E1', label='')
    dot.edge('E1', 'E2')
    dot.edge('E2', 'E3')
    dot.edge('E3', 'E4')
    dot.edge('E4', 'E5')
    dot.edge('E5', 'BN')
    
    # Generator main path (decoder)
    dot.edge('BN', 'D5')
    dot.edge('D5', 'D4')
    dot.edge('D4', 'D3')
    dot.edge('D3', 'D2')
    dot.edge('D2', 'D1')
    dot.edge('D1', 'Final')
    dot.edge('Final', 'Generated')

    # Skip connections (dashed, labeled with +)
    skip_style = {
        'style': 'dashed', 
        'color': '#4682B4', 
        'fontcolor': '#4682B4', 
        'label': '+',
        'constraint': 'false'
    }
    dot.edge('E1', 'D1', **skip_style)
    dot.edge('E2', 'D2', **skip_style)
    dot.edge('E3', 'D3', **skip_style)
    dot.edge('E4', 'D4', **skip_style)
    dot.edge('E5', 'D5', **skip_style)

    # Discriminator path
    # Conditional: concatenate noisy input with candidate
    dot.edge('Noisy_Input', 'D_C1', style='dotted', label='Condition', color='#FF4500', fontcolor='#FF4500')
    dot.edge('Generated', 'D_C1', label='Fake', color='#FF6347')
    dot.edge('Clean_Target', 'D_C1', label='Real', color='#228B22')
    
    # Discriminator internal flow
    dot.edge('D_C1', 'D_C2')
    dot.edge('D_C2', 'D_C3')
    dot.edge('D_C3', 'D_C4')
    dot.edge('D_C4', 'D_C5')
    dot.edge('D_C5', 'D_C6')
    dot.edge('D_C6', 'D_Pool')
    dot.edge('D_Pool', 'Score')

    # Subgraphs for grouping
    with dot.subgraph(name='cluster_generator') as g:
        g.attr(label='Generator (1D U-Net)', color='blue', style='dashed', fontsize='14')
        g.node('E1')
        g.node('E5')
        g.node('BN')
        g.node('D5')
        g.node('D1')
        g.node('Final')
        
    with dot.subgraph(name='cluster_discriminator') as d:
        d.attr(label='Discriminator (Critic)', color='red', style='dashed', fontsize='14')
        d.node('D_C1')
        d.node('D_C6')
        d.node('D_Pool')

    # Render
    try:
        output_file = dot.render(output_path, view=view, format=format, cleanup=True)
        print(f"Generated: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error rendering: {e}")
        print("Make sure Graphviz is installed and in PATH")
        # Save DOT source anyway
        dot.save(f"{output_path}.gv")
        print(f"Saved DOT source to: {output_path}.gv")
        return f"{output_path}.gv"


def draw_generator_detailed(
    output_path: str = 'generator_detailed',
    view: bool = True,
    format: str = 'png'
) -> str:
    """
    Draw detailed generator (U-Net) architecture.
    
    Shows all layers with exact channel/length specifications.
    """
    dot = graphviz.Digraph('Generator_Detailed', comment='1D U-Net Generator Details')
    
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.3', ranksep='0.4')
    dot.attr('node', fontname='Courier', fontsize='10', shape='record')
    
    # Title
    dot.attr('node', shape='plaintext', fontsize='16', fontname='Helvetica-Bold')
    dot.node('title', '1D U-Net Generator Architecture')
    
    # Layer specifications
    layers = [
        # Encoder
        ('enc1_1', 'Conv1D(2→32, k=3, s=2)', '1024→512', '#E6F3FF'),
        ('enc1_2', 'Conv1D(32→32, k=3, s=1)', '512', '#E6F3FF'),
        ('enc2_1', 'Conv1D(32→64, k=3, s=2)', '512→256', '#CCE5FF'),
        ('enc2_2', 'Conv1D(64→64, k=3, s=1)', '256', '#CCE5FF'),
        ('enc3_1', 'Conv1D(64→128, k=3, s=2)', '256→128', '#B3D9FF'),
        ('enc3_2', 'Conv1D(128→128, k=3, s=1)', '128', '#B3D9FF'),
        ('enc4_1', 'Conv1D(128→256, k=3, s=2)', '128→64', '#99CCFF'),
        ('enc4_2', 'Conv1D(256→256, k=3, s=1)', '64', '#99CCFF'),
        ('enc5_1', 'Conv1D(256→512, k=3, s=2)', '64→32', '#80BFFF'),
        ('enc5_2', 'Conv1D(512→512, k=3, s=1)', '32', '#80BFFF'),
        # Bottleneck
        ('bottle1', 'Conv1D(512→512, k=3, s=1)', '32', '#4169E1'),
        ('bottle2', 'Conv1D(512→512, k=3, s=1)', '32', '#4169E1'),
        # Decoder
        ('dec5_up', 'Upsample(×2) + Add Skip', '32→64', '#90EE90'),
        ('dec5_1', 'Conv1D(512→512, k=3, s=1)', '64', '#90EE90'),
        ('dec5_2', 'Conv1D(512→512, k=3, s=1)', '64', '#90EE90'),
        ('dec4_up', 'Upsample(×2) + Add Skip', '64→128', '#7CCD7C'),
        ('dec4_1', 'Conv1D(512→256, k=3, s=1)', '128', '#7CCD7C'),
        ('dec4_2', 'Conv1D(256→256, k=3, s=1)', '128', '#7CCD7C'),
        ('dec3_up', 'Upsample(×2) + Add Skip', '128→256', '#66CD66'),
        ('dec3_1', 'Conv1D(256→128, k=3, s=1)', '256', '#66CD66'),
        ('dec3_2', 'Conv1D(128→128, k=3, s=1)', '256', '#66CD66'),
        ('dec2_up', 'Upsample(×2) + Add Skip', '256→512', '#4CBB4C'),
        ('dec2_1', 'Conv1D(128→64, k=3, s=1)', '512', '#4CBB4C'),
        ('dec2_2', 'Conv1D(64→64, k=3, s=1)', '512', '#4CBB4C'),
        ('dec1_up', 'Upsample(×2) + Add Skip', '512→1024', '#32CD32'),
        ('dec1_1', 'Conv1D(64→32, k=3, s=1)', '1024', '#32CD32'),
        ('dec1_2', 'Conv1D(32→32, k=3, s=1)', '1024', '#32CD32'),
        ('final', 'Conv1D(32→2, k=3, s=1) + tanh', '1024', '#228B22'),
    ]
    
    dot.attr('node', shape='box', style='filled')
    
    prev_node = 'title'
    for name, desc, length, color in layers:
        label = f"{name}|{desc}|L: {length}"
        dot.node(name, label, fillcolor=color, fontcolor='black' if color != '#4169E1' else 'white')
        dot.edge(prev_node, name)
        prev_node = name
        
    # Render
    try:
        output_file = dot.render(output_path, view=view, format=format, cleanup=True)
        print(f"Generated: {output_file}")
        return output_file
    except Exception as e:
        dot.save(f"{output_path}.gv")
        return f"{output_path}.gv"


def draw_discriminator_detailed(
    output_path: str = 'discriminator_detailed',
    view: bool = True,
    format: str = 'png'
) -> str:
    """
    Draw detailed discriminator (critic) architecture.
    """
    dot = graphviz.Digraph('Discriminator_Detailed', comment='1D CNN Discriminator Details')
    
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.3', ranksep='0.4')
    dot.attr('node', fontname='Courier', fontsize='10', shape='record')
    
    # Title
    dot.attr('node', shape='plaintext', fontsize='16', fontname='Helvetica-Bold')
    dot.node('title', 'Conditional Discriminator (Critic)')
    
    # Input explanation
    dot.attr('node', shape='box', style='filled', fillcolor='#FFE4E1')
    dot.node('input_explain', 
             'Input: concat(Candidate[2], Condition[2]) = [4×1024]\\n'
             'Candidate = Real or Fake signal\\n'
             'Condition = Noisy input signal')
    
    # Layers
    layers = [
        ('concat', 'Concatenate(Candidate, Condition)', '[4, 1024]', '#FFE4E1'),
        ('c1', 'Conv1D(4→32, k=3, s=2) + LeakyReLU', '[32, 512]', '#FFB6C1'),
        ('c2', 'Conv1D(32→64, k=3, s=2) + LeakyReLU', '[64, 256]', '#FF9999'),
        ('c3', 'Conv1D(64→128, k=3, s=2) + LeakyReLU', '[128, 128]', '#FF7F7F'),
        ('c4', 'Conv1D(128→256, k=3, s=2) + LeakyReLU', '[256, 64]', '#FF6666'),
        ('c5', 'Conv1D(256→512, k=3, s=2) + LeakyReLU', '[512, 32]', '#FF4C4C'),
        ('c6', 'Conv1D(512→512, k=3, s=2) + LeakyReLU', '[512, 16]', '#FF3333'),
        ('pool', 'Global Sum Pooling', '[512]', '#DC143C'),
        ('dense', 'Dense(512→1)', '[1]', '#8B0000'),
    ]
    
    dot.attr('node', shape='box', style='filled')
    dot.edge('title', 'input_explain')
    
    prev_node = 'input_explain'
    for name, desc, shape, color in layers:
        fontcolor = 'white' if color in ['#DC143C', '#8B0000'] else 'black'
        dot.node(name, f"{desc}\\n→ Shape: {shape}", fillcolor=color, fontcolor=fontcolor)
        dot.edge(prev_node, name)
        prev_node = name
        
    # Render
    try:
        output_file = dot.render(output_path, view=view, format=format, cleanup=True)
        print(f"Generated: {output_file}")
        return output_file
    except Exception as e:
        dot.save(f"{output_path}.gv")
        return f"{output_path}.gv"


def draw_training_flow(
    output_path: str = 'training_flow',
    view: bool = True,
    format: str = 'png'
) -> str:
    """
    Draw the CWGAN-GP training data flow diagram.
    """
    dot = graphviz.Digraph('Training_Flow', comment='CWGAN-GP Training Flow')
    
    dot.attr(rankdir='TB', splines='curved', nodesep='0.5')
    dot.attr('node', fontname='Helvetica', fontsize='11')
    
    # Data nodes
    dot.attr('node', shape='cylinder', style='filled', fillcolor='#E8E8E8')
    dot.node('data', 'Training Data\n(Image Files)')
    
    # Process nodes
    dot.attr('node', shape='box', fillcolor='#FFFACD')
    dot.node('img2ofdm', 'Image → OFDM\n(QAM Modulation)')
    dot.node('channel', 'Channel Model\n(AWGN/Rayleigh)')
    
    # Signal nodes
    dot.attr('node', shape='ellipse', fillcolor='#98FB98')
    dot.node('clean', 'Clean OFDM\n(Target)')
    dot.node('noisy', 'Noisy OFDM\n(Input)')
    
    # Generator
    dot.attr('node', shape='box', fillcolor='#ADD8E6')
    dot.node('gen', 'Generator G\n(U-Net)')
    
    dot.attr('node', shape='ellipse', fillcolor='#87CEEB')
    dot.node('fake', 'Enhanced OFDM\n(Fake)')
    
    # Discriminator
    dot.attr('node', shape='box', fillcolor='#FFB6C1')
    dot.node('disc', 'Discriminator D\n(Critic)')
    
    # Loss nodes
    dot.attr('node', shape='diamond', fillcolor='#DDA0DD')
    dot.node('d_loss', 'Critic Loss\nE[D(real)] - E[D(fake)]\n+ λ·GP')
    dot.node('g_loss', 'Generator Loss\n-E[D(fake)]\n+ λ_rec·L1(fake, real)')
    
    # Update nodes
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#FFA07A')
    dot.node('d_update', 'Update D\n(5 steps)')
    dot.node('g_update', 'Update G\n(1 step)')
    
    # Edges
    dot.edge('data', 'img2ofdm')
    dot.edge('img2ofdm', 'clean')
    dot.edge('clean', 'channel')
    dot.edge('channel', 'noisy')
    
    # Generator path
    dot.edge('noisy', 'gen')
    dot.edge('gen', 'fake')
    
    # Discriminator inputs
    dot.edge('noisy', 'disc', label='condition', style='dashed')
    dot.edge('clean', 'disc', label='real')
    dot.edge('fake', 'disc', label='fake')
    
    # Losses
    dot.edge('disc', 'd_loss')
    dot.edge('disc', 'g_loss')
    dot.edge('clean', 'g_loss', label='L1 target', style='dotted')
    dot.edge('fake', 'g_loss', style='dotted')
    
    # Updates
    dot.edge('d_loss', 'd_update')
    dot.edge('g_loss', 'g_update')
    
    # Render
    try:
        output_file = dot.render(output_path, view=view, format=format, cleanup=True)
        print(f"Generated: {output_file}")
        return output_file
    except Exception as e:
        dot.save(f"{output_path}.gv")
        return f"{output_path}.gv"


def draw_fpga_overview(
    output_path: str = 'fpga_overview',
    view: bool = True,
    format: str = 'png'
) -> str:
    """
    Draw FPGA implementation overview.
    """
    dot = graphviz.Digraph('FPGA_Overview', comment='FPGA Generator Implementation')
    
    dot.attr(rankdir='LR', splines='polyline', nodesep='0.4')
    dot.attr('node', fontname='Helvetica', fontsize='10')
    
    # External components
    dot.attr('node', shape='box3d', style='filled', fillcolor='#E6E6FA')
    dot.node('ddr', 'DDR Memory\n(Weights: 5.5MB)\n(INT8)')
    dot.node('input_buffer', 'Input Buffer\n(I/Q Samples)\n(INT16)')
    dot.node('output_buffer', 'Output Buffer\n(Enhanced I/Q)\n(INT16)')
    
    # Processing blocks
    dot.attr('node', shape='box', fillcolor='#87CEEB')
    dot.node('dma', 'DMA Controller\n(Weight Fetch)')
    dot.node('conv_engine', 'Conv1D Engine\n(DSP48 Array)\n~100 DSPs')
    dot.node('skip_ram', 'Skip Buffer RAM\n(160 KB BRAM)')
    dot.node('upsample', 'Upsample Unit\n(×2 NN)')
    dot.node('activation', 'Activation Unit\n(LeakyReLU LUT)')
    dot.node('tanh_lut', 'Tanh LUT\n(256 entries)')
    
    # Control
    dot.attr('node', shape='hexagon', fillcolor='#98FB98')
    dot.node('ctrl', 'FSM Controller\n(Layer Sequencer)')
    
    # Edges
    dot.edge('ddr', 'dma')
    dot.edge('dma', 'conv_engine', label='weights')
    dot.edge('input_buffer', 'conv_engine', label='input')
    dot.edge('conv_engine', 'activation')
    dot.edge('activation', 'skip_ram', style='dashed', label='encoder')
    dot.edge('skip_ram', 'conv_engine', style='dashed', label='decoder')
    dot.edge('conv_engine', 'upsample', label='decoder')
    dot.edge('upsample', 'conv_engine')
    dot.edge('conv_engine', 'tanh_lut', label='final')
    dot.edge('tanh_lut', 'output_buffer')
    
    dot.edge('ctrl', 'dma', style='dotted')
    dot.edge('ctrl', 'conv_engine', style='dotted')
    dot.edge('ctrl', 'skip_ram', style='dotted')
    
    # Subgraph for FPGA
    with dot.subgraph(name='cluster_fpga') as f:
        f.attr(label='FPGA (Xilinx ZCU104)', color='black', style='bold')
        f.node('dma')
        f.node('conv_engine')
        f.node('skip_ram')
        f.node('upsample')
        f.node('activation')
        f.node('tanh_lut')
        f.node('ctrl')
        
    # Render
    try:
        output_file = dot.render(output_path, view=view, format=format, cleanup=True)
        print(f"Generated: {output_file}")
        return output_file
    except Exception as e:
        dot.save(f"{output_path}.gv")
        return f"{output_path}.gv"


def generate_all_diagrams(
    output_dir: str = './diagrams',
    view: bool = False,
    format: str = 'png'
):
    """
    Generate all architecture diagrams.
    
    Args:
        output_dir: Directory to save diagrams
        view: Whether to open diagrams after generation
        format: Output format
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating architecture diagrams...")
    print("=" * 50)
    
    diagrams = [
        ('Full GAN Architecture', draw_full_architecture, 'gan_architecture'),
        ('Generator Detailed', draw_generator_detailed, 'generator_detailed'),
        ('Discriminator Detailed', draw_discriminator_detailed, 'discriminator_detailed'),
        ('Training Flow', draw_training_flow, 'training_flow'),
        ('FPGA Overview', draw_fpga_overview, 'fpga_overview'),
    ]
    
    for name, func, filename in diagrams:
        print(f"\n{name}...")
        try:
            func(str(output_path / filename), view=view, format=format)
        except Exception as e:
            print(f"  Error: {e}")
            
    print("\n" + "=" * 50)
    print(f"Diagrams saved to: {output_path.absolute()}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CWGAN-GP Architecture Visualization")
    print("=" * 60)
    
    # Generate all diagrams
    generate_all_diagrams(view=False)
    
    print("\n✓ Diagram generation complete!")
