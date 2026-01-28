#!/usr/bin/env python3
"""
Generate instruction file and run script for a new kernel workspace.

Usage:
    python scripts/generate_kernel_setup.py topk_workspace --gpu 3 --agent-num 3
    python scripts/generate_kernel_setup.py mykernel_workspace --gpu 2 --agent-num 10 --original-path "ops/mykernel.py"
"""

import argparse
import sys
from pathlib import Path


def generate_instruction_file(
    workspace_dir: str,
    kernel_name: str,
    gpu_device: str,
    agent_num: int,
    original_path: str,
    output_path: Path
) -> None:
    """Generate agent instruction file from template."""
    
    template_path = Path(__file__).parent.parent / "KERNEL_TEMPLATE.md"
    
    if not template_path.exists():
        print(f"Error: Template not found at {template_path}")
        sys.exit(1)
    
    template = template_path.read_text()
    
    # Replace placeholders
    content = template.replace("{KERNEL_NAME}", kernel_name)
    content = content.replace("{GPU_DEVICE}", gpu_device)
    content = content.replace("{WORKSPACE_DIR}", workspace_dir)
    content = content.replace("{ORIGINAL_PATH}", original_path)
    content = content.replace("{KERNEL_SLUG}", kernel_name.lower().replace(" ", "_"))
    
    output_path.write_text(content)
    print(f"✓ Created instruction file: {output_path}")


def generate_run_script(
    workspace_dir: str,
    kernel_name: str,
    gpu_device: str,
    iterations: int,
    output_path: Path
) -> None:
    """Generate run script from template."""
    
    template_path = Path(__file__).parent.parent / "RUN_SCRIPT_TEMPLATE.sh"
    
    if not template_path.exists():
        print(f"Error: Template not found at {template_path}")
        sys.exit(1)
    
    template = template_path.read_text()
    
    # Replace placeholders
    content = template.replace("{KERNEL_NAME}", kernel_name)
    content = content.replace("{GPU_DEVICE}", gpu_device)
    content = content.replace("{WORKSPACE_DIR}", workspace_dir)
    content = content.replace("{ITERATIONS}", str(iterations))
    
    output_path.write_text(content)
    output_path.chmod(0o755)  # Make executable
    print(f"✓ Created run script: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate instruction file and run script for a kernel workspace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate files for topk_workspace on GPU 3
  python scripts/generate_kernel_setup.py topk_workspace --gpu 3 --agent-num 3

  # Generate files for new mykernel_workspace on GPU 2
  python scripts/generate_kernel_setup.py mykernel_workspace \\
      --gpu 2 \\
      --agent-num 10 \\
      --kernel-name "MyKernel" \\
      --original-path "ops/triton/mykernel.py"
        """
    )
    
    parser.add_argument(
        "workspace_dir",
        help="Workspace directory name (e.g., topk_workspace)"
    )
    parser.add_argument(
        "--gpu", "-g",
        default="3",
        help="GPU device number (default: 3)"
    )
    parser.add_argument(
        "--agent-num", "-a",
        type=int,
        required=True,
        help="Agent number for instruction file (e.g., 3 for AGENT_3_*.md)"
    )
    parser.add_argument(
        "--kernel-name", "-k",
        help="Kernel name (default: derived from workspace_dir)"
    )
    parser.add_argument(
        "--original-path", "-o",
        default="",
        help="Path to original kernel in aiter repo (e.g., 'ops/triton/topk.py')"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=50,
        help="OpenEvolve iterations (default: 50)"
    )
    parser.add_argument(
        "--no-run-script",
        action="store_true",
        help="Skip generating run script (only create instruction file)"
    )
    
    args = parser.parse_args()
    
    # Derive kernel name from workspace_dir if not provided
    workspace_dir = args.workspace_dir.rstrip("/")
    if args.kernel_name:
        kernel_name = args.kernel_name
    else:
        # Convert "topk_workspace" -> "TopK"
        kernel_name = workspace_dir.replace("_workspace", "").replace("_", " ").title()
    
    # Set up paths
    repo_root = Path(__file__).parent.parent
    workspace_path = repo_root / workspace_dir
    
    # Check if workspace exists, create if needed
    if not workspace_path.exists():
        print(f"Creating workspace directory: {workspace_path}")
        workspace_path.mkdir(parents=True, exist_ok=True)
    
    # Put generated files inside the workspace
    instruction_path = workspace_path / f"AGENT_{args.agent_num}_{kernel_name.upper().replace(' ', '_')}.md"
    run_script_path = workspace_path / "run.sh"
    
    # Generate files
    print(f"\nGenerating setup for: {kernel_name}")
    print(f"  Workspace: {workspace_dir}")
    print(f"  GPU: {args.gpu}")
    print(f"  Agent: {args.agent_num}")
    print()
    
    generate_instruction_file(
        workspace_dir=workspace_dir,
        kernel_name=kernel_name,
        gpu_device=args.gpu,
        agent_num=args.agent_num,
        original_path=args.original_path,
        output_path=instruction_path
    )
    
    if not args.no_run_script:
        generate_run_script(
            workspace_dir=workspace_dir,
            kernel_name=kernel_name,
            gpu_device=args.gpu,
            iterations=args.iterations,
            output_path=run_script_path
        )
    
    print("\n✓ Setup complete!")
    print(f"\nNext steps:")
    kernel_py_path = workspace_path / "kernel.py"
    if not kernel_py_path.exists():
        print(f"  1. Add kernel: cp <source_kernel.py> {workspace_dir}/kernel.py")
    print(f"  2. Reference instruction: @{workspace_dir}/{instruction_path.name}")
    if not args.no_run_script:
        print(f"  3. Run optimization: ./{workspace_dir}/run.sh")


if __name__ == "__main__":
    main()
