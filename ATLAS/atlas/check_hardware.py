#!/usr/bin/env python
"""
ATLAS Hardware Checker

Check hardware capabilities and get recommendations for ATLAS configuration.
Run with: python -m atlas.check_hardware
"""

import sys


def check_hardware(verbose: bool = True) -> dict:
    """
    Check hardware capabilities and return recommendations.

    Args:
        verbose: If True, print detailed information

    Returns:
        Dictionary with hardware info and recommendations
    """
    try:
        from atlas.utils.hardware import (
            detect_hardware_capabilities,
            print_hardware_info,
            get_hardware_info,
        )
    except ImportError:
        print("ERROR: Could not import ATLAS hardware utilities.")
        print("Ensure ATLAS is properly installed: pip install -e .")
        sys.exit(1)

    if verbose:
        print_hardware_info()

    info = get_hardware_info()
    caps = detect_hardware_capabilities()

    # Generate recommendations
    recommendations = _generate_recommendations(caps)

    if verbose:
        _print_recommendations(recommendations)

    return {
        "info": info,
        "capabilities": caps,
        "recommendations": recommendations,
    }


def _generate_recommendations(caps) -> dict:
    """Generate configuration recommendations based on hardware."""
    recs = {
        "resolution": 512,
        "batch_size": 1,
        "kernel_solver": "rff",
        "use_mixed_precision": False,
        "enable_cuda_graphs": False,
        "enable_tiling": False,
        "rff_features": 2048,
        "warnings": [],
        "tips": [],
    }

    # CPU recommendations
    if caps.device_type == "cpu":
        recs["resolution"] = 256
        recs["batch_size"] = 1
        recs["kernel_solver"] = "rff"
        recs["rff_features"] = 512
        recs["warnings"].append(
            "Running on CPU: expect 5-10x slower performance than GPU"
        )
        recs["tips"].append("Consider using a CUDA GPU for better performance")
        recs["tips"].append("Reduce resolution to 256-512px for reasonable speed")
        return recs

    # GPU recommendations based on memory
    free_gb = caps.free_memory_gb

    if free_gb >= 20:
        recs["resolution"] = 1536
        recs["batch_size"] = 8
        recs["enable_tiling"] = True  # For >1536
        recs["tips"].append("High memory available: native 2K support enabled")
    elif free_gb >= 14:
        recs["resolution"] = 1024
        recs["batch_size"] = 4
        recs["tips"].append("Good memory: 1K generation at batch 4")
    elif free_gb >= 10:
        recs["resolution"] = 1024
        recs["batch_size"] = 2
        recs["tips"].append("Medium memory: 1K generation at batch 2")
    elif free_gb >= 6:
        recs["resolution"] = 512
        recs["batch_size"] = 2
        recs["rff_features"] = 1024
        recs["warnings"].append("Limited memory: reduced to 512px")
    else:
        recs["resolution"] = 512
        recs["batch_size"] = 1
        recs["rff_features"] = 1024
        recs["warnings"].append("Very limited memory: 512px at batch 1")
        recs["tips"].append("Consider upgrading GPU for higher resolutions")

    # Precision recommendations
    if caps.bf16_supported:
        recs["use_mixed_precision"] = True
        recs["precision"] = "bf16"
        recs["tips"].append("BF16 + TF32 acceleration available")
    elif caps.fp16_supported:
        recs["use_mixed_precision"] = True
        recs["precision"] = "fp16"
        recs["tips"].append("FP16 mixed precision available")
    else:
        recs["precision"] = "fp32"
        recs["warnings"].append("No mixed precision support - upgrade to Volta+ GPU")

    # CUDA graphs
    if caps.cuda_graphs_supported and free_gb >= 8:
        recs["enable_cuda_graphs"] = True
        recs["tips"].append("CUDA graphs available (10-30% speedup)")
    elif caps.cuda_graphs_supported:
        recs["warnings"].append("CUDA graphs available but limited memory")

    # Kernel solver
    if recs["resolution"] >= 768:
        recs["kernel_solver"] = "fft"
        recs["tips"].append("FFT kernels optimal for grid-structured data")
    else:
        recs["kernel_solver"] = "rff"

    return recs


def _print_recommendations(recs: dict) -> None:
    """Print formatted recommendations."""
    print("\n" + "=" * 60)
    print("ATLAS Configuration Recommendations")
    print("=" * 60)

    print("\nRecommended Settings:")
    print(f"  Resolution: {recs['resolution']}px")
    print(f"  Batch Size: {recs['batch_size']}")
    print(f"  Kernel Solver: {recs['kernel_solver']}")
    print(f"  RFF Features: {recs['rff_features']}")
    print(f"  Precision: {recs.get('precision', 'fp32')}")
    print(f"  Mixed Precision: {'Enabled' if recs['use_mixed_precision'] else 'Disabled'}")
    print(f"  CUDA Graphs: {'Enabled' if recs['enable_cuda_graphs'] else 'Disabled'}")

    if recs["enable_tiling"]:
        print(f"  Tiling: Enabled (for >{recs['resolution']}px)")

    if recs["warnings"]:
        print("\n  Warnings:")
        for warning in recs["warnings"]:
            print(f"  - {warning}")

    if recs["tips"]:
        print("\n Tips:")
        for tip in recs["tips"]:
            print(f"  - {tip}")

    print("\nExample Configuration:")
    print("```python")
    print("from atlas.easy_api import create_sampler")
    print()
    print("sampler = create_sampler(")
    print("    checkpoint='model.pt',")
    print("    gpu_memory='auto',  # Or specify: '8GB', '16GB', '24GB'")
    print(f"    resolution={recs['resolution']},")
    print(f"    batch_size={recs['batch_size']},")
    if recs["enable_cuda_graphs"]:
        print("    enable_cuda_graphs=True,")
    if recs["enable_tiling"]:
        print("    tile_size=512,")
        print("    tile_overlap=0.125,")
    print(")")
    print("```")
    print("=" * 60 + "\n")


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check ATLAS hardware capabilities and get recommendations"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only show summary, not detailed info",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )

    args = parser.parse_args()

    result = check_hardware(verbose=not args.quiet)

    if args.json:
        import json

        # Convert dataclass to dict
        output = {
            "device": result["info"]["device"],
            "device_name": result["info"]["device_name"],
            "free_memory_gb": result["info"]["free_memory_gb"],
            "recommendations": result["recommendations"],
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

