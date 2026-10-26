#!/usr/bin/env python3
"""Argon Engine Model Context Protocol (MCP) server for Google Antigravity & Gemini.

Implements the standard JSON-RPC 2.0 stdio transport protocol for MCP, enabling
LLM agents to inspect hardware, run cross-backend parity checks, execute neuromorphic
ticks, and tune homeostatic weights directly.
"""

import json
import os
import subprocess
import sys

# Ensure bindings/python is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bindings", "python")))

ENGINE_INSTANCE = None


def get_engine():
    global ENGINE_INSTANCE
    if ENGINE_INSTANCE is None:
        from argon_engine import ArgonEngine
        ENGINE_INSTANCE = ArgonEngine(num_layers=2, tile_dim=8)
    return ENGINE_INSTANCE


TOOLS = [
    {
        "name": "argon_inspect",
        "description": "Probes local hardware, GPU acceleration platforms (Vulkan, Metal, ROCm, CUDA), and available local model servers.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "argon_verify_parity",
        "description": "Executes the cross-backend ADR-0001 golden-vector parity verification suite across CPU, Vulkan, Metal, and ROCm.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "argon_step",
        "description": "Executes one synchronous zero-copy Forward-Forward inference and plasticity tick on resident SRAM.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Input feature vector matching tile_dim (default 8).",
                },
                "polarity": {
                    "type": "number",
                    "description": "Phase polarity (1.0 for positive stimulus, -1.0 for negative distractor).",
                    "default": 1.0,
                },
            },
            "required": ["features"],
        },
    },
    {
        "name": "argon_weight_checksum",
        "description": "Calculates the deterministic 64-bit checksum of all synaptic weights resident in Argon.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "argon_scale_weights",
        "description": "Rescales synaptic weights across all layers towards a target homeostatic norm.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_norm": {
                    "type": "number",
                    "description": "Target L2 Frobenius norm for synaptic weight matrices.",
                }
            },
            "required": ["target_norm"],
        },
    },
]


def handle_tool_call(name: str, arguments: dict) -> str:
    if name == "argon_inspect":
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        exe_name = "argon-cli.exe" if sys.platform == "win32" else "argon-cli"
        cli_candidates = [
            os.path.join(workspace_root, "target", "debug", exe_name),
            os.path.join(workspace_root, "target", "release", exe_name),
        ]
        cli_path = next((p for p in cli_candidates if os.path.exists(p)), None)
        if cli_path:
            out = subprocess.check_output([cli_path, "inspect", "--format", "json"], text=True)
            return out.strip()
        return json.dumps({"status": "available", "info": "argon-cli not pre-built in target/"})

    elif name == "argon_verify_parity":
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        exe_name = "argon-cli.exe" if sys.platform == "win32" else "argon-cli"
        cli_candidates = [
            os.path.join(workspace_root, "target", "debug", exe_name),
            os.path.join(workspace_root, "target", "release", exe_name),
        ]
        cli_path = next((p for p in cli_candidates if os.path.exists(p)), None)
        if cli_path:
            out = subprocess.check_output([cli_path, "verify-parity", "--format", "json"], text=True)
            return out.strip()
        return json.dumps({"error": "argon-cli executable not found"})

    elif name == "argon_step":
        import numpy as np
        engine = get_engine()
        features = np.array(arguments.get("features", []), dtype=np.float32)
        polarity = float(arguments.get("polarity", 1.0))
        output, telemetry = engine.step(features, polarity=polarity)
        return json.dumps({
            "sequence_id": telemetry.sequence_id,
            "mean_goodness": telemetry.mean_goodness,
            "output_l2_norm": telemetry.output_l2_norm,
            "total_ticks": telemetry.total_ticks,
            "activations": output.tolist(),
        })

    elif name == "argon_weight_checksum":
        engine = get_engine()
        checksum = engine.weight_checksum()
        return json.dumps({"weight_checksum": hex(checksum), "decimal": checksum})

    elif name == "argon_scale_weights":
        engine = get_engine()
        target = float(arguments.get("target_norm", 1.0))
        engine.scale_weights(target)
        new_checksum = engine.weight_checksum()
        return json.dumps({"status": "scaled", "new_checksum": hex(new_checksum)})

    else:
        raise ValueError(f"Unknown tool: {name}")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "argon-mcp-server", "version": "0.1.0"},
                },
            }
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS},
            }
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            try:
                result_text = handle_tool_call(tool_name, tool_args)
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": result_text}],
                        "isError": False,
                    },
                }
            except Exception as e:
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error executing tool: {str(e)}"}],
                        "isError": True,
                    },
                }
        else:
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
