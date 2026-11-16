# Device Management

Understanding and managing Qualcomm Cloud AI device topology.

## Cloud AI 100 Ultra Topology

Each Cloud AI 100 Ultra card contains **4 QIDs** (Qualcomm device IDs), and each QID has **16 NSP cores**:

```text
Cloud AI 100 Ultra Card (1 board serial)
+-- QID 0  (16 NSP cores)
+-- QID 1  (16 NSP cores)
+-- QID 2  (16 NSP cores)
+-- QID 3  (16 NSP cores)
```

Multiple cards in a system provide additional QID groups (e.g., QIDs 0-3, 4-7, 8-11...).

## Inspecting Device Status

```bash
# View all devices, their status, and board serial numbers
/opt/qti-aic/tools/qaic-util | grep -i "qid\|status\|board serial"

# Example output:
# QID 0: Status Ready, Board Serial ABC123
# QID 1: Status Ready, Board Serial ABC123
# QID 2: Status Busy, Board Serial ABC123
# QID 3: Status Ready, Board Serial ABC123
```

## Device Visibility

Control which QIDs are accessible to your process:

```bash
# Single device
export QAIC_VISIBLE_DEVICES=0

# Multiple devices (tensor parallel)
export QAIC_VISIBLE_DEVICES=0,1,2,3

# Specific devices on a multi-card system
export QAIC_VISIBLE_DEVICES=4,5,6,7
```

## Device Group Configuration

The `device_group` field in `additional_config` specifies which QID(s) to use:

```python
# Single QID
additional_config={"override_qaic_config": {"device_group": [0]}}

# Tensor parallel across 4 QIDs (one Ultra card)
additional_config={"override_qaic_config": {"device_group": [0, 1, 2, 3]}}
```

!!! tip "Same board serial"
    For tensor-parallel execution across multiple QIDs, always select QIDs that share
    the same board serial number (i.e., on the same physical card) for optimal
    inter-device communication.

## Core Allocation

Each QID has 16 NSP cores. For speculative decoding, cores are split between target and draft models:

```python
# 10 cores for target, 6 for draft (on same QID)
additional_config={
    "override_qaic_config": {"device_group": [0], "num_cores": 10},
    "draft_override_qaic_config": {"device_group": [0], "num_cores": 6},
}
```

## Multi-Card Deployment

For large models requiring more than one card:

```bash
# Make all QIDs on two cards visible
export QAIC_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Configure tensor parallel across 8 QIDs
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --additional-config '{"override_qaic_config":{"device_group":[0,1,2,3,4,5,6,7]}}'
```

## Device Recovery

If a device is busy or unavailable:

1. Check status: `/opt/qti-aic/tools/qaic-util | grep -i "qid\|status\|board serial"`
2. Find free QIDs that share a board serial
3. Update `QAIC_VISIBLE_DEVICES` and `device_group` accordingly
4. Retry your workload

If a device is in **Error** state, perform an SOC reset:

```bash
sudo /opt/qti-aic/tools/qaic-util -s
```

!!! warning
    An SOC reset will affect all workloads running on the card. Ensure no other processes are using the device before resetting.
