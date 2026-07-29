#!/usr/bin/env python3
from pathlib import Path
import xml.etree.ElementTree as ET


SDK_JOINT_ORDER = [
    f"{leg}_{joint}"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
]


def _names(root: ET.Element, section: str) -> list[str]:
    node = root.find(section)
    if node is None:
        raise RuntimeError(f"Missing <{section}> in Go2 MJCF")
    return [child.attrib["name"] for child in node]


def main() -> int:
    root = ET.parse(Path("robots/go2/mjcf/robot/go2.xml")).getroot()
    actuator_names = _names(root, "actuator")
    sensor_names = _names(root, "sensor")

    checks = {
        "actuator": actuator_names[:12] == SDK_JOINT_ORDER,
        "jointpos": sensor_names[:12] == [f"{name}_pos" for name in SDK_JOINT_ORDER],
        "jointvel": sensor_names[12:24] == [f"{name}_vel" for name in SDK_JOINT_ORDER],
        "torque": sensor_names[24:36] == [f"{name}_torque" for name in SDK_JOINT_ORDER],
    }
    for name, ok in checks.items():
        print(f"{name}_order_ok={ok}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
