"""生成 H1 训练 XML，将停用的定义完整保留在 XML 注释中。

从仓库根目录运行：
    uv run python scripts/data/prepare_h1_asset.py

不修改源文件，不覆盖已有训练资产。模型先通过 CPU MuJoCo 检查，再写入文件。
"""

from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "h1_data/h1.xml"
DESTINATION = ROOT / "h1_data/h1_training.xml"

DISABLED_SECTIONS = {
    "actuator": "原 motor 停用；位置执行器由 robots/h1.py 的 mjlab 配置添加。",
    "sensor": "原传感器定义保留；当前训练从仿真状态构造观测。",
    "keyframe": "原关键帧保留；默认姿态由 robots/h1.py 的 InitialStateCfg 定义。",
}


def _comment_out(parent: ET.Element, element: ET.Element, reason: str) -> None:
    """用含有完整原定义的注释替代生效节点，不丢弃其内容。"""
    saved = copy.deepcopy(element)
    saved.tail = None
    original = ET.tostring(saved, encoding="unicode")
    # XML 注释不能嵌套；内部原注释改用可读标记保存在外层注释中。
    original = original.replace("<!--", "[原注释：").replace("-->", "]")
    if "--" in original:
        raise ValueError("待停用内容含 XML 注释不允许的连续连字符，请先检查源文件。")
    comment = ET.Comment(f" {reason}\n{original}\n")
    comment.tail = element.tail
    index = list(parent).index(element)
    parent[index] = comment


def prepare_h1_asset(source: Path = SOURCE, destination: Path = DESTINATION) -> Path:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"训练资产已存在，不覆盖：{destination}")
    if source.parent != destination.parent:
        raise ValueError("输出 XML 必须与源 XML 位于同一目录，以保留相对网格路径。")

    # 保留源文件里已经写好的注释，包括用户手动停用的配置。
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(source, parser=parser)
    root = tree.getroot()
    for element in list(root):
        if element.tag == "include" and element.get("file") == "scene.xml":
            _comment_out(root, element, "原场景引用停用；训练地面由 mjlab 创建。")
        elif element.tag in DISABLED_SECTIONS:
            _comment_out(root, element, DISABLED_SECTIONS[element.tag])

    world = root.find("worldbody")
    if world is None:
        raise ValueError("H1 XML 缺少 worldbody。")
    for body in list(world):
        if body.tag == "body" and body.get("name") == "not_use_body":
            _comment_out(world, body, "占位 body 停用；训练只使用 19 个实际关节。")

    for body in world.iter("body"):
        for i, geom in enumerate(body.findall("geom[@class='collision']")):
            # 已有名称予以保留；为原先未命名的碰撞体添加可定位名称。
            if geom.get("name") is None:
                suffix = "" if i == 0 else f"_{i}"
                geom.set("name", f"{body.get('name')}_collision{suffix}")

    # 内存检查使用绝对网格路径；实际输出仍保留 XML 的相对路径。
    validation_root = copy.deepcopy(root)
    compiler = validation_root.find("compiler")
    if compiler is None:
        raise ValueError("H1 XML 缺少 compiler。")
    meshdir = compiler.get("meshdir", ".")
    compiler.set("meshdir", str((source.parent / meshdir).resolve()))
    model = mujoco.MjModel.from_xml_string(ET.tostring(validation_root, encoding="unicode"))
    if (model.nq, model.nv, model.nu, model.nbody - 1) != (26, 25, 0, 20):
        raise ValueError(
            f"H1 训练资产尺寸不符：nq={model.nq}, nv={model.nv}, "
            f"nu={model.nu}, bodies={model.nbody - 1}"
        )
    for name in ("left_ankle_link_collision", "right_ankle_link_collision"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0:
            raise ValueError(f"缺少脚部碰撞体：{name}")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "not_use_joint") >= 0:
        raise ValueError("占位关节仍然生效。")

    ET.indent(tree, space="  ")
    # x 模式也阻止检查之后、写入之前出现的同名文件被覆盖。
    with destination.open("x", encoding="utf-8") as stream:
        tree.write(stream, encoding="unicode", xml_declaration=True)
        stream.write("\n")
    return destination


if __name__ == "__main__":
    result = prepare_h1_asset()
    print(f"H1 训练资产已生成：{result}")
    print("nq=26, nv=25, nu=0；位置执行器将在 mjlab 构建实体时添加。")
