"""修补 mmaction2 1.2.0 pip 包的已知导入问题。

mmaction2 1.2.0 的 pip wheel 缺少 drn 模块目录，导致导入时报错。
此脚本自动修补该问题。

用法:
    uv run python patch_mmaction.py
"""

import importlib
import sys
from pathlib import Path


def patch_localizers_init():
    """修补 mmaction.models.localizers.__init__.py 中的 drn 导入。"""
    try:
        import mmaction
    except ImportError:
        print("[错误] mmaction2 未安装。")
        return False

    mmaction_path = Path(mmaction.__file__).parent
    init_file = mmaction_path / "models" / "localizers" / "__init__.py"

    if not init_file.exists():
        print(f"[跳过] 文件不存在: {init_file}")
        return True

    content = init_file.read_text(encoding="utf-8")

    if "try:" in content and "from .drn.drn import DRN" in content:
        print("[跳过] 已经修补过。")
        return True

    if "from .drn.drn import DRN" not in content:
        print("[跳过] 不需要修补（未找到 drn 导入）。")
        return True

    # 检查 drn 目录是否真的缺失
    drn_dir = mmaction_path / "models" / "localizers" / "drn"
    if drn_dir.exists():
        print("[跳过] drn 模块存在，不需要修补。")
        return True

    new_content = content.replace(
        "from .drn.drn import DRN\n"
        "from .tcanet import TCANet\n"
        "\n"
        "__all__ = ['TEM', 'PEM', 'BMN', 'TCANet', 'DRN']",
        "from .tcanet import TCANet\n"
        "\n"
        "try:\n"
        "    from .drn.drn import DRN\n"
        "    __all__ = ['TEM', 'PEM', 'BMN', 'TCANet', 'DRN']\n"
        "except (ImportError, ModuleNotFoundError):\n"
        "    __all__ = ['TEM', 'PEM', 'BMN', 'TCANet']",
    )

    init_file.write_text(new_content, encoding="utf-8")
    print(f"[已修补] {init_file}")
    return True


def main():
    print("修补 mmaction2 1.2.0 已知问题...")
    success = patch_localizers_init()
    if success:
        print("修补完成。")
    else:
        print("修补失败。")
        sys.exit(1)


if __name__ == "__main__":
    main()
