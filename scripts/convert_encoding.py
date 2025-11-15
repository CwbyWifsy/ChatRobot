import chardet
from pathlib import Path


from pathlib import Path


def convert_to_utf8(file_path: Path):
    print(f"➡ 进入文件：{file_path.name}")
    raw = file_path.read_bytes()

    # Step 1: Try UTF-8 (quick and safe)
    try:
        raw.decode("utf-8")
        print("   └─ 已是 UTF-8 文件，跳过\n")
        return
    except:
        pass

    # Step 2: Backup before conversion
    bak = file_path.with_suffix(file_path.suffix + ".bak")
    bak.write_bytes(raw)
    print(f"   ├─ 已创建备份：{bak.name}")

    # Step 3: Try gb18030 safely
    try:
        text = raw.decode("gb18030")  # 不用 ignore，避免吞字
        file_path.write_text(text, encoding="utf-8")
        print(f"   ├─ 成功写回 UTF-8：{file_path.name}")

        bak.unlink()
        print(f"   └─ 已删除备份\n")

    except Exception as e:
        print(f"   ❌ 转换失败：{e}, 正在恢复文件")
        file_path.write_bytes(raw)
        print(f"   └─ 已恢复原文件\n")


def convert_directory(directory: str):
    folder = Path(directory)
    if not folder.exists():
        print(f"❌ Directory not found: {directory}")
        return

    all_txt = list(folder.rglob("*.txt"))
    print(f"📦 Found {len(all_txt)} text files in {directory}\n")

    for f in all_txt:
        convert_to_utf8(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert novels to UTF-8 safely")
    parser.add_argument("directory", type=str, help="Directory containing .txt files")
    args = parser.parse_args()

    convert_directory(args.directory)
