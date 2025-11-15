import re
from pathlib import Path
from pypinyin import lazy_pinyin, Style


def make_collection_name_from_path(path: Path) -> str:
    """
    根据文件名生成集合名：
    - 中文：转为拼音全拼
    - 每个音节首字母大写，后面小写，然后拼接：DouLuoDaLu
    - 去掉非字母数字字符
    - 确保第一个字符是字母或下划线（Milvus 对名字有要求）
    """
    base = path.stem  # 去掉 .txt 后缀

    # 优先用 pypinyin 转拼音
    if lazy_pinyin is not None and Style is not None:
        try:
            p_list = lazy_pinyin(base, style=Style.NORMAL)
        except Exception:
            p_list = [base]
    else:
        # 没装 pypinyin 就直接把连续的字母数字当作 token
        p_list = re.findall(r"[A-Za-z0-9]+", base) or [base]

    parts: list[str] = []
    for token in p_list:
        # 清理掉非字母数字
        clean = re.sub(r"[^A-Za-z0-9]", "", token)
        if not clean:
            continue
        # 首字母大写，后面小写：douluo → Douluo
        parts.append(clean[0].upper() + clean[1:].lower())

    slug = "".join(parts) or "Novel"

    # Milvus 限制：必须以字母或下划线开头
    if not (slug[0].isalpha() or slug[0] == "_"):
        slug = "N" + slug

    # 这里直接返回 CamelCase 名字，例如 DouLuoDaLu
    return slug
