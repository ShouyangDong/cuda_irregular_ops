import markdown
from bs4 import BeautifulSoup
import json


def parse_markdown(file_path):
    """解析 Markdown 文件并提取标题及其对应的段落内容"""
    # 读取 Markdown 文件
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # 将 Markdown 转换为 HTML
    html_content = markdown.markdown(md_content)

    # 使用 BeautifulSoup 解析 HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # 提取标题和段落
    titles_and_paragraphs = []
    current_title = None

    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
        if element.name.startswith("h"):  # 判断是否为标题
            # 如果当前标题不为空，保存之前的标题和段落
            if current_title is not None:
                titles_and_paragraphs.append((current_title, paragraphs))
            current_title = element.get_text()  # 更新当前标题
            paragraphs = []  # 重置段落列表
        elif element.name == "p" and current_title is not None:
            paragraphs.append(element.get_text())  # 添加段落到当前标题的段落列表

    # 添加最后一个标题和段落
    if current_title is not None:
        titles_and_paragraphs.append((current_title, paragraphs))

    return titles_and_paragraphs


if __name__ == "__main__":
    # 输入 Markdown 文件路径
    markdown_file = (
        "cntoolkit_3.5.2_cambricon_bang_c_4.5.1.md"  # 替换为您的 Markdown 文件路径
    )

    # 解析 Markdown 文件
    titles_and_paragraphs = parse_markdown(markdown_file)
    BANG_DOC = {}
    # 打印提取的标题和对应的段落
    for title, paragraphs in titles_and_paragraphs:
        if "bang" in title or "memcpy" in title:
            inst_name = title.split(" ")[1]
            print(f"标题: {paragraphs}")
            BANG_DOC[inst_name] = " ".join(paragraphs)

    with open("./bang_c_user_guide.json", "w", encoding="utf8") as json_file:
        json.dump(BANG_DOC, json_file, ensure_ascii=False, indent=2)
