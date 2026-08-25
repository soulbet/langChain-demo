import fitz  # PyMuPDF
import os
import re


def extract_chapters_by_bookmarks(pdf_path, output_dir):
    """
    利用PDF书签（目录）提取章节内容，保存为单独的txt文件。
    """
    # 1. 打开PDF
    doc = fitz.open(pdf_path)

    # 2. 获取书签（目录）列表
    # toc 是一个列表，每个元素为 [层级, 标题, 页码]
    toc = doc.get_toc()

    if not toc:
        print("此PDF未包含书签信息。")
        doc.close()
        return

    print(f"共发现 {len(toc)} 个书签条目。")

    # 3. 过滤出你需要的章节（根据你的截图，大致是第1级书签）
    # 这里我们假设所有章节都是顶级书签（层级为1）
    # 你的书签列表中，译者序、前言、第1章等都在同一层级
    chapters = []
    for item in toc:
        level, title, page_num = item
        if level == 1:  # 只取顶级书签，对应你的截图内容
            # 将页码从0基转换为0基（PyMuPDF页码从0开始，书签页码从1开始）
            # 注意：书签记录的页码是物理页码，转换为PyMuPDF页码需要减1
            pymupdf_page = page_num - 1
            chapters.append({
                "title": title,
                "page": pymupdf_page
            })

    # 4. 为每个章节提取内容
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, chapter in enumerate(chapters):
        start_page = chapter["page"]
        # 确定结束页码：如果这是最后一章，则到文档末尾；否则到下一章开始页
        if idx + 1 < len(chapters):
            end_page = chapters[idx + 1]["page"] - 1
        else:
            end_page = len(doc) - 1

        # 提取内容
        content = ""
        for page_num in range(start_page, end_page + 1):
            page = doc[page_num]
            content += page.get_text()

        # 清理文件名和内容
        safe_title = re.sub(r'[^\w\s]', '', chapter["title"]).strip().replace(' ', '_')
        # 限制文件名长度
        if len(safe_title) > 50:
            safe_title = safe_title[:50]
        filename = f"{idx + 1:02d}_{safe_title}.txt"
        filepath = os.path.join(output_dir, filename)

        # 保存文件，在开头添加章节标题
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{chapter['title']}\n\n{content}")
        print(f"已保存: {filepath}")

    doc.close()
    print(f"\n✅ 共提取 {len(chapters)} 个章节。")


# 使用示例
if __name__ == "__main__":
    pdf_file = "D:\work_space\python\python\practice\处理文件\data\input\AI工程大模型应用开发实战 (【越】奇普·萱) (z-library.sk, 1lib.sk, z-lib.sk).pdf"  # 替换为你的PDF文件路径
    output_folder = r"D:\work_space\python\python\practice\处理文件\data\output"
    extract_chapters_by_bookmarks(pdf_file, output_folder)