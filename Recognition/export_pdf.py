#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将Markdown文档导出为PDF的工具
"""

import os
import sys
import subprocess
from pathlib import Path

def install_required_packages():
    """安装所需的Python包"""
    packages = [
        'markdown',
        'weasyprint',
        'pygments',
        'pymdown-extensions'
    ]
    
    print("正在安装必要的Python包...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} 安装成功")
        except subprocess.CalledProcessError:
            print(f"✗ {package} 安装失败")
            return False
    return True

def markdown_to_html_with_syntax_highlight(md_content):
    """将Markdown转换为带有语法高亮的HTML"""
    try:
        import markdown
        from pymdown import superfences
    except ImportError:
        print("缺少必要的包，请先运行安装命令")
        return None
    
    # 配置Markdown扩展
    md_extensions = [
        'extra',  # 支持表格、代码块等
        'codehilite',  # 代码语法高亮
        'toc',  # 目录
        'meta',  # 元数据
        'pymdownx.superfences',  # 更好的代码块支持
        'pymdownx.highlight',  # 语法高亮增强
        'pymdownx.arithmatex',  # 数学公式支持
    ]
    
    # 扩展配置
    extension_configs = {
        'codehilite': {
            'css_class': 'highlight',
            'use_pygments': True,
            'noclasses': False,
        },
        'pymdownx.highlight': {
            'css_class': 'highlight',
            'use_pygments': True,
            'noclasses': False,
        },
        'toc': {
            'permalink': True,
        }
    }
    
    # 创建Markdown实例
    md = markdown.Markdown(
        extensions=md_extensions,
        extension_configs=extension_configs
    )
    
    # 转换为HTML
    html_content = md.convert(md_content)
    
    # 添加CSS样式
    css_styles = """
    <style>
        body {
            font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            page-break-before: always;
        }
        
        h2 {
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
            margin-top: 30px;
        }
        
        h3 {
            color: #7f8c8d;
            margin-top: 25px;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 14px;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #495057;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 14px;
            color: #e83e8c;
        }
        
        pre {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
            font-size: 13px;
            line-height: 1.4;
        }
        
        .highlight {
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        
        .toc {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }
        
        .toc li {
            margin: 5px 0;
        }
        
        .toc a {
            color: #3498db;
            text-decoration: none;
        }
        
        .toc a:hover {
            text-decoration: underline;
        }
        
        @page {
            margin: 2cm;
            size: A4;
        }
        
        @media print {
            body {
                font-size: 12pt;
                line-height: 1.5;
            }
            
            h1 {
                page-break-before: always;
            }
            
            h1, h2, h3 {
                page-break-after: avoid;
            }
            
            table, pre {
                page-break-inside: avoid;
            }
        }
    </style>
    """
    
    # 完整的HTML文档
    full_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>深度学习大作业文档</title>
        {css_styles}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    return full_html

def html_to_pdf(html_content, output_path):
    """将HTML转换为PDF"""
    try:
        import weasyprint
    except ImportError:
        print("缺少weasyprint包，请先运行安装命令")
        return False
    
    try:
        # 创建PDF文档
        document = weasyprint.HTML(string=html_content)
        
        # 生成PDF
        document.write_pdf(output_path)
        
        print(f"✓ PDF文件已成功生成: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ PDF生成失败: {str(e)}")
        return False

def convert_markdown_to_pdf(markdown_file, pdf_file):
    """主转换函数"""
    print(f"正在转换: {markdown_file} -> {pdf_file}")
    
    # 读取Markdown文件
    try:
        with open(markdown_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        print(f"✓ 成功读取Markdown文件: {len(md_content)} 字符")
    except Exception as e:
        print(f"✗ 读取Markdown文件失败: {str(e)}")
        return False
    
    # 转换为HTML
    print("正在转换为HTML...")
    html_content = markdown_to_html_with_syntax_highlight(md_content)
    if html_content is None:
        return False
    
    print(f"✓ HTML转换完成: {len(html_content)} 字符")
    
    # 转换为PDF
    print("正在生成PDF...")
    return html_to_pdf(html_content, pdf_file)

def main():
    """主函数"""
    # 设置文件路径
    project_dir = Path("d:/pythons/deeplearn/2315925647_王文玉_深度学习大作业文档")
    markdown_file = project_dir / "大作业文档.md"
    pdf_file = project_dir / "大作业文档.pdf"
    
    # 检查文件是否存在
    if not markdown_file.exists():
        print(f"错误: Markdown文件不存在: {markdown_file}")
        return False
    
    print("=" * 50)
    print("深度学习大作业文档 PDF 导出工具")
    print("=" * 50)
    
    # 安装必要的包
    if not install_required_packages():
        print("✗ 包安装失败，请手动安装后重试")
        return False
    
    # 执行转换
    success = convert_markdown_to_pdf(markdown_file, pdf_file)
    
    if success:
        print("\n" + "=" * 50)
        print("✓ 转换成功完成!")
        print(f"PDF文件位置: {pdf_file}")
        print(f"文件大小: {pdf_file.stat().st_size / 1024 / 1024:.2f} MB")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("✗ 转换失败，请检查错误信息")
        print("=" * 50)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)