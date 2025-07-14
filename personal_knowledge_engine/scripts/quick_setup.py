#!/usr/bin/env python3
"""
快速论文分析环境设置脚本
Quick Paper Analysis Setup Script

用途: 一键创建论文分析目录并提供后续指导
Usage: python scripts/quick_setup.py <paper_name>

作者: 资深AI研发架构师
版本: 1.0
日期: 2025年7月13日
"""

import os
import sys
import shutil
from pathlib import Path
import datetime


def quick_setup_paper(paper_name: str) -> None:
    """
    快速设置论文分析环境
    
    Args:
        paper_name (str): 论文名称 (例如: swin_transformer)
    """
    
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 定义路径
    template_path = project_root / "_TEMPLATE"
    case_files_root = project_root / "case_files"
    target_path = case_files_root / paper_name
    
    print("🚀 快速论文分析环境设置")
    print("=" * 40)
    print(f"📝 论文名称: {paper_name}")
    print(f"📍 目标路径: {target_path}")
    
    # 检查是否已存在
    if target_path.exists():
        print(f"⚠️  目录已存在，是否覆盖? (y/N): ", end="")
        response = input().strip().lower()
        if response != 'y':
            print("❌ 操作已取消")
            return
        shutil.rmtree(target_path)
    
    try:
        # 创建目录结构
        print("📁 创建目录结构...")
        case_files_root.mkdir(exist_ok=True)
        shutil.copytree(template_path, target_path)
        
        # 创建src_code目录
        src_code_dir = target_path / "src_code"
        src_code_dir.mkdir(exist_ok=True)
        
        # 创建快速指导文件
        guide_content = f"""# {paper_name} - 快速分析指南

## 📋 当前状态
- ✅ 目录结构已创建
- ⏳ 等待源代码
- ⏳ 等待网页端理论分析

## 🎯 下一步操作清单

### 1. 准备材料
- [ ] 将源代码克隆到 `src_code/` 目录
- [ ] 使用网页端模型分析PDF，生成 `2_THEORY_DECONSTRUCTION.md`
- [ ] 确认代码和理论分析都已就位

### 2. 工作流程
**第一步**: 网页端理论分析
- 使用网页端模型分析论文PDF
- 生成 `2_THEORY_DECONSTRUCTION.md` 文件

**第二步**: AI Agent代码分析
当理论分析和代码都准备完成后，使用以下指令：

```
请按照个人技术知识引擎的分析规则，基于已有的理论分析和源代码，完成剩余的分析文档。

材料位置:
- 理论分析: {target_path}/2_THEORY_DECONSTRUCTION.md (已完成)
- 源代码: {target_path}/src_code/

请严格按照以下顺序执行:
1. 仔细阅读已完成的理论分析
2. 深入分析源代码结构和实现
3. 将理论创新点与代码实现进行精确映射
4. 完成剩余分析文件: 1_SUMMARY.md, 3_ENGINEERING_ANALYSIS.md, 4_CRITIQUE_AND_REFLECTION.md

开始分析: 1_SUMMARY.md
```

### 3. 分析文件清单
- [ ] `1_SUMMARY.md` - 概览与一句话结论 (Agent完成)
- [x] `2_THEORY_DECONSTRUCTION.md` - 理论层拆解 (网页端完成)
- [ ] `3_ENGINEERING_ANALYSIS.md` - 工程实现分析 (Agent完成)
- [ ] `4_CRITIQUE_AND_REFLECTION.md` - 批判性思考 (Agent完成)

## 📝 创建信息
- 创建时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 创建脚本: quick_setup.py
"""
        
        guide_file = target_path / "QUICK_GUIDE.md"
        guide_file.write_text(guide_content, encoding='utf-8')
        
        # 创建状态跟踪文件
        status_content = f"""# 分析状态跟踪

## 基本信息
- 论文名称: {paper_name}
- 创建时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 状态: 初始化完成

## 材料准备状态
- [ ] 源代码已克隆
- [ ] 网页端理论分析已完成
- [ ] 开始AI Agent分析

## 分析进度
- [ ] 1_SUMMARY.md (0%) - Agent完成
- [ ] 2_THEORY_DECONSTRUCTION.md (0%) - 网页端完成
- [ ] 3_ENGINEERING_ANALYSIS.md (0%) - Agent完成
- [ ] 4_CRITIQUE_AND_REFLECTION.md (0%) - Agent完成

## 质量检查
- [ ] 所有引用都有明确来源
- [ ] 代码分析包含具体文件路径和行号
- [ ] 遵循"展示而非描述"原则
- [ ] 完成批判性思考

## 备注
[在此记录分析过程中的重要发现和问题]
"""
        
        status_file = target_path / "STATUS.md"
        status_file.write_text(status_content, encoding='utf-8')
        
        print("✅ 环境设置完成!")
        print(f"📂 已创建目录: {target_path}")
        print(f"📋 快速指南: {guide_file}")
        print(f"📊 状态跟踪: {status_file}")
        
        print("\n🎯 接下来请:")
        print("1. 将源代码克隆到 src_code/ 目录")
        print("2. 使用网页端模型分析PDF，生成 2_THEORY_DECONSTRUCTION.md")
        print("3. 查看 QUICK_GUIDE.md 获取AI Agent分析指令")
        
    except Exception as e:
        print(f"❌ 设置失败: {e}")
        if target_path.exists():
            shutil.rmtree(target_path)
        sys.exit(1)


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("❌ 用法: python scripts/quick_setup.py <YYYY_paper_name>")
        print("📝 示例: python scripts/quick_setup.py 2021_swin_transformer")
        print("📝 示例: python scripts/quick_setup.py 2017_attention_is_all_you_need")
        print("📝 示例: python scripts/quick_setup.py 2024_llama3_vision")
        print("\n📋 命名规则:")
        print("   - 必须以4位年份开头")
        print("   - 年份后跟下划线")
        print("   - 论文名称使用下划线分隔")
        sys.exit(1)

    paper_name = sys.argv[1].strip()
    if not paper_name:
        print("❌ 论文名称不能为空")
        sys.exit(1)

    # 验证命名格式：必须以年份开头
    if not paper_name.startswith(('19', '20')) or len(paper_name) < 5 or paper_name[4] != '_':
        print("❌ 命名格式错误!")
        print("📋 正确格式: YYYY_paper_name")
        print("📝 示例: 2021_swin_transformer")
        print("📝 示例: 2017_attention_is_all_you_need")
        sys.exit(1)

    # 验证年份部分是数字
    year_part = paper_name[:4]
    if not year_part.isdigit():
        print("❌ 年份必须是4位数字")
        sys.exit(1)

    # 验证整体格式
    if not paper_name.replace('_', '').replace('-', '').isalnum():
        print("❌ 论文名称只能包含字母、数字、下划线和连字符")
        sys.exit(1)

    quick_setup_paper(paper_name)


if __name__ == "__main__":
    main()
