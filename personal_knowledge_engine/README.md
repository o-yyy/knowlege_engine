# 个人技术知识引擎 (Personal Tech Knowledge Engine)

> **版本**: V1.0  
> **创建日期**: 2025年7月13日  
> **设计理念**: 结构即系统 (Structure as System)

## 🎯 项目愿景

构建一个结构化的个人知识管理系统，对AI领域的关键技术（以论文和代码为载体）进行**深度、可追溯、可复用**的分析。这不仅是一个学习笔记仓库，更是一个训练**"算法-系统"双螺旋思维**、连接理论与实践、并能产出高层次技术洞察的个人知识引擎。

## 🏗️ 系统架构

本知识引擎采用**双层架构**：

### 📊 深度分析层 (Deep Dive Layer)
- **架构模式**: "卷宗式"架构 (Case File Architecture)
- **目标**: 对单点技术进行深度穿透分析
- **产物**: 结构化的技术"卷宗"

### 🔄 横向对比层 (Comparative Layer)  
- **架构模式**: 手动触发的"结构化对比分析"
- **目标**: 实现跨技术点的横向连接，生成高层次洞察
- **产物**: 标准化对比分析报告

## 📁 目录结构

```
/personal_knowledge_engine
├── README.md                     # 项目说明文档
├── /_TEMPLATE/                   # 分析模板源头 (只读)
│   ├── 1_SUMMARY.md
│   ├── 2_THEORY_DECONSTRUCTION.md
│   ├── 3_ENGINEERING_ANALYSIS.md
│   └── 4_CRITIQUE_AND_REFLECTION.md
├── /case_files/                  # 所有技术"卷宗"存放目录
│   └── /[paper_name]/            # 单个卷宗示例
│       ├── paper.pdf
│       ├── /src_code/            # 对应代码库
│       ├── 1_SUMMARY.md
│       ├── 2_THEORY_DECONSTRUCTION.md
│       ├── 3_ENGINEERING_ANALYSIS.md
│       └── 4_CRITIQUE_AND_REFLECTION.md
├── /comparisons/                 # 横向对比报告目录
│   └── COMPARISON_*.md
└── /scripts/                     # 自动化脚本目录
    └── create_case.py            # 卷宗创建脚本
```

## 🎯 核心原则

### 原则一：可验证性原则 (Principle of Verifiability)
- **描述**: 每一个断言都必须有明确的引用来源
- **执行标准**: 
  - 理论观点需注明论文章节
  - 代码逻辑需注明文件路径与行号
  - 实验数据需注明来源表格

### 原则二：静态穿透原则 (Principle of Static Penetration)
- **描述**: 在**不配置和运行代码**的前提下，通过静态分析洞察设计意图和性能特征
- **执行标准**:
  - 逻辑追踪核心变量的变换路径
  - 识别理论上的重操作并预判性能瓶颈
  - 分析项目依赖并识别特殊软硬件需求

### 原则三："展示，而非描述"原则 (Principle of "Show, Don't Tell")
- **描述**: 优先使用图、表、代码块、公式等可视化元素传递信息
- **执行标准**:
  - 结构用架构图
  - 算法用伪代码  
  - 结果用表格
  - 关键代码用代码块加注释

## 🚀 快速开始

### 1. 创建新的技术卷宗
```bash
# 使用自动化脚本创建新卷宗
python scripts/create_case.py swin_transformer_v1

# 或者创建其他论文的卷宗
python scripts/create_case.py attention_is_all_you_need
```

### 2. 填充卷宗内容
1. **放置论文**: 将PDF文件重命名为`paper.pdf`并放入卷宗目录
2. **克隆代码**: 将官方代码仓库克隆到`src_code/`目录
3. **分析填充**: 按照模板逐一填充四个分析文件

### 3. 生成对比报告
当积累足够多的卷宗后，可手动创建对比分析报告：
```bash
# 在comparisons/目录下创建对比文件
# 例如: COMPARISON_ViT_vs_Swin.md
```

## 📋 分析模板说明

### 1_SUMMARY.md - 概览与一句话结论
- 论文基本信息
- 核心贡献清单
- 个人评级 (Game-Changer/Significant/Incremental/Niche)
- 关键数据概览

### 2_THEORY_DECONSTRUCTION.md - 理论层拆解
- 动机与问题定义 (质询式分析)
- 方法论解剖 (模型结构、关键模块)
- 实验分析 (结果复现、消融实验)

### 3_ENGINEERING_ANALYSIS.md - 工程实现分析
- 代码依赖与环境分析
- 理论-代码映射表
- 推理流程静态分析
- 资源消耗理论分析

### 4_CRITIQUE_AND_REFLECTION.md - 批判性思考
- 局限性与待解决问题
- 后续工作追踪
- 潜在改进方向
- 个人启发与收获

## 🔧 工具与脚本

### create_case.py
自动化创建新卷宗的目录结构脚本
- **功能**: 复制模板、创建目录、生成占位文件
- **用法**: `python scripts/create_case.py <case_name>`
- **输出**: 完整的卷宗目录结构

## 💡 使用建议

### 与AI Agent协作
1. **深度分析阶段**: 指令AI Agent基于"质询式模板"和卷宗内资料进行分析
2. **横向对比阶段**: 指令AI Agent基于多个卷宗生成结构化对比报告
3. **持续迭代**: 根据新的理解不断完善和更新分析内容

### 质量控制
- 严格遵循三大核心原则
- 确保每个断言都有明确来源
- 优先使用可视化元素
- 保持分析的客观性和批判性

## 🎓 预期收益

通过系统性使用本知识引擎，您将获得：
- **深度技术理解**: 对AI技术的深层次洞察
- **架构师思维**: "算法-系统"双螺旋思维能力
- **可追溯知识库**: 高质量的个人技术资产
- **对比分析能力**: 跨技术点的横向连接能力

---

**开始您的技术知识引擎之旅吧！** 🚀
