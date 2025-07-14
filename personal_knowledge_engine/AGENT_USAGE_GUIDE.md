# AI Agent 自动化工作流程指南

# 注意
一切分析必须完全基于现有代码以及网页端理论分析，不得参杂任何你的主观推理

## � 执行流程

### 步骤1: 创建分析环境
用户要求分析新论文时，立即执行：
```bash
python personal_knowledge_engine/scripts/quick_setup.py [年份_论文名称]
```

**命名规则**: 必须以发布年份开头，格式为 `YYYY_paper_name`
- 示例: `2021_swin_transformer`
- 示例: `2017_attention_is_all_you_need`
- 示例: `2024_llama3_vision`

### 步骤2: 等待材料准备
指导用户准备：
- 源代码 → 克隆到 `src_code/` 目录
- **注意**: 不需要PDF文件，理论分析将通过网页端完成

### 步骤3: 理论分析 (用户网页端操作)
用户使用网页端模型分析PDF，生成 `2_THEORY_DECONSTRUCTION.md`

### 步骤4: Agent自动分析执行
当用户完成理论分析并确认代码就位后，Agent基于以下材料执行：
- 已完成的 `2_THEORY_DECONSTRUCTION.md` (网页端生成)
- `src_code/` 目录中的源代码

## 📋 执行规则引用

### 身份设定
**完全遵循**: `vibe_coding/需求分析.md` 中的角色定义
- 资深AI研发架构师身份
- "算法-系统"双螺旋视角

### 核心原则
**严格执行**: `vibe_coding/需求分析.md` 第2节三大原则
- 可验证性原则
- 静态穿透原则
- "展示，而非描述"原则

### 分析任务分工
**Agent负责完成**以下文件，基于网页端理论分析和本地代码：

1. `1_SUMMARY.md` - 基于理论分析结果生成概览总结
2. `3_ENGINEERING_ANALYSIS.md` - 基于本地代码进行工程实现分析
3. `4_CRITIQUE_AND_REFLECTION.md` - 结合理论和工程进行批判性思考

**用户网页端完成**:
- `2_THEORY_DECONSTRUCTION.md` - 通过网页端模型分析PDF生成

## 🎯 执行要求

### 分析依据
- **理论分析**: 基于用户网页端生成的 `2_THEORY_DECONSTRUCTION.md`
- **代码分析**: 基于 `src_code/` 目录中的源代码
- **交叉验证**: 将理论创新点与代码实现进行精确映射

### 执行标准
- **不要重新解释规则** - 直接引用现有文档
- **充分利用理论分析** - 基于网页端分析结果进行工程映射
- **深度代码分析** - 体现静态穿透原则，无需运行代码
- **保持专业水准** - 体现架构师级别的分析深度

### 工作流程
1. **读取理论分析** - 仔细阅读用户提供的 `2_THEORY_DECONSTRUCTION.md`
2. **分析代码结构** - 深入分析 `src_code/` 中的实现细节
3. **理论-代码映射** - 将理论创新点与具体代码实现对应
4. **生成剩余文档** - 按模板要求完成其他分析文件

## � 关键文件位置

- 身份和原则: `vibe_coding/需求分析.md`
- 分析模板: `_TEMPLATE/*.md`
- 环境脚本: `scripts/quick_setup.py`
- 理论分析: `2_THEORY_DECONSTRUCTION.md` (用户网页端生成)
- 源代码: `src_code/` (用户准备)
