#!/bin/bash
# CPU 模式运行脚本
# 运行 RAG 处理脚本（仅使用 API，无需 GPU）

# 设置错误时退出
set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   CPU 模式 RAG 处理任务               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# 1. 检查虚拟环境
if [ ! -d "venv_cpu" ]; then
    echo -e "${RED}错误: 虚拟环境不存在${NC}"
    echo -e "${YELLOW}请先运行: bash cpu_setup.sh${NC}"
    exit 1
fi

# 2. 激活虚拟环境
echo -e "${YELLOW}激活虚拟环境...${NC}"
source venv_cpu/bin/activate

# 3. 检查必要文件
echo -e "${YELLOW}检查必要文件...${NC}"
missing_files=0

if [ ! -f "rag_from_page_chunks_original.py" ]; then
    echo -e "${RED}✗ 缺少: rag_from_page_chunks_original.py${NC}"
    missing_files=1
else
    echo -e "${GREEN}✓ rag_from_page_chunks_original.py${NC}"
fi

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠ 警告: 未找到 .env 文件${NC}"
    echo -e "${YELLOW}  请确保已配置 API 密钥和 URL${NC}"
else
    echo -e "${GREEN}✓ .env 文件存在${NC}"
fi

if [ ! -f "all_pdf_page_chunks_merged.json" ]; then
    echo -e "${YELLOW}⚠ 警告: 未找到 all_pdf_page_chunks_merged.json${NC}"
    echo -e "${YELLOW}  请确保数据文件已准备好${NC}"
else
    echo -e "${GREEN}✓ all_pdf_page_chunks_merged.json${NC}"
fi

if [ $missing_files -eq 1 ]; then
    echo -e "${RED}缺少必要文件，请检查后重试${NC}"
    exit 1
fi

# 4. 显示系统信息
echo ""
echo -e "${GREEN}系统信息:${NC}"
echo "  Python: $(python3 --version)"
echo "  工作目录: $(pwd)"
echo "  开始时间: $(date)"
echo ""

# 5. 运行主脚本
echo -e "${GREEN}开始运行 RAG 处理脚本...${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

python3 rag_from_page_chunks_original.py

echo ""
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${GREEN}任务完成！${NC}"
echo "  结束时间: $(date)"
echo ""

# 6. 检查输出文件
if [ -f "rag_top1_pred.json" ]; then
    echo -e "${GREEN}✓ 输出文件已生成: rag_top1_pred.json${NC}"
fi
if [ -f "rag_top1_pred_raw.json" ]; then
    echo -e "${GREEN}✓ 原始结果已生成: rag_top1_pred_raw.json${NC}"
fi