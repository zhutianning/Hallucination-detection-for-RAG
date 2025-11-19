#!/bin/bash
# CPU 模式环境设置脚本
# 用于安装 Python 包（仅 API 模式，不需要 GPU 和本地模型）

# 设置错误时退出
set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   CPU 模式 RAG 项目环境设置           ║${NC}"
echo -e "${BLUE}║   (仅使用 API，无需 GPU)              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# 检查 Python 版本
echo -e "${YELLOW}检查 Python 版本...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 python3，请先安装 Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
echo -e "${GREEN}Python 版本: $(python3 --version)${NC}"

# 检查 Python 版本是否 >= 3.8
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "${RED}错误: 需要 Python 3.8 或更高版本${NC}"
    exit 1
fi

# 1. 创建虚拟环境（如果不存在）
if [ ! -d "venv_cpu" ]; then
    echo -e "${YELLOW}创建 Python 虚拟环境...${NC}"
    python3 -m venv venv_cpu
    echo -e "${GREEN}虚拟环境创建完成${NC}"
else
    echo -e "${GREEN}虚拟环境已存在，跳过创建${NC}"
fi

# 2. 激活虚拟环境
echo -e "${YELLOW}激活虚拟环境...${NC}"
source venv_cpu/bin/activate

# 3. 升级 pip
echo -e "${YELLOW}升级 pip...${NC}"
pip install --upgrade pip setuptools wheel --quiet

# 4. 安装项目依赖（仅 API 模式所需）
echo -e "${YELLOW}安装项目依赖（CPU + API 模式）...${NC}"
if [ -f "requirements_cpu_api.txt" ]; then
    pip install -r requirements_cpu_api.txt
else
    # 如果没有专用文件，安装最小依赖
    echo -e "${YELLOW}使用最小依赖安装...${NC}"
    pip install tqdm python-dotenv openai numpy
fi

# 5. 检查必要的文件
echo -e "${YELLOW}检查必要文件...${NC}"
missing_files=0

if [ ! -f "rag_from_page_chunks_original.py" ]; then
    echo -e "${RED}✗ 缺少: rag_from_page_chunks_original.py${NC}"
    missing_files=1
else
    echo -e "${GREEN}✓ rag_from_page_chunks_original.py${NC}"
fi

if [ ! -f "get_text_embedding.py" ]; then
    echo -e "${RED}✗ 缺少: get_text_embedding.py${NC}"
    missing_files=1
else
    echo -e "${GREEN}✓ get_text_embedding.py${NC}"
fi

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠ 警告: 未找到 .env 文件${NC}"
    echo -e "${YELLOW}  请确保已配置 API 密钥和 URL${NC}"
    if [ -f "uppmax_env_template.env" ]; then
        echo -e "${YELLOW}  可以从 uppmax_env_template.env 复制并配置${NC}"
    fi
else
    echo -e "${GREEN}✓ .env 文件存在${NC}"
fi

if [ $missing_files -eq 1 ]; then
    echo -e "${RED}缺少必要文件，请检查后重试${NC}"
    exit 1
fi

# 6. 验证安装
echo -e "${YELLOW}验证安装...${NC}"
python3 -c "import tqdm, dotenv, openai, numpy; print('✓ 所有依赖包安装成功')" 2>/dev/null || {
    echo -e "${RED}✗ 依赖包验证失败${NC}"
    exit 1
}

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   环境设置完成！                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}下一步：${NC}"
echo -e "1. 确保已配置 .env 文件（API 密钥和 URL）"
echo -e "2. 运行: ${GREEN}bash cpu_run.sh${NC} 或 ${GREEN}python3 rag_from_page_chunks_original.py${NC}"
echo ""