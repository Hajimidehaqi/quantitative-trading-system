@echo off
chcp 65001 >nul
echo ========================================
echo 葡萄酒品质分析系统 - 自动安装脚本
echo ========================================
echo.

echo [1/3] 检查 Python 环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未找到 Python，请先安装 Python 3.7+
    pause
    exit /b 1
)
python --version
echo Python 环境正常
echo.

echo [2/3] 检查 requirements.txt 文件...
if not exist "requirements.txt" (
    echo 错误：未找到 requirements.txt 文件
    echo 请确保 requirements.txt 与 install.bat 在同一目录
    pause
    exit /b 1
)
echo requirements.txt 文件存在
echo.

echo [3/3] 正在安装依赖包...
echo 这可能需要几分钟时间，请耐心等待...
echo.
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo 安装失败！可能的原因：
    echo 1. 网络连接问题
    echo 2. pip 需要更新（尝试：python -m pip install --upgrade pip）
    echo 3. 权限问题（尝试以管理员身份运行）
    pause
    exit /b 1
)
echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 正在启动程序...
echo.
python wine_analysis.py
pause


