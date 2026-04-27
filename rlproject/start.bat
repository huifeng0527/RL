@echo off
chcp 65001 >nul
echo ========================================
echo    康复评估系统启动器
echo ========================================
echo.

cd /d "%~dp0"

:: 检查 conda 环境
where conda >nul 2>&1
if %errorlevel%==0 (
    echo [激活 conda 环境: rl]
    call conda activate rl
)

:: 启动系统
echo [启动系统中...]
python start_system.py %*

:: 如果直接关闭了，等待用户查看
if %errorlevel% neq 0 (
    echo.
    echo 系统异常退出，按任意键退出...
    pause >nul
)
