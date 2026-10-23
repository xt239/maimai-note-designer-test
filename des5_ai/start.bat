@echo off
chcp 65001 >nul
setlocal
title maimai 谱面谱师识别（inote -^> des）
cd /d "%~dp0"

rem ================= 可调参数 =================
if not defined DES5_PORT set "DES5_PORT=8765"
set "NO_BROWSER=0"
if /i "%~1"=="--no-browser" set "NO_BROWSER=1"
rem ===========================================

set "PORT=%DES5_PORT%"
set "URL=http://127.0.0.1:%PORT%/"

echo.
echo   ============================================================
echo      maimai 谱面谱师识别   inote_N 区间  -^>  des_N 区间
echo   ============================================================
echo.

rem ---------- 1. 定位 Python ----------
set "PYEXE="
set "PYFLAGS="
if defined PYTHON if exist "%PYTHON%" set "PYEXE=%PYTHON%"
if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PYEXE for %%P in (python.exe) do if not defined PYEXE if not "%%~$PATH:P"=="" set "PYEXE=%%~$PATH:P"
if not defined PYEXE for %%P in (py.exe) do if not defined PYEXE if not "%%~$PATH:P"=="" ( set "PYEXE=%%~$PATH:P" & set "PYFLAGS=-3" )
if not defined PYEXE (
    echo   [1/4] [错误] 没有找到 Python。
    echo.
    echo     请先安装 Python 3.11 或更高版本 ^(https://www.python.org/downloads/^)，
    echo     安装时勾选 "Add python.exe to PATH"；
    echo     或安装后在本目录执行：  set PYTHON=C:\...\python.exe
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%V in ('"%PYEXE%" -V 2^>^&1') do set "PYVER=%%V"

rem ---------- 2. numpy 检测 / 自动安装 ----------
rem 代码用到 np.random.default_rng，需要 numpy >= 1.17
rem 注意：让 python 只输出 ASCII 版本号到临时文件，中文提示交给 echo。
rem       python 直接 print 中文在 chcp 65001 下会因 stdout 编码而报错（errorlevel 非 0）。
set "NPVERFILE=%TEMP%\des5_numpy_ver.txt"
set "NPJUST=0"
"%PYEXE%" %PYFLAGS% -c "import numpy;print(numpy.__version__)" > "%NPVERFILE%" 2>nul
if not errorlevel 1 goto :HAVE_NUMPY

:INSTALL_NUMPY
echo   [2/4] 未检测到 numpy，正在自动下载安装（需联网，约 10~90 秒）……
echo.
set "NPJUST=1"
rem 先确保 pip 本身可用（精简安装可能没带）
"%PYEXE%" %PYFLAGS% -m ensurepip --upgrade >nul 2>&1
"%PYEXE%" %PYFLAGS% -m pip install --disable-pip-version-check --no-input "numpy>=1.17"
if not errorlevel 1 goto :NUMPY_RETRY
echo.
echo   常规安装未成功 ^(可能是权限不足^)，改用 --user 重试……
"%PYEXE%" %PYFLAGS% -m pip install --disable-pip-version-check --no-input --user "numpy>=1.17"

:NUMPY_RETRY
"%PYEXE%" %PYFLAGS% -c "import numpy;print(numpy.__version__)" > "%NPVERFILE%" 2>nul
if errorlevel 1 goto :NUMPY_FAIL
goto :HAVE_NUMPY

:HAVE_NUMPY
set "NPVER=未知"
if exist "%NPVERFILE%" set /p NPVER=<"%NPVERFILE%"
del "%NPVERFILE%" >nul 2>&1
if "%NPJUST%"=="1" goto :NUMPY_JUST_INSTALLED
echo   [2/4] numpy %NPVER%  ^(已安装^)
goto :NUMPY_DONE
:NUMPY_JUST_INSTALLED
echo   [2/4] numpy %NPVER%  ^(刚刚自动安装完成^)
goto :NUMPY_DONE

:NUMPY_FAIL
echo.
echo   [错误] numpy 自动安装失败。
echo          可能原因：网络不通 / 需要代理 / 磁盘空间不足 / pip 被禁用
echo.
echo          可手动执行以下命令后重试：
echo            "%PYEXE%" %PYFLAGS% -m pip install "numpy>=1.17"
echo.
echo          国内网络可换清华镜像：
echo            "%PYEXE%" %PYFLAGS% -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "numpy>=1.17"
echo.
pause
exit /b 1

:NUMPY_DONE
echo.

rem ---------- 3. 模型文件检查 ----------
if not exist "model_bag.npz" (
    echo   [3/4] [警告] 缺少 model_bag.npz，预测功能不可用。
    echo          请先运行：  "%PYEXE%" train.py --epochs 25 --out model_bag.npz
    echo.
) else (
    echo   [3/4] 模型文件就绪。
)

rem ---------- 4. 端口检查 ----------
echo   [4/4] 检查端口 %PORT%……
call :IsBusy %PORT%
if not "%BUSY%"=="1" goto :LAUNCH

rem 端口被占：若正是本服务，直接打开浏览器
curl -s -m 2 "%URL%api/health" 2>nul | findstr /c:"arch" >nul 2>nul
if not errorlevel 1 (
    echo   [提示] 服务已在运行 ^(端口 %PORT%^)，直接打开浏览器。
    echo.
    if "%NO_BROWSER%"=="0" start "" "%URL%"
    timeout /t 3 >nul
    exit /b 0
)

rem 端口被别的程序占用：自动往后找一个空闲端口
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "$s=%DES5_PORT%; for($i=0;$i -lt 30;$i++){ $c=$s+$i; if(-not (Get-NetTCPConnection -LocalPort $c -State Listen -ErrorAction SilentlyContinue)){ $c; break } }"`) do set "PORT=%%P"
if not defined PORT set "PORT=%DES5_PORT%"
set "URL=http://127.0.0.1:%PORT%/"
echo   [提示] 原端口被占用，已改用端口 %PORT%。
echo.

:LAUNCH
echo   Python : %PYEXE% %PYFLAGS%  (%PYVER%)
echo   地址   : %URL%
echo   停止   : 关闭本窗口，或按 Ctrl+C
echo.
echo   正在启动，首次加载模型约需 3~8 秒……
echo.

if "%NO_BROWSER%"=="0" (
    start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process '%URL%'"
)

"%PYEXE%" %PYFLAGS% server.py --port %PORT%

echo.
echo   服务已停止。
timeout /t 2 >nul
exit /b 0

rem ---------- 子程序：判断端口是否被监听 ----------
:IsBusy
set "BUSY=0"
netstat -ano -p tcp 2>nul | findstr /r /c:":%1 .*LISTENING" >nul 2>nul && set "BUSY=1"
exit /b 0
