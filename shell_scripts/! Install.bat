:main
    @setlocal enableextensions enabledelayedexpansion
    @echo off
	
	for %%d in (%~dp0..) do set ParentDirectory=%%~fd
	
	SET /P INSTALLMATHPLOTLIB="Install mathplotlib & tensorboard for stats (Y/[N])?: "
	IF /I "%INSTALLMATHPLOTLIB%" == "Y" call :InstallMathPlotLib
	
	pip install %ParentDirectory%
	
	pause
	
	endlocal
	goto :eof

:InstallMathPlotLib
	setlocal
	pip install tensorflow
	pip install tensorboard
	pip install matplotlib
	endlocal
	goto :eof