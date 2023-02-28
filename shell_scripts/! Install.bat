:main
    @echo off
	
	for %%d in (%~dp0..) do set ParentDirectory=%%~fd
	
	SET /P INSTALLMATHPLOTLIB="Install mathplotlib & tensorboard for stats (Y/[N])?: "
	IF /I "%INSTALLMATHPLOTLIB%" == "Y" call :InstallMathPlotLib
	
	pip install "%ParentDirectory%"
	
	pause
	
	goto :eof

:InstallMathPlotLib
	pip install tensorflow tensorboard matplotlib
	goto :eof