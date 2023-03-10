:main
    @echo off
	
	for %%d in (%~dp0..) do set ParentDirectory=%%~fd
	
	SET /P INSTALLMATPLOTLIBTENSORFLOW="Install matplotlib & tensorboard for stats (Y/[N])?: "
	IF /I "%INSTALLMATPLOTLIBTENSORFLOW%" == "Y" call :InstallMatPlotLibTensorflow
	
	pip install "%ParentDirectory%"
	
	pause
	
	goto :eof

:InstallMatPlotLibTensorflow
	pip install tensorflow tensorboard matplotlib
	goto :eof