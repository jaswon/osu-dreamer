:main
	@setlocal enableextensions enabledelayedexpansion
	@echo off

	for %%d in (%~dp0..) do set ParentDirectory=%%~fd

	
	python -m tensorboard.main --logdir="%ParentDirectory%\lightning_logs\"
	@echo on

	pause
	
	endlocal
	goto :eof