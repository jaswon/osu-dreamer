:main
	@setlocal enableextensions enabledelayedexpansion
    @echo off
	
	for %%d in (%~dp0..) do set ParentDirectory=%%~fd
	cd "%ParentDirectory%"
	
	set /p CheckpointPath="Checkpoint Path: "

	python "%ParentDirectory%/scripts/cli.py" fit -c "%ParentDirectory%/config.yml" -c "%ParentDirectory%/osu_dreamer/model/model.yml" --ckpt_path "%CheckpointPath%"

	pause

	endlocal
	goto :eof