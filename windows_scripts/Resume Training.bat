:main
    @echo off
	
	for %%d in (%~dp0..) do set ParentDirectory=%%~fd
	cd "%ParentDirectory%"
	
	set /p CheckpointPath="Checkpoint Path: "

	python -m model fit -c "%ParentDirectory%/model/config.yml" --ckpt_path "%CheckpointPath%"

	pause

	goto :eof