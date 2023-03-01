:main
    @echo off
	
	for %%d in (%~dp0..) do set ParentDirectory=%%~fd
	cd "%ParentDirectory%"
	
	set /p SongsDir="Songs Directory: "

	python "%ParentDirectory%\scripts\cli.py" fit -c "%ParentDirectory%\config.yml" -c "%ParentDirectory%\osu_dreamer\model\model.yml" --data.src_path "%SongsDir%"

	pause

	goto :eof