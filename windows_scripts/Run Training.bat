:main
    @echo off
	
	for %%d in (%~dp0..) do set ParentDirectory=%%~fd
	cd "%ParentDirectory%"
	
	set /p SongsDir="Songs Directory: "

	python -m model fit -c "%ParentDirectory%\model\config.yml" --data.src_path "%SongsDir%"

	pause

	goto :eof