:main
    @echo off
	
	for %%d in (%~dp0..) do set ParentDirectory=%%~fd
	cd "%ParentDirectory%"

	set /p ModelCheckpointPath="Model Checkpoint Path: "
	set /p SongPath="Song Path: "
	set /p Artist="Artist: "
	set /p Title="Title: "
	set /p Samples="Number of samples to generate: "
	set /p Steps="Sample steps: "

	python -m model predict --num_samples %Samples% --sample_steps %Steps% --title "%Title%" --artist "%Artist%" "%ModelCheckpointPath%" "%SongPath%"
	
	pause

	goto :eof