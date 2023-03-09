:main
    @echo off
	
	for %%d in (%~dp0..) do set ParentDirectory=%%~fd
	cd "%ParentDirectory%"
	
	SET /P USE_TIMING_POINTS="Use timing points from a beatmap (Y/[N])?: "

	set /p ModelCheckpointPath="Model Checkpoint Path: "
	set /p SongPath="Song Path: "
	IF /I "%USE_TIMING_POINTS%" == "Y" set /p TimingPointsBeatmapPath="Beatmap path to take timing points from: "
	set /p Artist="Artist: "
	set /p Title="Title: "
	IF /I not "%USE_TIMING_POINTS%" == "Y" set /p BPM="BPM (Input 0 to skip): "
	set /p Samples="Number of samples to generate: "
	set /p Steps="Sample steps: "
	
	IF /I "%USE_TIMING_POINTS%" == "Y" (
		call :GenerateBeatmapFromTimingPoints
	) ELSE (
		call :GenerateBeatmap
	)
	
	pause

	goto :eof
	
:GenerateBeatmap
    IF /I "%BPM%" == "0" (
		python "%ParentDirectory%\scripts\pred.py" --num_samples %Samples% --sample_steps %Steps% --title "%Title%" --artist "%Artist%" "%ModelCheckpointPath%" "%SongPath%"
	) ELSE (
		python "%ParentDirectory%\scripts\pred.py" --num_samples %Samples% --sample_steps %Steps% --bpm %BPM% --title "%Title%" --artist "%Artist%" "%ModelCheckpointPath%" "%SongPath%"
	)
	
	goto :eof
	
:GenerateBeatmapFromTimingPoints
	python "%ParentDirectory%\scripts\pred.py" --timing_points_from "%TimingPointsBeatmapPath%" --num_samples %Samples% --sample_steps %Steps% --title "%Title%" --artist "%Artist%" "%ModelCheckpointPath%" "%SongPath%"
	goto :eof