:main
	@setlocal enableextensions enabledelayedexpansion
    @echo off
	
	for %%d in (%~dp0..) do set ParentDirectory=%%~fd
	cd %ParentDirectory%
	
	SET /P USE_TIMING_POINTS="Use timing points from a beatmap (Y/[N])?: "

	set /p ModelCheckpointPath="Model Checkpoint Path: "
	set /p SongPath="Song Path: "
	IF /I "%USE_TIMING_POINTS%" == "Y" set /p TimingPointsBeatmapPath="Beatmap to take timing points from: "
	set /p Artist="Artist: "
	set /p Title="Title: "
	set /p BPM="BPM: "
	set /p Samples="Number of samples to generate: "
	set /p Steps="Sample steps: "
	
	IF /I "%USE_TIMING_POINTS%" == "N" call :GenerateBeatmap
	IF /I "%USE_TIMING_POINTS%" == "Y" call :GenerateBeatmapFromTimingPoints
	
	pause

	endlocal
	goto :eof
	
:GenerateBeatmap
	setlocal
	python %ParentDirectory%\scripts\pred.py --num_samples %Samples% --sample_steps %Steps% --bpm %BPM% --title "%Title%" --artist "%Artist%" "%ModelCheckpointPath%" "%SongPath%"
	endlocal
	goto :eof
	
:GenerateBeatmapFromTimingPoints
	setlocal
	python %ParentDirectory%\scripts\pred.py --timing_points_from %TimingPointsBeatmapPath% --num_samples %Samples% --sample_steps %Steps% --bpm %BPM% --title "%Title%" --artist "%Artist%" "%ModelCheckpointPath%" "%SongPath%"
	endlocal
	goto :eof