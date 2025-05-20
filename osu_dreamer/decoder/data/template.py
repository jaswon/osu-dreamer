
map_template = \
f"""osu file format v14

[General]
AudioFilename: {{audio_filename}}
AudioLeadIn: 0
Mode: 0

[Metadata]
Title: {{title}}
TitleUnicode: {{title}}
Artist: {{artist}}
ArtistUnicode: {{artist}}
Creator: osu!dreamer
Version: {{version}}
Tags: osu_dreamer

[Difficulty]
HPDrainRate: {{hp}}
CircleSize: {{cs}}
OverallDifficulty: {{od}}
ApproachRate: {{ar}}
SliderMultiplier: 1
SliderTickRate: 1

[Events]
{{breaks}}

[TimingPoints]
{{timing_points}}

[HitObjects]
{{hit_objects}}
"""