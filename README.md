# TODO


VERSION 2: Breaking changes (how the data ist stored) ! 

## Customize

Folder, title etc.


## config.yaml example

```yaml
templates:
    default:
        general:
            target_file: data/data.json

        parser:
            root_folder: /my/media/root/folder/
            special:
                - Extras
                - Specials
                - Special
            video_formats:
                - mp4
                - mkv
                - avi
            trailer_names:
                - tvshow-trailer
            ignore_files:
                - metadata
                - extrafanart
                - theme-music
                - Music
                - Reportage
            exception_on_error: false

        keybindings:
            abort: ControlG

        picker:
            picker_type: interactive
            config:
                entries_to_show: 12
                show_full_list: False
                play_sound: true

        metadata:
            type: tmdb
            config:
                api_key: <your_api_key>

    names:
        fast:
            scanner:
                scanner_type: config
                config:
                    scan_amount:
                        language: 50
                        metadata: 100000
                    start_position:
                        language: 0
                        metadata: 0
                    types: both

            classifier:
                batch_settings:
                    batch_type: "auto"
                    keep_free: "10%"
                accuracy:
                    normal_threshold: "95%"
                    final_threshold: "55%"
                    use_picker_at_end: False
                scan_config:
                    minimum: "20%"
                    maximum: "50%"
        manual:
            scanner:
                scanner_type: config
                config:
                    scan_amount:
                        language: 10
                        metadata: 100000
                    start_position:
                        language: 0
                        metadata: 0
                    types: both

            classifier:
                batch_settings:
                    batch_type: "auto"
                    keep_free: "10%"
                accuracy:
                    normal_threshold: "95%"
                    final_threshold: "55%"
                    use_picker_at_end: true
                scan_config:
                    minimum: "20%"

    aliases:
        man: manual

    use: fast
    settings:
        prefer_cli_template: true
```


## Current version 

```yaml
2.5.3
```
