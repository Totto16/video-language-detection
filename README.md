# TODO


VERSION 2: Breaking changes (how the data ist stored) ! 

## Customize

Folder, title etc.


## config.yaml example

```yaml
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
    ignore_files:
        - metadata
        - extrafanart
        - theme-music
    exception_on_error: false

scanner:
    scanner_type: config
    config:
        scan_amount: 100
        start_position: 0
```


## Current version 

```yaml
2.5.1
```
