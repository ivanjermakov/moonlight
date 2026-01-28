# moonlight

WebGPU path tracer

## glTF limitations

Some scene features needed for full path tracer functionality are not present in glTF format.

| Feature | Solution |
| -- | -- |
| Camera focus distance | `focus_distance` custom property on camera |
| Camera fstop | `aperture_fstop` custom property on camera |
| Environment map | `map_env` custom property on camera with a path to .exr file |

## Further reading

- [Ray Tracing playlist by Sebastian Lague](https://youtube.com/playlist?list=PLFt_AvWsXl0dlgwe4JQ0oZuleqOTjmox3&si=aR0Y1UJI4DiwlLzq)
- [OCIO, Display Transforms and Misconceptions](https://chrisbrejon.com/articles/ocio-display-transforms-and-misconceptions/)
- [Physically Based Rendering in Filament](https://google.github.io/filament/Filament.md.html#materialsystem/improvingthebrdfs/energylossinspecularreflectance)
