# moonlight

WebGPU path tracer

## glTF limitations

Some scene features needed for full path tracer functionality are not present in glTF format.

| Feature | Solution |
| -- | -- |
| Camera focus distance | `focus_distance` custom property on camera |
| Camera fstop | `aperture_fstop` custom property on camera |
| Environment map | `map_env` custom property on camera with a path to .exr file |
