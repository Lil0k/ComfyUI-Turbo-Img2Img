# ComfyUI Turbo Img2Img Sigmas

A sigma scheduler node for img2img with few-step distilled flow models, such as Qwen-Image-2.1-Turbo.

With a plain `denoise` on a distilled schedule, whole steps drop off the top, so the result changes in big, uneven jumps. The reference also locks the layout from about sigma 0.95. This node maps `denoise` to a start sigma on a curve that puts most of the slider near 1.0. Below that point it follows the model's own trained sigmas.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Lil0k/ComfyUI-Turbo-Img2Img
```

Restart ComfyUI. The node is under `sampling/custom_sampling/sigmas` as **Turbo Img2Img Sigmas**.

## Wiring

- `VAEEncode(reference)` → `SamplerCustomAdvanced.latent_image`
- `Turbo Img2Img Sigmas.sigmas` → `SamplerCustomAdvanced.sigmas`

`SamplerCustomAdvanced` noises the latent up to `sigmas[0]` on its own, so no other nodes are needed.

## Inputs

| Input | Default | Meaning |
|---|---|---|
| `denoise` | 0.6 | 1.0 ignores the reference (text-to-image). Lower values stay closer to the reference. |
| `curve` | 8.0 | How `denoise` maps to the start sigma. 1 is linear. Raise it if 0.7–0.9 still copies too much; lower it if 0.3–0.5 changes too much. |
| `min_steps` | 3 | The fewest steps to sample. If too few trained sigmas fall below the start, the steps are spread evenly down to 0. |
| `schedule` | Qwen-Image-2.1-Turbo | The model's trained sigmas, from high to low, without the final 0. |

## Outputs

- `sigmas`: the schedule to pass to the sampler
- `start_sigma`: the start sigma that was picked, for display or debugging
