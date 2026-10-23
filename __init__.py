"""Turbo Img2Img Sigmas — img2img with a smooth denoise for few-step distilled
flow models (Qwen-Image-2.1-Turbo and similar).

Plain denoise on a distilled schedule just drops whole steps from the top, so it
jumps in big, uneven chunks, and the reference already locks the layout at
sigma ~0.95. This node maps denoise to a start sigma on a curve that spends most
of the slider near 1.0, then continues on the model's own trained sigmas below
that point.

Wire: VAEEncode(reference) -> SamplerCustomAdvanced.latent_image,
      this node -> SamplerCustomAdvanced.sigmas.
SamplerCustomAdvanced noises the latent to sigmas[0] by itself
(sigma * noise + (1 - sigma) * latent), so nothing else is needed.
"""

import torch

QWEN21_TURBO = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568"


def start_sigma(denoise, curve):
    # same form as a flow "shift": curve=1 is linear, higher spends more of the
    # slider close to sigma 1.0, where the image changes most
    if denoise >= 1.0:
        return 1.0
    return curve * denoise / (1.0 + (curve - 1.0) * denoise)


class TurboImg2ImgSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "denoise": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "1.0 = ignore the reference latent (plain text-to-image). Lower = closer to the reference."}),
                "curve": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 30.0, "step": 0.5,
                                    "tooltip": "How denoise maps to the start sigma. 1 = linear (the usual cliff). Higher = more of the slider spent on big changes. If 0.7-0.9 still copies too much, raise it; if 0.3-0.5 changes too much, lower it."}),
                "min_steps": ("INT", {"default": 3, "min": 1, "max": 20,
                                      "tooltip": "Fewest sampling steps. Low denoise starts below most trained sigmas; extra steps are added evenly below the start."}),
                "schedule": ("STRING", {"default": QWEN21_TURBO,
                                        "tooltip": "The model's trained sigmas, high to low, without the final 0. Default: Qwen-Image-2.1-Turbo (model_index.json)."}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "FLOAT")
    RETURN_NAMES = ("sigmas", "start_sigma")
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/sigmas"

    def get_sigmas(self, denoise, curve, min_steps, schedule):
        nodes = sorted((float(x) for x in schedule.split(",") if x.strip()), reverse=True)
        s = start_sigma(denoise, curve)
        if denoise <= 0.0:
            return (torch.zeros(1), 0.0)
        # trained sigmas below the start; skip ones too close to it to be a real step
        rest = [x for x in nodes if x < s - 0.01]
        if len(rest) + 1 < min_steps:
            # not enough trained points left: spread the steps evenly from s to 0
            n = min_steps
            rest = [s * (n - i) / n for i in range(1, n)]
        sigmas = torch.tensor([s] + rest + [0.0], dtype=torch.float32)
        return (sigmas, float(s))


NODE_CLASS_MAPPINGS = {"TurboImg2ImgSigmas": TurboImg2ImgSigmas}
NODE_DISPLAY_NAME_MAPPINGS = {"TurboImg2ImgSigmas": "Turbo Img2Img Sigmas"}
