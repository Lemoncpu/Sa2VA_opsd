DEFAULT_MASK_TO_CAPTION_QUESTION = (
    "<image>"
    "Describe the target marked by region1 with one detailed, localized caption. "
    "Write it in the style of a precise visual object description: first identify the target category, then describe "
    "its visible appearance in detail, such as color, shape, size, material, texture, parts, markings, clothing, pose, "
    "facial or body attributes, and other distinctive traits that can be directly seen. "
    "Include only the minimum nearby spatial or relational cue if it is necessary to distinguish this target from similar nearby objects. "
    "Keep the caption focused on the target itself rather than the whole scene. Use one natural, complete sentence. "
    "Do not mention unnecessary background, unrelated objects or people, inferred activities, emotions, or scene-level interpretation. "
    "Describe only what is directly visible and useful for localizing the target. "
    "Do not output segmentation tokens, tags, or placeholder text."
)
