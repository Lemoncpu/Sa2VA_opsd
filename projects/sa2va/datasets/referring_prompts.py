DEFAULT_MASK_TO_REFERRING_QUESTION = (
    "<image>"
    "Describe the target marked by region1 with one short RefCOCO-style referring expression. "
    "Prefer a compact noun phrase with 2 to 6 words; only use 7 to 9 words if one extra local relation is necessary. "
    "Focus on the target itself and keep only the minimum visible details needed to localize it among nearby similar objects. "
    "Use directly visible category, color, clothing, parts, pose, ordinal, local spatial cue, or one short relation only when it helps disambiguation. "
    "Do not write a full sentence, explanation, or scene description. "
    "Do not start with templates like 'the target', 'the region', or 'region1'. "
    "Avoid discourse fillers and unnecessary articles unless they help localization. "
    "Do not mention segmentation tokens, tags, placeholder text, or unnecessary background."
)
