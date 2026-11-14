"""The 8 texture sets: each pairs a GNM identity (semantic identity sampler class)
with a skin tone, eye colour, hair colour and age-related detail."""

SETS = [
    dict(id="01", name="aiko", identity="female_asian", skin_srgb=(226, 184, 160), undertone="warm",
         redness=0.9, freckles=0.0, beard=0.0, age=0.15, eye="dark_brown", hair_srgb=(28, 22, 20),
         brow_density=0.75, seed=11),
    dict(id="02", name="arjun", identity="male_middle_eastern", skin_srgb=(176, 122, 92), undertone="olive",
         redness=0.7, freckles=0.0, beard=0.9, age=0.35, eye="brown", hair_srgb=(24, 19, 17),
         brow_density=1.0, seed=22),
    dict(id="03", name="clara", identity="female_white", skin_srgb=(236, 200, 182), undertone="pink",
         redness=1.2, freckles=0.8, beard=0.0, age=0.25, eye="blue", hair_srgb=(120, 78, 45),
         brow_density=0.7, seed=33),
    dict(id="04", name="kofi", identity="male_black", skin_srgb=(110, 72, 52), undertone="warm",
         redness=0.35, freckles=0.0, beard=0.75, age=0.3, eye="dark_brown", hair_srgb=(18, 14, 12),
         brow_density=0.9, seed=44),
    dict(id="05", name="amara", identity="female_black", skin_srgb=(132, 88, 62), undertone="warm",
         redness=0.4, freckles=0.0, beard=0.0, age=0.2, eye="amber", hair_srgb=(20, 16, 14),
         brow_density=0.75, seed=55),
    dict(id="06", name="erik", identity="male_white", skin_srgb=(222, 178, 154), undertone="pink",
         redness=1.3, freckles=0.25, beard=0.6, age=0.55, eye="green", hair_srgb=(92, 66, 44),
         brow_density=0.95, seed=66),
    dict(id="07", name="layla", identity="female_middle_eastern", skin_srgb=(198, 146, 112), undertone="olive",
         redness=0.75, freckles=0.0, beard=0.0, age=0.3, eye="hazel", hair_srgb=(30, 22, 18),
         brow_density=0.85, seed=77),
    dict(id="08", name="kenji", identity="male_asian", skin_srgb=(214, 168, 134), undertone="neutral",
         redness=0.7, freckles=0.0, beard=0.35, age=0.6, eye="dark_brown", hair_srgb=(26, 22, 20),
         brow_density=0.9, seed=88),
]

# iris palettes (sRGB): stroma base, fibre highlight, pupillary zone (collarette inside), limbal ring
IRIS = {
    "dark_brown": dict(base=(58, 34, 20), fibre=(92, 58, 34), inner=(48, 28, 16), limbus=(22, 14, 10), crypt=0.6),
    "brown": dict(base=(86, 52, 28), fibre=(132, 88, 52), inner=(70, 40, 20), limbus=(30, 20, 14), crypt=0.7),
    "amber": dict(base=(150, 96, 40), fibre=(196, 140, 70), inner=(120, 70, 26), limbus=(52, 34, 18), crypt=0.6),
    "hazel": dict(base=(104, 98, 58), fibre=(150, 138, 86), inner=(122, 74, 32), limbus=(40, 36, 24), crypt=0.8),
    "green": dict(base=(84, 110, 74), fibre=(140, 160, 112), inner=(132, 104, 52), limbus=(34, 44, 34), crypt=0.9),
    "blue": dict(base=(78, 110, 146), fibre=(170, 192, 214), inner=(112, 116, 112), limbus=(30, 42, 62), crypt=1.0),
}
