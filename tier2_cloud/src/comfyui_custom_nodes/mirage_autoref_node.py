"""
MIRAGE_AutoRefPrompt - v5  (silhouette-driven, diversity engine)
==================================================================
This node is the BRAIN of the automated reference-generation loop.

WHAT IT SEES
------------
It ONLY ever receives 4-5 GREY SILHOUETTE frames (filled body masks) taken
from the tracked video. There is NO face, NO skin, NO clothing texture and NO
colour in its input - just the solid outline of the person. Everything it
decides is therefore inferred from SHAPE alone:
    * gender            → from body proportions / overall silhouette
    * structural fit    → from how tight/loose/flared the outline is
    * garment cut/type  → from the silhouette of upper & lower garments

WHAT IT LOCKS (must stay constant every run)
--------------------------------------------
    * gender (strong Juggernaut-XL tokens so the subject is never mis-gendered)
    * the detected clothing STRUCTURE / FIT / CUT
      (e.g. loose, tight, frock, skinny jeans, oversized hoodie, wide shalwar)

WHAT IT RANDOMISES (must vary heavily every run → diverse characters)
---------------------------------------------------------------------
    * race / ethnicity / skin tone
    * hair colour AND hair style
    * clothing colours, prints and design details
    * minor aesthetic traits (age band, accessory, mood)
The randomisation is driven by the `seed` widget (set it to
control_after_generate = randomize) so every queue produces a brand-new look
while the locked structure & gender are preserved.

OUTPUTS
-------
    positive, negative              → the PRIMARY full-body front character
    info                            → human-readable detection / roll summary
    secondary_positive, secondary_negative
                                    → prompts for the SECONDARY collage chain
                                      (identity is carried by IPAdapter, so these
                                      only describe the multi-view collage format
                                      while keeping gender + garment locked)

IS_CHANGED returns random → node re-executes every queue → KSampler re-runs.
"""

import os, io, json, hashlib, random
import numpy as np
import torch
from PIL import Image

_ANALYSIS_CACHE = {}

# ---------------------------------------------------------------------------
# VLM prompt - written specifically for SILHOUETTE (filled-mask) input.
# ---------------------------------------------------------------------------
ANALYSIS_PROMPT = (
    "You are shown 4-5 GREY SILHOUETTE frames of ONE single person, taken from "
    "different moments of a short video. Each frame is a solid filled body mask: "
    "there is NO face, NO skin, NO hair detail, NO colour and NO fabric texture - "
    "only the flat outline/shape of the person and their clothing.\n"
    "Judge EVERYTHING from the SHAPE of the outline across all frames.\n"
    "1) Determine GENDER strictly and accurately. Weigh these silhouette cues together, "
    "across ALL frames, and do not decide from a single frame or from clothing style alone:\n"
    "  - shoulder width relative to hip width (males typically wider shoulders than hips; "
    "females typically hip width equal to or greater than shoulder width);\n"
    "  - waist definition and the taper from ribcage to waist to hip;\n"
    "  - chest/bust curvature on the outline;\n"
    "  - neck thickness relative to the head, and overall limb mass;\n"
    "  - head-to-body height ratio and general skeletal build;\n"
    "  - hair mass visible in the outline beyond the skull, and garment drape/length.\n"
    "   Do NOT infer gender from the garment being long, loose or robe-like - long loose "
    "clothing is worn by every gender in many cultures. Report your honest uncertainty in "
    "gender_confidence: use a value BELOW 0.6 whenever the cues genuinely conflict or the "
    "outline is ambiguous, rather than guessing confidently.\n"
    "2) Determine the structural CLOTHING FIT and the CUT/TYPE of the garments from "
    "the outline only (how close to the body, how flared, how long).\n"
    "3) Determine the subject's HEIGHT CLASS from body PROPORTIONS only - head-to-body height "
    "ratio, leg-length-to-torso ratio and overall slenderness of the standing outline. Use three "
    "coarse bins relative to an average adult: short / average / tall. Never guess from the frame "
    "size or from how much of the image the person fills (camera distance is unknown).\n"
    "Return ONLY this JSON, no other text:\n"
    '{"gender":"male" or "female",'
    '"gender_confidence":0.0 to 1.0,'
    '"build":"slim"|"average"|"heavy",'
    '"height_class":"short"|"average"|"tall",'
    '"garment_fit":"tight"|"fitted"|"regular"|"loose"|"oversized"|"baggy",'
    '"upper_garment":"short structural phrase describing the UPPER garment shape '
    'with NO colour and NO gender words, e.g. fitted t-shirt, loose kurta, '
    'oversized hoodie, structured blazer, flowing tunic",'
    '"lower_garment":"short structural phrase for the LOWER garment shape, or '
    "'not visible', NO colour, NO gender words, e.g. skinny jeans, wide trousers, "
    "flared long frock, straight shalwar, pleated skirt, tailored shorts\","
    '"silhouette_summary":"one short neutral sentence describing the overall outfit '
    'silhouette and how it sits on the body, no colour, no gender words"}\n'
    "The outfit may be from ANY culture or region - describe the actual silhouette, "
    "never default to Western clothing, never guess colour or pattern."
)

# Locked, structure-only fit vocabulary kept verbatim into the prompt.
_VALID_FIT = ("tight", "fitted", "regular", "loose", "oversized", "baggy")
# Coarse height bins. THREE bins on purpose: the subject's absolute height is already carried
# across the tier boundary by mask.mkv (ATTENTION_AND_RISKS §3.5 - "quantize scale to K population
# bins" is the documented mitigation), so reproducing it as a 3-way population bin adds no channel
# that the mask does not already carry, while stopping the generated character from being wildly
# mis-proportioned against the plate hole it has to fill.
_VALID_HEIGHT = ("short", "average", "tall")

# Default attribute dict - ONE definition, used as the fallback everywhere a VLM call can fail.
_ATTR_DEFAULTS = {"gender": "male", "gender_confidence": 0.0, "build": "average",
                  "height_class": "average", "garment_fit": "regular",
                  "upper_garment": "", "lower_garment": "", "silhouette_summary": ""}


def _normalize_attrs(d):
    """Coerce a raw VLM JSON dict into the strict attribute contract.

    Single definition shared by BOTH detectors (_analyze / _analyze_openai). It used to be
    duplicated verbatim in each, which is exactly how the two providers drift apart - and a
    drifted attribute set silently produces two DIFFERENT people across the two reference
    images, which is the failure this node exists to prevent."""
    g = str(d.get("gender", "male")).lower()
    d["gender"] = "female" if g.startswith("f") else "male"
    try:
        d["gender_confidence"] = float(d.get("gender_confidence", 0.7))
    except Exception:
        d["gender_confidence"] = 0.7
    d["build"] = str(d.get("build", "average")).lower()
    h = str(d.get("height_class", "average")).lower().strip()
    d["height_class"] = h if h in _VALID_HEIGHT else "average"
    f = str(d.get("garment_fit", "regular")).lower()
    d["garment_fit"] = f if f in _VALID_FIT else "regular"
    d["upper_garment"] = str(d.get("upper_garment", "")).strip()
    low = str(d.get("lower_garment", "")).strip()
    d["lower_garment"] = "" if low.lower() in ("", "not visible", "n/a", "none") else low
    d["silhouette_summary"] = str(d.get("silhouette_summary", "")).strip()
    return d


# ---------------------------------------------------------------------------
# HARD CONSTRAINTS shared VERBATIM by the reference-sheet prompt and the full-body
# prompt. Both images feed the SAME WanAnimate character, so any constraint stated
# in only one of them is a licence for the two references to disagree.
# ---------------------------------------------------------------------------
HARD_CONSTRAINTS = (
    "HARD CONSTRAINTS (apply to every panel and every pixel): a photorealistic PHOTOGRAPH of a "
    "real human being - never an illustration, never a painting, never anime, never a 3D render, "
    "never CGI, never a mannequin or doll; exactly ONE individual and one identity throughout; "
    "plain uniform seamless neutral grey studio background (hex 808080) with no scene, no room, "
    "no furniture, no outdoor location and no props or held objects; soft even studio lighting "
    "with no hard clipped highlights; sharp focus with true-to-life skin texture and visible "
    "fabric weave; and absolutely no text, captions, labels, watermarks, logos, borders or frames."
)
# Framing/pose spec for any FULL-BODY view (used by the full-body prompt AND quoted inside the
# reference sheet's full-length panel, so the two agree on what "full body" means).
FULLBODY_FRAMING = (
    "the ENTIRE body from the top of the head to the soles of the shoes inside the frame with "
    "clear margin on every side and nothing cropped at any edge, standing upright and facing the "
    "camera in a neutral relaxed A-pose - feet about shoulder width apart, arms hanging relaxed "
    "and angled slightly away from the torso so the garment silhouette is unobstructed, weight "
    "even on both feet, neutral calm expression"
)

# Shared photoreal aesthetic. Appended to BOTH the primary and secondary
# positives. Written to push Juggernaut-XL toward true photographic skin/fabric
# micro-detail while staying LIGHTING-NEUTRAL: the character is harmonized to the
# plate downstream (the light_map conditioning), so the reference sheet keeps
# SOFT, EVENLY-EXPOSED directional light (never hard/dramatic) and a uniform
# neutral-grey backdrop so SAM2 can segment the character cleanly.
AESTHETIC = (
    "photorealistic photograph, shot on a full-frame DSLR with an 85mm f/4 lens, "
    "ultra-sharp focus across the whole body, high dynamic range, true-to-life colour, "
    "(visible skin pores:1.2), fine vellus facial hair, natural skin micro-texture, "
    "realistic subsurface scattering, subtle natural skin imperfections, non-uniform lifelike skin, "
    "(macroscopic fabric weave:1.2), individually visible textile threads, "
    "tactile woven cloth fibres, detailed seams and stitching, "
    "natural soft directional key light with gentle cinematic falloff that rakes across "
    "the skin and fabric to reveal texture, delicate soft shadows for form, "
    "evenly exposed, no hard clipping, 8k ultra-high detail, sharp focus, "
    "perfectly uniform matte neutral grey seamless studio backdrop hex 808080"
)

# ---------------------------------------------------------------------------
# PHASE-4 required token blocks. The character is generated on a grey backdrop
# (for clean SAM2 segmentation) but its KEY LIGHT / colour is pushed to match
# the plate; true scene shadows & integration are finished in TIER-2C compositing.
# ---------------------------------------------------------------------------
REQUIRED_POS = (
    "high-texture hyper-realistic skin, natural physical variations, "
    "diverse characteristics, seamless scene integration, 8k resolution"
)
REQUIRED_NEG = (
    "glowing aura, disconnected shadows, flat lighting, plastic skin, "
    "jittery eyes, asymmetric eyes, mutating irises, breaking geometry, "
    "flaky edges, CGI, 3d render, "
    # keep the reference backdrop clean & grey (segmentation safety)
    "scene background, environment background, location, outdoor scenery, "
    "colored backdrop, busy backdrop"
)


def _analyze_bg_lighting(bg):
    """LOCAL (never-cloud) analysis of the person-REMOVED filled background plate.
    Infers key-light direction, colour temperature and palette so the reference
    character can be lit to match the scene.  Input is threat-safe: it contains
    NO subject pixels (temporal-median background with the person masked out)."""
    a = bg.detach().cpu().numpy() if hasattr(bg, "detach") else np.asarray(bg)
    if a.ndim == 3:
        a = a[None, ...]
    a = a.astype(np.float32).mean(axis=0)                 # temporal average [H,W,3]
    H, W, _ = a.shape
    R, G, Bc = float(a[..., 0].mean()), float(a[..., 1].mean()), float(a[..., 2].mean())
    lum = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    brightness = float(lum.mean())
    warmth = R - Bc
    ys = np.linspace(-1, 1, H)[:, None]
    xs = np.linspace(-1, 1, W)[None, :]
    wsum = float(lum.sum()) + 1e-6
    cx = float((lum * xs).sum() / wsum)
    cy = float((lum * ys).sum() / wsum)

    tone = ("bright high-key" if brightness > 0.62 else
            "soft naturally-lit" if brightness > 0.32 else "dim low-key moody")
    temp = ("warm golden-hour" if warmth > 0.06 else
            "cool blue" if warmth < -0.05 else "neutral white-balanced")
    hx = "left" if cx < -0.08 else ("right" if cx > 0.08 else "")
    vy = "upper" if cy < -0.08 else ("lower" if cy > 0.08 else "")
    where = (vy + " " + hx).strip() or "frontal"
    hexc = "#%02x%02x%02x" % (int(np.clip(R, 0, 1) * 255),
                              int(np.clip(G, 0, 1) * 255),
                              int(np.clip(Bc, 0, 1) * 255))
    clause = (
        f"{tone} {temp} cinematic lighting matching the background scene, "
        f"soft directional key light on the subject from the {where}, "
        f"directional environmental shadows consistent with the scene, "
        f"skin tones and colour palette harmonising with the scene ambience ({hexc})"
    )
    summary = f"tone={tone} temp={temp} keylight={where} avg={hexc} bright={brightness:.2f}"
    return clause, summary

# ---------------------------------------------------------------------------
# Diversity pools - these are what get RANDOMISED every run.
# Gender and clothing-structure are NEVER drawn from here; they are locked.
# ---------------------------------------------------------------------------
RACE_POOL = [
    "South Asian", "Pakistani", "North Indian", "South Indian", "Bangladeshi",
    "Sri Lankan", "Nepali", "East Asian", "Korean", "Japanese", "Han Chinese",
    "Tibetan", "Mongolian", "Southeast Asian", "Filipino", "Vietnamese", "Thai",
    "Indonesian", "Malaysian", "Burmese", "Middle Eastern", "Persian", "Arab",
    "Lebanese", "Turkish", "Kurdish", "Central Asian", "Kazakh", "Uzbek",
    "North African", "Egyptian", "Moroccan", "Berber", "West African", "Nigerian",
    "Ghanaian", "East African", "Ethiopian", "Somali", "Kenyan", "Congolese",
    "African American", "Afro-Caribbean", "Jamaican", "Afro-Latino",
    "Northern European", "Scandinavian", "Celtic", "Slavic", "Eastern European",
    "Mediterranean", "Italian", "Greek", "Iberian", "Armenian", "Georgian",
    "Hispanic", "Latino", "Brazilian", "Mexican", "Colombian", "Peruvian",
    "Indigenous American", "Polynesian", "Native Hawaiian", "Māori",
    "Aboriginal Australian", "Eurasian", "mixed-race", "multiracial",
]
SKIN_POOL = [
    "fair porcelain skin", "pale ivory skin", "light skin", "light olive skin",
    "warm beige skin", "sun-kissed tan skin", "medium tan skin", "golden tan skin",
    "honey-toned skin", "deep olive skin", "light brown skin", "caramel skin",
    "medium brown skin", "bronze skin", "rich brown skin", "sienna skin",
    "deep brown skin", "mahogany skin", "dark ebony skin", "espresso skin",
]
HAIR_COLOR_POOL = [
    "jet black", "natural black", "soft black with brown undertones", "dark brown",
    "chocolate brown", "chestnut brown", "light golden brown", "warm auburn",
    "reddish auburn", "copper red", "strawberry blonde", "honey blonde",
    "sandy blonde", "ash blonde", "platinum blonde", "icy platinum", "dark grey",
    "salt-and-pepper", "silver-grey", "burgundy-tinted", "dark with caramel highlights",
    "dark with subtle balayage",
]
# MEN → LONG ONLY (user decision 2026-07-26: "all must be long hairs for their own gender").
# Previously MID-LENGTH only, which itself replaced an even shorter pool. Long *masculine*
# styles - length past the shoulders, but cut/tied in ways that read male, so the VLM's
# (already accurate) gender read is not fought by the hair.
HAIR_STYLE_MALE = [
    "long loose hair past the shoulders", "long wavy hair worn down",
    "long hair tied in a low bun", "long hair in a half-up top knot",
    "long straight hair tucked behind the ears", "long tousled hair with natural volume",
    "long curly hair worn loose", "long hair in a low ponytail",
    "long layered hair parted in the middle", "long shoulder-length waves",
    "long hair in a loose man-bun with strands framing the face",
    "long dreadlocks gathered back", "long braided hair", "long hair swept back",
    "long flowing hair with a slight wave", "long hair in a top knot with an undercut-free nape",
]
# WOMEN → LONG ONLY (always well past the shoulders). No pixie/bob/lob/short.
HAIR_STYLE_FEMALE = [
    "long straight hair", "long loose waves", "long soft beach waves",
    "long voluminous curly hair", "long loose curls", "long flowing afro",
    "long box braids", "long cornrow braids", "long braided ponytail",
    "long center-parted hair", "long layered hair", "long half-up half-down style",
    "waist-length wavy hair", "long sleek hair", "long hair with a soft fringe",
    "long high ponytail",
]
CLOTH_COLOR_POOL = [
    "deep navy", "steel blue", "royal blue", "powder blue", "dusty blue", "indigo",
    "teal", "turquoise", "aqua", "charcoal", "slate grey", "forest green",
    "emerald", "sage green", "mint green", "moss green", "olive", "khaki",
    "mustard yellow", "ochre", "pale yellow", "rust orange", "terracotta",
    "brick red", "burgundy", "wine red", "maroon", "coral", "salmon", "peach",
    "dusty rose", "blush pink", "mauve", "lavender", "plum", "deep purple",
    "cream", "off-white", "warm beige", "sand", "camel", "chocolate brown", "rose gold",
]
# CLOTHING DESIGN. Every entry must be VISIBLE AT A GLANCE on a full-body shot - 
# the old pool opened with "solid colour" / "subtle tonal texture" and was ordered
# plain -> decorative, so _bucket()'s CONTIGUOUS split handed Person 0 the plain half
# every single time (and Person 1 the subtle-texture half). That is exactly why the
# characters came out "different colours but still too plain". Two fixes:
#   1. the flat/plain entries are GONE (a bare "solid colour" is never a design);
#   2. the pool is INTERLEAVED (bold/ethnic/texture/geometric rotate) and the design
#      pools are bucketed with _bucket_strided(), so neither person can be handed a
#      run of same-energy entries. See _bucket_strided().
CLOTH_DESIGN_POOL = [
    "bold woven ethnic motifs", "large floral embroidery across the chest and sleeves",
    "wide vertical colour stripes", "intricate paisley print", "geometric colour-blocked panels",
    "hand-blocked batik print", "rich damask brocade", "graphic statement print",
    "dense tribal motif print", "ornate embroidered neckline and cuffs",
    "tie-dye gradient", "bold checkered pattern", "mirrored shisha embroidery",
    "abstract painterly print", "chunky ribbed knit texture", "patchwork panels",
    "gold-thread zari embroidery", "ikat woven pattern", "large polka dots",
    "bold tropical leaf print", "quilted diamond texture", "colour gradient ombré",
    "houndstooth pattern", "kantha running-stitch embroidery", "wide horizontal bands",
    "phulkari floral thread-work", "corduroy ribbing", "animal print accent panels",
    "screen-printed folk motifs", "contrast piping and embroidered trim",
    "woven herringbone", "lace overlay detail", "mosaic-tile geometric print",
    "bandhani tie-dye dots", "satin sheen with tonal jacquard", "block-printed border motifs",
]
# Overall styling vibe - randomised so every character reads differently.
STYLE_FLAVOR_POOL = [
    "relaxed bohemian style", "modern streetwear style", "urban hip-hop style",
    "sporty street style", "smart-casual style", "business-casual style",
    "preppy style", "minimalist contemporary style", "normcore style",
    "traditional ethnic style", "athleisure style", "vintage retro style",
    "y2k revival style", "grunge style", "artsy eclectic style", "avant-garde style",
    "cottagecore style", "romantic feminine style", "elegant chic style",
    "utilitarian workwear style", "rugged outdoor style", "casual everyday style",
    "festival style", "coastal resort style",
]
# BAGGY garment silhouettes. User decision 2026-07-26: "keep them baggy like shalwar kameez,
# gown, baggy clothes". The previous pool was deliberately NOT baggy ("just comfortably
# relaxed"), which fought the GARMENT_POOL entries - those are already shalwar/thobe/abaya/
# kaftan cuts and read wrong when described as neatly tailored.
# HALO-SAFE by construction: a baggier garment covers MORE of the background-plate hole than a
# fitted one, never less (STATE.md → V8.3), so this cannot open a §2 reveal.
LOOSE_SILHOUETTE_POOL = [
    "baggy and voluminous", "loose and flowing", "generously oversized",
    "wide and draping", "billowing and airy", "roomy with deep folds",
    "oversized with a heavy drape", "loose-fitting with generous volume",
    "flowing with wide sweeping lines", "baggy with soft gathered folds",
    "voluminous and unstructured", "wide-cut and softly falling",
]
# ---------------------------------------------------------------------------
# GARMENT TYPE. Previously the garment CUT was locked to whatever the VLM read off
# the silhouette (_garment_lock), which made every character wear the same generic
# top+trousers as the real person. These pools give real cultural variety while
# staying inside the pipeline's existing hard constraint: everything here is LOOSE,
# ROOMY or FLOWING, because _loosen() already forces the fit to loose..baggy anyway.
# That is also HALO-SAFE: a roomier garment covers MORE of the background-plate hole
# (see STATE.md -> V8.3), never less. `None` for the lower piece = a one-piece garment.
# Entries deliberately alternate sub-continental / African / Middle-Eastern / Western
# so _bucket_strided() cannot hand one person a single cultural family.
# ---------------------------------------------------------------------------
GARMENT_POOL_MALE = [
    ("a long loose kameez tunic", "baggy pleated shalwar trousers"),
    ("an oversized hoodie", "baggy joggers"),
    ("a flowing full-length thobe robe", None),
    ("a knee-length kurta", "loose pyjama trousers"),
    ("a wide-sleeved boubou robe", None),
    ("a loose linen shirt", "wide-leg trousers"),
    ("an embroidered dashiki", "loose drawstring trousers"),
    ("an oversized crewneck sweatshirt", "baggy cargo trousers"),
    ("a long open kaftan over a plain tunic", "loose trousers"),
    ("a loose short-sleeved guayabera shirt", "loose pleated chinos"),
    ("a draped shawl over a long loose tunic", "wide trousers"),
    ("an oversized bomber jacket over a loose tee", "baggy jeans"),
    ("a collarless loose grandad shirt", "wide drawstring trousers"),
    ("a long quilted waistcoat over a loose kurta", "baggy shalwar trousers"),
]
GARMENT_POOL_FEMALE = [
    ("a long loose kameez tunic with a draped dupatta", "baggy pleated shalwar trousers"),
    ("a floor-length flowing gown", None),
    ("an oversized hoodie", "baggy joggers"),
    ("a flowing full-length abaya", None),
    ("a loose flowing tunic", "wide palazzo trousers"),
    ("a voluminous anarkali frock", "loose churidar trousers"),
    ("a wide flowing kaftan", None),
    ("an oversized knit jumper", "wide-leg trousers"),
    ("a wide-sleeved boubou robe", None),
    ("a long loose shirt-dress", None),
    ("a draped wrap over a loose blouse", "a long flowing skirt"),
    ("a long loose tunic with an embroidered dupatta", "baggy trousers"),
    ("a tiered maxi dress with a loose bodice", None),
    ("an oversized denim jacket over a loose midi dress", None),
]

AGE_POOL = [
    "in their early 20s", "in their mid 20s", "in their late 20s",
    "in their early 30s", "in their mid 30s",
]
ACCESSORY_POOL = [
    "no accessories", "small stud earrings", "a simple necklace", "thin-framed glasses",
    "a wristwatch", "a subtle ring", "minimal jewellery", "a light scarf",
]


def _sharpness(a):
    g = a.astype(np.float32).mean(axis=2) if a.ndim == 3 else a.astype(np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = 4*g[1:-1,1:-1] - g[:-2,1:-1] - g[2:,1:-1] - g[1:-1,:-2] - g[1:-1,2:]
    return float(lap.var())


def _pick_frames(images, n):
    """Pick up to n distinct, well-formed silhouette frames spread across the clip."""
    arr = images.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[None, ...]
    B = arr.shape[0]
    if B <= n:
        idxs = list(range(B))
    else:
        cand = np.linspace(0, B-1, min(B, max(n*4, n))).astype(int)
        scored = sorted(
            ((_sharpness(np.clip(arr[i]*255,0,255).astype(np.uint8)), int(i)) for i in cand),
            reverse=True
        )
        idxs = sorted(i for _, i in scored[:n])
    pils = [Image.fromarray(np.clip(arr[i]*255,0,255).astype(np.uint8)) for i in idxs]
    key  = hashlib.md5(
        np.ascontiguousarray((arr[idxs][:,::8,::8]*255).astype(np.uint8))
    ).hexdigest()
    return pils, key


def _analyze(frames, model, api_key):
    """Run Gemini on the silhouette frames → structured shape description."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    parts  = [types.Part.from_text(text=ANALYSIS_PROMPT)]
    for img in frames:
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90)
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
    resp = client.models.generate_content(
        model=model, contents=parts,
        config=types.GenerateContentConfig(temperature=0.1,
                                            response_mime_type="application/json"))
    txt = resp.text or ""
    s, e = txt.find("{"), txt.rfind("}")
    return _normalize_attrs(json.loads(txt[s:e+1]))


# Fit ladder, loosest last. The detected fit is pushed TWO steps looser (and
# never allowed below "loose") so the rendered clothing always reads roomy - 
# frocks, baggy, oversized - instead of tight.
FIT_LADDER = ["tight", "fitted", "regular", "loose", "oversized", "baggy"]

def _loosen(fit, steps=2, floor="loose", ceil=None):
    try:    i = FIT_LADDER.index(fit)
    except ValueError: i = FIT_LADDER.index("regular")
    i = max(i + steps, FIT_LADDER.index(floor))
    i = min(i, len(FIT_LADDER) - 1)
    if ceil is not None:                      # cap so clothes stay only SLIGHTLY loose
        i = min(i, FIT_LADDER.index(ceil))
    return FIT_LADDER[i]


def _bucket(pool, idx, total):
    """Deterministic per-person DIVERGENCE: split `pool` into `total` disjoint
    contiguous segments and return segment `idx`. Person 0 and Person 1 therefore
    draw their race / skin / hair-colour / clothing-colour / style from
    NON-OVERLAPPING slices of each pool, so the two generated characters can never
    converge on the same look (the pools are region/colour ordered, so the split
    also lands them in different ethnicity families and different colour families).
    The `seed` still randomises freely WITHIN each person's segment."""
    if total is None or total <= 1 or not pool:
        return pool
    idx = max(0, int(idx)) % int(total)
    n = len(pool)
    lo = (n * idx) // total
    hi = (n * (idx + 1)) // total
    seg = pool[lo:hi]
    return seg if seg else pool


def _bucket_strided(pool, idx, total):
    """Per-person divergence for pools whose ORDER carries meaning other than
    'family' - i.e. design and garment, where a contiguous _bucket() slice would hand
    one person a run of same-energy entries.

    _bucket() takes a CONTIGUOUS slice, which is right for region/colour-ordered pools
    (it lands the two people in different ethnicity/colour families on purpose). It was
    WRONG for CLOTH_DESIGN_POOL: that pool ran plain -> decorative, so Person 0 was
    permanently confined to the plain half ("solid colour", "subtle tonal texture",
    pinstripes...) and Person 1 to the subtle-texture half. Both read as plain - the
    "different colours but still too plain" bug.

    Dealing the pool like cards (pool[idx::total]) keeps the two people DISJOINT while
    giving each the full spread of the pool's range."""
    if total is None or total <= 1 or not pool:
        return list(pool)
    idx = max(0, int(idx)) % int(total)
    seg = list(pool)[idx::int(total)]
    return seg if seg else list(pool)


def _garment_lock(a, fit, silhouette):
    """LOCKED garment-type clause, rendered at the loosened `fit`."""
    parts = []
    if a.get("upper_garment"):
        parts.append(a["upper_garment"])
    if a.get("lower_garment"):
        parts.append(a["lower_garment"])
    if parts:
        return f"wearing {silhouette} {fit}-fit {' and '.join(parts)}"
    return f"wearing {silhouette} {fit}-fitting clothing"


class MIRAGE_AutoRefPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "silhouette_frames": ("IMAGE",),
            "seed":            ("INT",    {"default": 0, "min": 0,
                                           "max": 0xffffffffffffffff,
                                           "control_after_generate": True}),
            "num_frames":      ("INT",    {"default": 5, "min": 1, "max": 8}),
            "gender_override": (["auto", "male", "female"],),
            "vlm_model":       ("STRING", {"default": "gemini-2.5-flash"}),
            "api_key":         ("STRING", {"default": ""}),
        },
        "optional": {
            # person-REMOVED filled background plate (temporal median) - analysed
            # LOCALLY for lighting/palette; NEVER sent to the cloud VLM.
            "background_plate": ("IMAGE",),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "info",
                    "secondary_positive", "secondary_negative")
    FUNCTION = "run"
    CATEGORY = "MIRAGE"

    def run(self, silhouette_frames, seed, num_frames, gender_override, vlm_model,
            api_key, background_plate=None):
        key = (api_key or "").strip() \
            or os.environ.get("GEMINI_API_KEY", "").strip() \
            or os.environ.get("GOOGLE_API_KEY", "").strip()

        # ---- detect STRUCTURE + GENDER from silhouettes (frame-hash cached) ----
        a = dict(_ATTR_DEFAULTS)
        frames, fhash = _pick_frames(silhouette_frames, int(num_frames))
        if key:
            # §2 EGRESS GUARD - this is the OTHER path that uploads silhouette_frames. The audit
            # note above _egress_check once claimed "no other tensor reaches an API call", but
            # `_analyze` below posts these exact frames to the VLM, and this call site had no
            # check at all: rewiring a real camera feed here would have leaked it silently while
            # the guarded sibling node reported everything was fine. Fail CLOSED, same as there.
            _egress_check(frames, strict=True)
            try:
                if fhash not in _ANALYSIS_CACHE:
                    _ANALYSIS_CACHE[fhash] = _analyze(frames, vlm_model, key)
                a = _ANALYSIS_CACHE[fhash]
            except Exception as ex:
                print("[MIRAGE_AutoRefPrompt] VLM failed, using defaults:", ex)

        gender, conf = a["gender"], a["gender_confidence"]
        if gender_override in ("male", "female"):
            gender, conf = gender_override, 1.0

        # ---- RANDOMISED aesthetics (seeded → diverse every queue) ----
        rng = random.Random(int(seed) ^ 0x5DEECE66D)
        race        = rng.choice(RACE_POOL)
        skin        = rng.choice(SKIN_POOL)
        hair_color  = rng.choice(HAIR_COLOR_POOL)
        cloth_a     = rng.choice(CLOTH_COLOR_POOL)
        cloth_b     = rng.choice([c for c in CLOTH_COLOR_POOL if c != cloth_a])
        design      = rng.choice(CLOTH_DESIGN_POOL)
        style       = rng.choice(STYLE_FLAVOR_POOL)
        silhouette  = rng.choice(LOOSE_SILHOUETTE_POOL)
        age         = rng.choice(AGE_POOL)
        accessory   = rng.choice(ACCESSORY_POOL)

        # Clothing FIT is locked to the detected structure but pushed two steps
        # looser so the result is always roomy (frock / baggy / oversized).
        detected_fit = a.get("garment_fit", "regular")
        locked_fit   = _loosen(detected_fit, steps=2, floor="loose")
        garment_lock = _garment_lock(a, locked_fit, silhouette)

        # ---- LOCKED gender tokens (critical for Juggernaut XL) ----
        if gender == "male":
            gender_tokens   = "(male:1.5), (man:1.5), masculine face, masculine jawline"
            gender_negative = ("woman, female, feminine, girl, lady, makeup, lipstick, "
                               "long eyelashes, mascara, eyeshadow, female face")
            hair_style_pool = HAIR_STYLE_MALE
        else:
            gender_tokens   = "(female:1.5), (woman:1.5), feminine face"
            gender_negative = ("man, male, masculine, guy, beard, stubble, mustache, "
                               "male face")
            hair_style_pool = HAIR_STYLE_FEMALE

        # hair STYLE depends on the locked gender pool (chosen here so the rng
        # stream stays deterministic for a given seed)
        hair_style = rng.choice(hair_style_pool)

        common_negative = (
            "blurry, grainy, low quality, deformed, disfigured, bad anatomy, "
            "asymmetric face, extra limbs, extra fingers, fused fingers, text, "
            "watermark, logo, cartoon, illustration, 3d render, "
            "plastic skin, waxy skin, airbrushed skin, over-smoothed skin, "
            "smooth featureless skin, doll-like, cgi skin, flat texture, "
            "blurred fabric, smeared fabric, "
            "colored background, gradient background, busy background"
        )
        # Keep the rendered outfit ROOMY - explicitly reject tight clothing.
        tight_negative = (
            "tight clothing, fitted clothing, skin-tight, form-fitting, bodycon, "
            "skinny jeans, tight trousers, tight pants, compression wear, "
            "tucked-in shirt"
        )

        # ---- data-driven scene lighting (Phase 4.4). Falls back to a generic
        #      cinematic clause when no background plate is supplied. ----
        if background_plate is not None:
            try:
                light_clause, light_summary = _analyze_bg_lighting(background_plate)
            except Exception as ex:
                print("[MIRAGE_AutoRefPrompt] bg lighting analysis failed:", ex)
                light_clause = ("cinematic lighting matching the background, "
                                "directional environmental shadows")
                light_summary = "bg-analysis-failed"
        else:
            light_clause = ("cinematic lighting matching the background, "
                            "directional environmental shadows")
            light_summary = "no-bg-plate"

        # ---- PRIMARY: front-facing full-body character (face clearly visible) ----
        positive = (
            f"full-body front-facing studio photograph of a {gender}, "
            f"{gender_tokens}, {age}, {race} ethnicity, {skin}, "
            f"{hair_color} {hair_style}, {a['build']} build, "
            f"{style}, {garment_lock}, {cloth_a} and {cloth_b} {design}, "
            f"loose comfortable roomy fit, {accessory}, "
            f"standing upright, head to toe in frame, face clearly visible and sharp, "
            f"looking directly at the camera, neutral confident expression, "
            f"full body shot, {light_clause}, {REQUIRED_POS}, {AESTHETIC}"
        )
        negative = (
            f"{gender_negative}, {tight_negative}, cropped head, cut-off body, "
            f"close-up only, {REQUIRED_NEG}, {common_negative}"
        )

        # ---- SECONDARY: collage-format prompt (identity comes from IPAdapter) ----
        # Gender + garment stay LOCKED; we do NOT re-roll aesthetics here so the
        # secondary keeps the EXACT same person/clothing as the primary image.
        secondary_positive = (
            f"professional full-body reference collage of the same single {gender}, "
            f"two full-body views side by side - a front three-quarter view and a "
            f"side profile view - the exact same person in both, "
            f"identical face matching the reference, same facial features, same hair, "
            f"same clothing in both views, clear sharp symmetrical undistorted face, "
            f"detailed natural facial features, {gender_tokens}, {a['build']} build, "
            f"{garment_lock}, head to toe, standing upright, "
            f"perfectly consistent single identity across both poses, "
            f"{light_clause}, {REQUIRED_POS}, {AESTHETIC}"
        )
        secondary_negative = (
            f"{gender_negative}, {tight_negative}, different people, two different faces, "
            f"inconsistent face, face that differs from the reference, "
            f"distorted face, warped face, melted face, mutated face, "
            f"inconsistent clothing between views, merged bodies, "
            f"cropped head, cut-off body, {REQUIRED_NEG}, {common_negative}"
        )

        info = (
            f"DETECTED gender={gender} ({conf:.2f}) build={a['build']} "
            f"fit={a['garment_fit']}->{locked_fit} (2 looser) "
            f"upper={a['upper_garment'] or 'n/a'} lower={a['lower_garment'] or 'n/a'} | "
            f"ROLL race={race} {style} hair={hair_color}/{hair_style} "
            f"cloth={cloth_a}+{cloth_b}/{design} sil={silhouette} | LIGHT {light_summary}"
        )
        print("[MIRAGE_AutoRefPrompt]", info)
        return (positive, negative, info, secondary_positive, secondary_negative)

    @classmethod
    def IS_CHANGED(cls, silhouette_frames, seed, num_frames,
                   gender_override, vlm_model, api_key, background_plate=None):
        # New random value every queue → node always re-executes → KSampler re-runs.
        return str(random.random())


# ===========================================================================
# MIRAGE_GeminiAutoRef - silhouette -> Gemini-generated PRIMARY + SECONDARY
# reference images of the SAME invented person & clothing.
# ===========================================================================
def _pil_to_tensor(img):
    a = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(a)[None, ...]                       # 1,H,W,3


def _gemini_image(prompt, ref_pils, model, key):
    """Generate ONE image with Gemini's image model. `ref_pils` are attached as
    reference parts (identity / clothing / pose guides). Returns a PIL RGB image."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    contents = [types.Part.from_text(text=prompt)]
    for img in (ref_pils or []):
        b = io.BytesIO(); img.convert("RGB").save(b, format="PNG")
        contents.append(types.Part.from_bytes(data=b.getvalue(), mime_type="image/png"))
    resp = client.models.generate_content(model=model, contents=contents)
    for cand in (resp.candidates or []):
        for part in (getattr(cand.content, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return Image.open(io.BytesIO(inline.data)).convert("RGB")
    # surface any text the model returned instead of an image (usually a refusal)
    note = (resp.text or "").strip()[:200] if getattr(resp, "text", None) else ""
    raise RuntimeError("Gemini returned no image. Ensure image_model supports image output and your "
                       "key/tier has image generation enabled." + (f" Model said: {note}" if note else ""))


def _openai_image(prompt, ref_pils, model, key, size, quality):
    """Generate ONE image via OpenAI gpt-image-1. With ref_pils -> images.edit (the
    references drive identity/outfit consistency); without -> images.generate."""
    import base64
    from openai import OpenAI
    client = OpenAI(api_key=key)
    kw = {"model": model, "prompt": prompt, "n": 1}
    if size and size != "auto":
        kw["size"] = size
    if quality and quality != "auto":
        kw["quality"] = quality
    if ref_pils:
        files = []
        for i, p in enumerate(ref_pils):
            b = io.BytesIO(); p.convert("RGB").save(b, format="PNG"); b.seek(0); b.name = f"ref{i}.png"
            files.append(b)
        resp = client.images.edit(image=files, **kw)
    else:
        resp = client.images.generate(**kw)
    b64 = getattr(resp.data[0], "b64_json", None)
    if not b64:
        raise RuntimeError("OpenAI returned no image data.")
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _analyze_openai(frames, model, key):
    """Silhouette -> structured gender/clothing description via OpenAI vision (cheap:
    gpt-4o-mini). Same ANALYSIS_PROMPT/normalisation as the Gemini detector."""
    import base64
    from openai import OpenAI
    client = OpenAI(api_key=key)
    content = [{"type": "text", "text": ANALYSIS_PROMPT}]
    for img in frames:
        b = io.BytesIO(); img.convert("RGB").save(b, format="JPEG", quality=90)
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()}})
    resp = client.chat.completions.create(model=model, temperature=0.1,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": content}])
    txt = resp.choices[0].message.content or ""
    s, e = txt.find("{"), txt.rfind("}")
    return _normalize_attrs(json.loads(txt[s:e + 1]))


# ---------------------------------------------------------------------------
# §2 EGRESS GUARD - the only images this node may upload to a third-party API are
# GREY SILHOUETTES (filled body masks) and images the API itself generated.
#
# What actually leaves (audited 2026-07-23; corrected the same day - the first audit said
# "both call sites" and MISSED one, so this list is now per-NODE and each entry names its guard):
#   MIRAGE_GeminiAutoRef.run - guarded at the top of run()
#     1. _analyze* : the `silhouette_frames` picked by _pick_frames  -> vision VLM
#     2. _genimg   : the FIRST `silhouette_refs` of those SAME frames -> image model
#     3. _genimg   : the generated collage, as the full-body reference -> image model
#   MIRAGE_AutoRefPrompt.run - WAS UNGUARDED until 2026-07-23; now guarded before _analyze
#     4. _analyze  : the same `silhouette_frames` -> vision VLM
#   `background_plate` is analysed by _analyze_bg_lighting with numpy ONLY and is
#   never attached to any request.
# Anything added here later MUST call _egress_check first. Grep for `_analyze` and `_genimg`
# before trusting this list - the last audit's blind spot was a sibling node, not a new path.
# So the whole question is whether `silhouette_frames` really is a silhouette - which
# depends on GRAPH WIRING this node cannot see. In V9_CLOUD_ONLY it is
# MaskToImage(mask_person_left/right), i.e. a pure mask. This guard verifies that at
# RUNTIME so a future rewiring cannot silently start uploading real camera pixels.
# ---------------------------------------------------------------------------
def _silhouette_stats(pils):
    """Measure how mask-like the frames about to leave the device are - PER FRAME.

    Returns a list of (mean_chroma, p99_chroma, top2_luminance_mass), one entry per frame.

    This used to aggregate across the batch (concatenated chroma, MEAN of top2), which made the
    guard defeatable by DILUTION: a greyscale photographic frame is refused on its own, but
    hidden 1-in-8 among masks its low tone-mass was averaged against seven 1.000s to 0.891 and
    sailed past the 0.85 threshold. Measured 2026-07-23. Every frame is uploaded individually,
    so every frame must be judged individually - and that holds for any batch size and any
    mask:photo ratio, rather than for the one ratio a threshold was tuned against.
    """
    out = []
    for img in pils:
        a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        c = a.max(axis=2) - a.min(axis=2)               # 0 for any true greyscale pixel
        lum = a.mean(axis=2)
        hist, _ = np.histogram(lum, bins=32, range=(0.0, 1.0))
        out.append((float(c.mean()), float(np.percentile(c, 99)),
                    float(np.sort(hist)[-2:].sum()) / max(1, lum.size)))
    return out


def _egress_check(pils, strict=True):
    """§2 gate on everything about to cross the trust boundary. Returns a report string;
    raises when `strict` and the frames do not look like filled masks.

    Two independent conditions, both of which a real camera frame fails:
      * COLOUR: a mask rendered through MaskToImage has R==G==B exactly, so chroma is 0.
        Any real RGB content shows up here immediately.
      * TONE  : a filled mask is essentially two-valued (fill + background), so two of 32
        luminance bins hold almost every pixel. A photograph spreads across the range.
    Fail CLOSED (raise) rather than upload: a false alarm costs one render, a miss uploads
    a bystander's real pixels to a third-party API and cannot be undone."""
    stats = _silhouette_stats(pils)
    if not stats:
        raise RuntimeError("[MIRAGE_GeminiAutoRef] §2 EGRESS GUARD: no frames to check.")
    # EVERY frame must pass on its OWN. One bad frame condemns the batch, however many good
    # frames surround it - dilution is exactly the bypass this guard exists to stop.
    offenders = []
    for i, (mean_c, p99_c, top2) in enumerate(stats):
        why = []
        if mean_c > 0.02 or p99_c > 0.08:
            why.append("carries COLOUR (a filled mask is pure greyscale)")
        if top2 < 0.85:
            why.append("tone distribution is photographic, not two-valued")
        if why:
            offenders.append(f"frame {i}: " + " and ".join(why)
                             + f" (chroma mean={mean_c:.4f} p99={p99_c:.4f} tone={top2:.3f})")
    worst = min(s[2] for s in stats)
    report = (f"{len(stats)} frame(s); worst chroma mean={max(s[0] for s in stats):.4f} "
              f"p99={max(s[1] for s in stats):.4f} worst top2-tone-mass={worst:.3f}")
    bad = offenders
    if bad and strict:
        raise RuntimeError(
            "[MIRAGE_GeminiAutoRef] §2 EGRESS GUARD REFUSED TO UPLOAD: silhouette_frames do not "
            "look like filled body masks - " + "; ".join(bad) + f" ({report}). This node may only "
            "send GREY SILHOUETTES to the cloud API; real camera pixels must never leave. Check "
            "what is wired into silhouette_frames (expected: MaskToImage of the person mask). "
            "Set egress_guard=False ONLY if you have confirmed by eye that the input is a mask.")
    if bad:
        # ASCII-only in prints: a cp1252 console raises UnicodeEncodeError on box/warning glyphs.
        print("[MIRAGE_GeminiAutoRef] WARNING: EGRESS GUARD DISABLED and input looks non-mask: "
              + "; ".join(bad) + f" ({report})")
    return report


def _validate_generated(img, want_wh=None, min_side=256, min_std=6.0, aspect_tol=0.12):
    """Sanity-check an image the API returned BEFORE it is passed downstream.
    Returns (ok, reason). Catches the failure modes that otherwise surface as a ruined
    render minutes later: a blank/solid-colour canvas, a thumbnail, or a wrong-aspect
    image that gets letterboxed into the character reference."""
    if img is None:
        return False, "no image returned"
    w, h = img.size
    if min(w, h) < min_side:
        return False, f"image too small ({w}x{h}, need min side {min_side})"
    std = float(np.asarray(img.convert("RGB"), dtype=np.float32).std())
    if std < min_std:
        return False, f"image is blank/near-uniform (pixel std {std:.2f} < {min_std})"
    if want_wh:
        want = float(want_wh[0]) / float(want_wh[1])
        got = float(w) / float(h)
        if abs(got - want) / want > aspect_tol:
            return False, f"aspect {got:.3f} differs from requested {want:.3f} by >{aspect_tol:.0%}"
    return True, f"{w}x{h} std={std:.1f}"


def _gen_checked(genfn, prompt, refs, want_wh, tag, attempts=2):
    """Generate with RETRY-ONCE. Retries on an API exception, an empty result, and on a
    returned image that fails validation - image models fail transiently often enough that
    a single retry converts most failures into a good render, and a validated-empty image
    is worse than a retry because it silently poisons the character reference."""
    why = "unknown"
    for i in range(max(1, int(attempts))):
        try:
            img = genfn(prompt, refs)
            ok, why = _validate_generated(img, want_wh)
            if ok:
                if i:
                    print(f"[MIRAGE_GeminiAutoRef] {tag}: retry {i} succeeded ({why})")
                return img
        except Exception as ex:
            why = f"{type(ex).__name__}: {ex}"
        print(f"[MIRAGE_GeminiAutoRef] {tag}: attempt {i + 1}/{attempts} rejected - {why}")
    raise RuntimeError(f"[MIRAGE_GeminiAutoRef] {tag} failed after {attempts} attempts: {why}")


def _strip_article(s):
    """'a loose flowing tunic' -> 'loose flowing tunic'. The GARMENT_POOL entries carry their
    own leading article because they are used mid-sentence elsewhere; in the descriptor they
    follow a colour word, where the article reads as a typo ('teal a loose flowing tunic')."""
    t = str(s).strip()
    for art in ("a ", "an ", "the "):
        if t.lower().startswith(art):
            return t[len(art):]
    return t


def _attribute_descriptor(noun, pronoun, age, race, skin, hair_color, hair_style, build,
                          height_class, style, accessory, cloth_a, cloth_b, upper_g, lower_g,
                          design_u, design_l, locked_fit):
    """THE canonical subject specification, injected VERBATIM into BOTH image prompts.

    This is the cross-reference-consistency lever: the reference sheet and the full-body
    shot previously described the person in two separately-worded prose paragraphs, so each
    naming of an attribute was an opportunity to drift - and WanAnimate is driven by BOTH
    images, so a drifted pair yields a character that changes between them. One shared,
    field-per-line block means the two prompts cannot disagree by construction."""
    up, lo = _strip_article(upper_g), _strip_article(lower_g) if lower_g else ""
    if lower_g:
        garments = (f"- upper garment: {cloth_a} {up}, patterned with {design_u}\n"
                    f"- lower garment: {cloth_b} {lo}, patterned with {design_l}\n")
    else:
        garments = (f"- garment (ONE PIECE, no separate lower garment): {cloth_a} {up}, "
                    f"patterned with {design_u}, accented with {cloth_b} {design_l} detailing\n")
    return (
        "SUBJECT SPECIFICATION - one single individual, identical in every image and every "
        "panel; follow every line exactly and never substitute:\n"
        f"- gender: {noun}; unmistakably {noun}-presenting in face, jawline, build and body "
        f"proportions, and {pronoun} must never read as androgynous, ambiguous, or as the other "
        f"gender in any panel\n"
        f"- age: {age}\n"
        f"- ethnicity: {race}\n"
        f"- skin: {skin}\n"
        f"- hair: {hair_color} {hair_style}\n"
        f"- body build: {build}\n"
        f"- height class: {height_class} for an adult (overall body proportions only)\n"
        f"- overall styling: {style}\n"
        f"- accessory: {accessory}\n"
        + garments +
        f"- garment fit: {locked_fit} - roomy, generously draped, plenty of slack; never tight, "
        f"never fitted, never form-fitting, never bodycon\n"
        f"- clothing colours: strictly {cloth_a} and {cloth_b} and no other colour, rendered "
        f"fully saturated and colour-accurate exactly as named\n"
        f"- clothing pattern: clearly visible and boldly rendered; never plain, never flat, "
        f"never an undecorated block of colour\n"
    )


class MIRAGE_GeminiAutoRef:
    """Grey silhouettes -> Gemini-generated PRIMARY and SECONDARY reference images.

    Flow (one instance per character):
      1. detect gender + clothing STRUCTURE from the silhouettes (Gemini VLM, same
         detector as MIRAGE_AutoRefPrompt);
      2. roll a DIVERSE appearance from `seed` (race / skin / hair / colours / style)
         so every run is a brand-new person, while gender + garment cut stay locked;
      3. Gemini's IMAGE model renders the PRIMARY full-body front reference;
      4. the PRIMARY image is fed BACK to Gemini with a "same exact person, same
         clothing" instruction to render the SECONDARY view - so primary & secondary
         are guaranteed to be the SAME identity + outfit (the manual Nano-Banana
         consistency trick, automated).

    Outputs two IMAGEs ready to drive the character (e.g. WanAnimate) refs.

    QUALITY / SAFETY LEVERS (2026-07-23):
      * ONE canonical subject specification (_attribute_descriptor) is injected VERBATIM
        into BOTH image prompts, so the sheet and the full-body shot cannot drift apart - 
        WanAnimate is driven by both, so a drifted pair is a character that changes;
      * HARD_CONSTRAINTS + FULLBODY_FRAMING are likewise shared verbatim (photoreal, one
        identity, plain grey seamless backdrop, no props/text/watermark, head-to-toe with
        nothing cropped, neutral relaxed A-pose);
      * every generation is VALIDATED (non-blank, min size, requested aspect) and RETRIED
        ONCE before it is allowed downstream;
      * a §2 EGRESS GUARD verifies at runtime that the frames leaving for the API really
        are filled body masks, and the `info` output reports exactly what was sent.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "silhouette_frames": ("IMAGE",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                             "control_after_generate": True}),
            "num_frames": ("INT", {"default": 5, "min": 1, "max": 8}),
            "gender_override": (["auto", "male", "female"],),
            "include_fullbody": ("BOOLEAN", {"default": True}),
            "silhouette_refs": ("INT", {"default": 1, "min": 0, "max": 3}),
            # provider for the IMAGE model. openai=gpt-image-1 (reference-consistent,
            # pay-as-you-go). gemini=gemini-3-pro-image (needs a billed Gemini tier).
            "provider": (["openai", "gemini"], {"default": "openai"}),
            "image_model": ("STRING", {"default": "gpt-image-1"}),
            # detection VLM (cheap): gpt-4o-mini (openai) or gemini-2.5-flash (gemini).
            "vlm_model": ("STRING", {"default": "gpt-4o-mini"}),
            # OpenAI cost knobs (Gemini ignores). 1024x1536 ~ portrait like 1.png.
            # quality defaults to "medium" (gpt-image-1 @1024x1536: low ~$0.016/img,
            # medium ~$0.063/img, high ~$0.25/img): the refs drive the whole render,
            # so low-quality refs are false economy - drop to "low" for smoke tests.
            # Saved workflows keep their own stored value; this only affects new nodes.
            "image_size": (["1024x1536", "1024x1024", "1536x1024", "auto"], {"default": "1024x1536"}),
            "image_quality": (["low", "medium", "high", "auto"], {"default": "medium"}),
            "api_key": ("STRING", {"default": ""}),
            # PER-PERSON DIVERGENCE: this character's 0-based index and how many
            # characters share the scene. Person 0 and Person 1 draw race / skin /
            # hair-colour / clothing-colour / style from DISJOINT slices of each
            # pool, so no two characters in one scene look like the same person.
            "person_index": ("INT", {"default": 0, "min": 0, "max": 7}),
            "num_people": ("INT", {"default": 2, "min": 1, "max": 8}),
        },
        "optional": {
            "background_plate": ("IMAGE",),          # local-only lighting cue
            # §2 egress guard. OPTIONAL and APPENDED AT THE END on purpose: ComfyUI restores
            # widgets_values POSITIONALLY, so a saved workflow (V9/V9_CLOUD_ONLY store the
            # pre-existing widget list for these nodes) still loads and simply leaves this at
            # its default, and API prompts that omit it fall back to run()'s keyword default.
            # Never insert or reorder above this line.
            "egress_guard": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("reference_collage", "full_body", "info")
    FUNCTION = "run"
    CATEGORY = "MIRAGE"

    def run(self, silhouette_frames, seed, num_frames, gender_override, include_fullbody,
            silhouette_refs, provider, image_model, vlm_model, image_size, image_quality,
            api_key, person_index=0, num_people=2, background_plate=None, egress_guard=True):
        # API KEY: env var is the intended source. The api_key widget is kept (removing it would
        # shift every later widget's positional value in saved workflows) but a pasted key gets
        # SAVED INTO THE WORKFLOW JSON, so warn loudly if one is present.
        if (api_key or "").strip():
            print("[MIRAGE_GeminiAutoRef] WARNING: api_key widget is non-empty - the key will be stored "
                  "in any saved workflow JSON. Clear it and export the key in the server env "
                  "instead (OPENAI_API_KEY / GEMINI_API_KEY).")
        if provider == "openai":
            key = (api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
            if not key:
                raise RuntimeError("No OpenAI API key. Paste it into api_key or set OPENAI_API_KEY.")
            _detect = lambda fr: _analyze_openai(fr, vlm_model, key)
            _genimg = lambda prompt, refs: _openai_image(prompt, refs, image_model, key, image_size, image_quality)
        else:
            key = (api_key or "").strip() or os.environ.get("GEMINI_API_KEY", "").strip() \
                or os.environ.get("GOOGLE_API_KEY", "").strip()
            if not key:
                raise RuntimeError("No Gemini API key. Paste it into api_key or set GEMINI_API_KEY.")
            _detect = lambda fr: _analyze(fr, vlm_model, key)
            _genimg = lambda prompt, refs: _gemini_image(prompt, refs, image_model, key)

        frames, fhash = _pick_frames(silhouette_frames, int(num_frames))

        # 0) §2 EGRESS GUARD - these exact frames are what the VLM and the image model receive,
        #    so verify HERE, before the first API call, that they really are filled body masks.
        egress = _egress_check(frames, strict=bool(egress_guard))

        # 1) detect gender + clothing structure (cached per silhouette hash + provider)
        a = dict(_ATTR_DEFAULTS)
        ckey = f"{provider}:{vlm_model}:{fhash}"
        try:
            if ckey not in _ANALYSIS_CACHE:
                _ANALYSIS_CACHE[ckey] = _detect(frames)
            a = _ANALYSIS_CACHE[ckey]
        except Exception as ex:
            print("[MIRAGE_GeminiAutoRef] VLM detect failed, using defaults:", ex)
        gender, conf = a["gender"], a["gender_confidence"]
        if gender_override in ("male", "female"):
            gender, conf = gender_override, 1.0

        # 2) diversity roll (seeded) - locked: gender + garment cut; varied: everything else.
        #    Each pool is first narrowed to THIS person's disjoint slice (_bucket) so
        #    Person 0 and Person 1 diverge in race, skin, hair colour, clothing colour
        #    and style by construction; the seed then randomises within that slice.
        rng = random.Random(int(seed) ^ 0x5DEECE66D)
        pidx, ppl = int(person_index), max(1, int(num_people))
        race = rng.choice(_bucket(RACE_POOL, pidx, ppl))
        skin = rng.choice(_bucket(SKIN_POOL, pidx, ppl))
        hair_color = rng.choice(_bucket(HAIR_COLOR_POOL, pidx, ppl))
        cloth_pool = _bucket(CLOTH_COLOR_POOL, pidx, ppl)
        cloth_a = rng.choice(cloth_pool)
        cloth_b = rng.choice([c for c in cloth_pool if c != cloth_a] or [cloth_a])
        # DESIGN: strided (not contiguous) - a contiguous slice confined P0 to the plain
        # half of the pool. Two INDEPENDENT designs now (upper vs lower) so an outfit
        # reads as designed rather than one flat pattern everywhere.
        design_pool = _bucket_strided(CLOTH_DESIGN_POOL, pidx, ppl)
        design_u = rng.choice(design_pool)
        design_l = rng.choice([d for d in design_pool if d != design_u] or [design_u])
        style = rng.choice(_bucket(STYLE_FLAVOR_POOL, pidx, ppl))
        silhouette = rng.choice(LOOSE_SILHOUETTE_POOL); age = rng.choice(AGE_POOL)
        accessory = rng.choice(ACCESSORY_POOL)
        hair_style = rng.choice(HAIR_STYLE_MALE if gender == "male" else HAIR_STYLE_FEMALE)
        # GARMENT TYPE now comes from the culturally-diverse LOOSE pool rather than being
        # locked to the VLM's read of the real person's cut (which made every character
        # wear the same generic top+trousers). Safe because _loosen() already overrides the
        # detected fit to loose..baggy regardless, and the silhouette_refs still steer the
        # drape/proportions. Roomier garments also COVER MORE plate (halo-safe).
        upper_g, lower_g = rng.choice(_bucket_strided(
            GARMENT_POOL_MALE if gender == "male" else GARMENT_POOL_FEMALE, pidx, ppl))
        # Baggy/loose is now the TARGET (not merely "slightly loose"): push the
        # detected fit up to two steps looser, floored at "loose", capped at "baggy".
        locked_fit = _loosen(a.get("garment_fit", "regular"), steps=2, floor="loose", ceil="baggy")
        # gendered NOUNS - image models honour "man"/"woman" far better than "male person"
        noun = "man" if gender == "male" else "woman"
        pronoun = "he" if gender == "male" else "she"
        # one clause describing the outfit, handling one-piece garments (lower_g is None)
        if lower_g:
            outfit = (f"a {cloth_a} {upper_g} with {design_u}, and {cloth_b} {lower_g} "
                      f"with {design_l}")
            outfit_short = f"the {cloth_a} {upper_g} and the {cloth_b} {lower_g}"
        else:
            outfit = (f"a {cloth_a} {upper_g} with {design_u}, accented with {cloth_b} "
                      f"{design_l} detailing")
            outfit_short = f"the {cloth_a} {upper_g}"

        light_clause = ""
        if background_plate is not None:
            try:
                light_clause, _ = _analyze_bg_lighting(background_plate)
            except Exception as ex:
                print("[MIRAGE_GeminiAutoRef] bg lighting analysis failed:", ex)

        sil_refs = frames[:int(silhouette_refs)] if int(silhouette_refs) > 0 else []
        sil_role = (" The extra silhouette frame(s) show ONLY the person's filled body outline - copy the "
                    "OUTFIT SILHOUETTE, garment drape and body proportions from them; they contain no face, "
                    "skin, hair or colour, so invent those freely from the description above.") if sil_refs else ""

        # THE canonical subject spec - built ONCE and injected VERBATIM into both prompts below,
        # so the reference sheet and the full-body shot cannot describe two different people.
        descriptor = _attribute_descriptor(
            noun, pronoun, age, race, skin, hair_color, hair_style, a["build"],
            a.get("height_class", "average"), style, accessory, cloth_a, cloth_b,
            upper_g, lower_g, design_u, design_l, locked_fit)

        # requested output geometry, used to validate what the API hands back
        want_wh = None
        if provider == "openai" and image_size and image_size != "auto" and "x" in image_size:
            try:
                want_wh = tuple(int(v) for v in image_size.split("x"))
            except Exception:
                want_wh = None

        # 3) REFERENCE-SHEET COLLAGE - one image, so identity is consistent by
        #    construction (matches the input 1.png / 2.png detail sheets).
        collage_prompt = (
            f"Create a single photorealistic CHARACTER REFERENCE SHEET - a tidy grid COLLAGE of "
            f"close-up reference crops, EVERY panel showing the SAME ONE adult {noun}.\n\n"
            f"{descriptor}\n"
            f"CONSISTENCY IS THE POINT OF THIS SHEET: every panel is the IDENTICAL {noun} - the same face, "
            f"the same {hair_color} {hair_style}, the same {skin}, and the same outfit with the same "
            f"{design_u} and {design_l} in the same {cloth_a}/{cloth_b} colours. The FULL-BODY panel must match "
            f"the close-up panels EXACTLY in face, hair, garment cut, pattern and colour - a viewer must be able "
            f"to tell at a glance that every panel is one single individual, not a set of similar people. "
            f"{sil_role}\n"
            f"The panels are: "
            f"an extreme macro close-up of one eye (iris, eyelashes, visible skin pores); a side profile of "
            f"the face and head; a back view of the head showing the hair; a close-up of the chest and collar "
            f"showing the garment pattern; one or two close-ups of a hand resting on a surface; a FULL-LENGTH "
            f"full-body standing panel showing {FULLBODY_FRAMING}; a three-quarter face close-up with a "
            f"soft neutral expression; and a close-up of the clothing fabric showing its weave and its "
            f"{design_u}. Arrange them as a clean professional photographic grid"
            + (f". {light_clause}" if light_clause else "") + ".\n\n"
            f"{HARD_CONSTRAINTS}"
        )
        collage_pil = _gen_checked(_genimg, collage_prompt, sil_refs, want_wh, "reference sheet")

        # 4) FULL-BODY - clean standing shot of the SAME person, generated FROM the
        #    collage so identity + outfit cannot drift (matches the 1_full.png ref).
        if include_fullbody:
            # Restate the FULL identity + outfit spec here. The attached collage is the
            # primary anchor, but the earlier prompt named almost none of these attributes,
            # which let the full-body drift away from the sheet (different face/garment).
            # Naming them again pins the two together.
            fb_prompt = (
                f"The attached image is a character reference sheet of ONE adult {noun}. Generate a single "
                f"clean full-body standing studio photograph of the EXACT SAME PERSON shown in that sheet - "
                f"the SAME {noun}, not a similar-looking different person. The sheet is the visual anchor; "
                f"the specification below is the same one it was built from and must be matched exactly, "
                f"attribute for attribute and garment for garment.\n\n"
                f"{descriptor}\n"
                f"FRAMING AND POSE: {FULLBODY_FRAMING}.\n\n"
                f"{HARD_CONSTRAINTS}"
            )
            fb_pil = _gen_checked(_genimg, fb_prompt, [collage_pil], want_wh, "full body")
        else:
            fb_pil = collage_pil

        info = (f"P{pidx}/{ppl} gender={gender}({conf:.2f}) build={a.get('build')} "
                f"height={a.get('height_class')} fit={a.get('garment_fit')}->{locked_fit} | "
                f"roll: {race}/{skin}/{hair_color}/{hair_style} | "
                f"garment={upper_g}" + (f" + {lower_g}" if lower_g else " (one-piece)") +
                f" | cloth={cloth_a}/{design_u} + {cloth_b}/{design_l} | {style} | "
                f"collage+{'fullbody' if include_fullbody else 'collage-only'} sil_refs={len(sil_refs)} | "
                # §2 accountability: exactly what crossed the trust boundary, and the measured
                # evidence that it was mask-only. Nothing else in this node reaches an API.
                f"EGRESS: {len(frames)} silhouette frames -> VLM, {len(sil_refs)} -> image model, "
                f"{'1 generated collage -> image model' if include_fullbody else 'no 2nd call'}; "
                f"guard={'on' if egress_guard else 'OFF'} [{egress}]")
        print("[MIRAGE_GeminiAutoRef]", info)
        return (_pil_to_tensor(collage_pil), _pil_to_tensor(fb_pil), info)

    @classmethod
    def IS_CHANGED(cls, silhouette_frames, seed, num_frames, gender_override, include_fullbody,
                   silhouette_refs, provider, image_model, vlm_model, image_size, image_quality,
                   api_key, person_index=0, num_people=2, background_plate=None, egress_guard=True):
        # re-run every queue so seed's control_after_generate=randomize yields a new person
        return str(random.random())


NODE_CLASS_MAPPINGS = {
    "MIRAGE_AutoRefPrompt": MIRAGE_AutoRefPrompt,
    "MIRAGE_GeminiAutoRef": MIRAGE_GeminiAutoRef,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MIRAGE_AutoRefPrompt": "MIRAGE AutoReference Prompt",
    "MIRAGE_GeminiAutoRef": "MIRAGE Gemini Auto-Reference (primary+secondary)",
}

# ---------------------------------------------------------------------------------------------
# PRE-RENAME COMPATIBILITY (2026-08-29). A saved ComfyUI graph stores each node's registry key in
# its `class_type` field, so every workflow exported before the rename says "SITARA...". Loading
# one against a registry that only knows "MIRAGE..." fails with an unhelpful missing-node error.
# Each node is therefore registered a SECOND time under its old key. The alias maps to the SAME
# class object, so the two names cannot drift apart, and new graphs save under the new name.
_LEGACY = {k.replace("MIRAGE", "SITARA", 1): v
           for k, v in NODE_CLASS_MAPPINGS.items() if k.startswith("MIRAGE")}
NODE_CLASS_MAPPINGS.update(_LEGACY)
NODE_DISPLAY_NAME_MAPPINGS.update({
    k: NODE_DISPLAY_NAME_MAPPINGS.get(k.replace("SITARA", "MIRAGE", 1), k) + " (legacy name)"
    for k in _LEGACY
})
