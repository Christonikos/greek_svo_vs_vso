import unicodedata

import numpy as np
import pandas as pd
from lexicon import words

# ---------- CONFIG ----------
OUT_PATH = "greek_sentences_balanced_noN1eqN2.csv"

# Extra profession nouns to use if pools need padding (gender-marked forms)
CANDIDATES = [
    ("ερευνητής", "ερευνητές", "ερευνήτρια", "ερευνήτριες", "researcher"),
    ("διευθυντής", "διευθυντές", "διευθύντρια", "διευθύντριες", "manager/director"),
    (
        "προγραμματιστής",
        "προγραμματιστές",
        "προγραμματίστρια",
        "προγραμματίστριες",
        "programmer",
    ),
    (
        "σκηνοθέτης",
        "σκηνοθέτες",
        "σκηνοθέτρια",
        "σκηνοθέτριες",
        "director (film/theatre)",
    ),
    ("τραγουδιστής", "τραγουδιστές", "τραγουδίστρια", "τραγουδίστριες", "singer"),
    ("νοσηλευτής", "νοσηλευτές", "νοσηλεύτρια", "νοσηλεύτριες", "nurse"),
    ("τεχνίτης", "τεχνίτες", "τεχνίτρια", "τεχνίτριες", "technician/craftsperson"),
    (
        "ζαχαροπλάστης",
        "ζαχαροπλάστες",
        "ζαχαροπλάστρια",
        "ζαχαροπλάστριες",
        "pastry chef",
    ),
    ("σερβιτόρος", "σερβιτόροι", "σερβιτόρα", "σερβιτόρες", "waiter/waitress"),
]


# ---------- HELPERS ----------
def norm(s):
    return unicodedata.normalize("NFC", str(s).strip())


def noun_token(np_str):
    toks = norm(np_str).split()
    return toks[1] if len(toks) >= 2 else np_str


def cap_article(np_str):
    np_str = norm(np_str)
    if np_str.startswith("η "):
        return "Η " + np_str[2:]
    if np_str.startswith("ο "):
        return "Ο " + np_str[2:]
    if np_str.startswith("οι "):
        return "Οι " + np_str[3:]
    if np_str.startswith("το "):
        return "Το " + np_str[3:]
    return np_str


def make_np(noun, num, gen, position):
    # position: 'N1' => sentence-initial article capitalized; 'N2' => lowercase
    if num == "sing" and gen == "m":
        art = "Ο" if position == "N1" else "ο"
    elif num == "sing" and gen == "f":
        art = "Η" if position == "N1" else "η"
    else:  # plur
        art = "Οι" if position == "N1" else "οι"
    return f"{art} {noun}"


def choose_form(ms, mp, fs, fp, num, gen):
    if num == "sing" and gen == "m":
        return ms
    if num == "sing" and gen == "f":
        return fs
    if num == "plur" and gen == "m":
        return mp
    return fp


# ---------- BUILD POOLS FROM LEXICON ----------
n1_pool, n2_pool = {}, {}
v1_pool, v2_pool = {}, {}
target_n1 = None
for num in ["sing", "plur"]:
    # Verbs directly from lexicon
    v1_pool[num] = sorted(set(map(norm, words["verbs"]["tran"][num])))
    v2_pool[num] = sorted(set(map(norm, words["verbs"]["intr"][num])))

    for gen in ["m", "f"]:
        # Determiner: generator uses words["det"][gender][number][0]
        det = norm(words["det"][gen][num][0])

        # N1: capitalized determiner in the NP
        det_N1 = det.capitalize()
        # N2: lowercase determiner in the NP (as in generator)
        det_N2 = det  # already lowercased in the lexicon

        nouns = [norm(x) for x in words["humans"][gen][num]]

        n1_pool[(num, gen)] = sorted({f"{det_N1} {n}" for n in nouns})
        n2_pool[(num, gen)] = sorted({f"{det_N2} {n}" for n in nouns})

# ---------- BALANCE NOUN POOLS ----------
target_n1 = max(len(v) for v in n1_pool.values())
target_n2 = max(len(v) for v in n2_pool.values())

n1_sets = {k: set(v) for k, v in n1_pool.items()}
n2_sets = {k: set(v) for k, v in n2_pool.items()}


def fill(pool_sets, pool_dict, position, target, num, gen):
    key = (num, gen)
    while len(pool_dict[key]) < target:
        did = False
        for ms, mp, fs, fp, _gloss in CANDIDATES:
            noun = choose_form(ms, mp, fs, fp, num, gen)
            np_form = make_np(noun, num, gen, position)
            if np_form not in pool_sets[key]:
                pool_sets[key].add(np_form)
                pool_dict[key].append(np_form)
                did = True
                break
        if not did:
            raise RuntimeError("Not enough candidate nouns to balance pools.")
    pool_dict[key] = sorted(pool_dict[key])


for num in ["sing", "plur"]:
    for gen in ["m", "f"]:
        fill(n1_sets, n1_pool, "N1", target_n1, num, gen)
        fill(n2_sets, n2_pool, "N2", target_n2, num, gen)

# ---------- ENFORCE N1 != N2 (BY NOUN TOKEN) + KEEP CELLS BALANCED ----------
pair_sets = {}
pair_counts = []
for num in ["sing", "plur"]:
    for n1g in ["m", "f"]:
        for n2g in ["m", "f"]:
            n1s = n1_pool[(num, n1g)]
            n2s = n2_pool[(num, n2g)]
            allowed = []
            for a in n1s:
                ta = noun_token(a)
                for b in n2s:
                    if ta != noun_token(b):
                        allowed.append((a, b))
            allowed = sorted(allowed)
            pair_sets[(num, n1g, n2g)] = allowed
            pair_counts.append((num, n1g, n2g, len(allowed)))

target_pairs = min(c[-1] for c in pair_counts)  # makes all cells equal-sized

# Generate dataset with exactly target_pairs noun-pairs in every (num,n1g,n2g)
frames = []
for num in ["sing", "plur"]:
    who = "Ποιος" if num == "sing" else "Ποιοι"
    v1s = v1_pool[num]
    v2s = v2_pool[num]

    for n1g in ["m", "f"]:
        for n2g in ["m", "f"]:
            pairs = pair_sets[(num, n1g, n2g)][:target_pairs]
            base = pd.DataFrame(pairs, columns=["N1_np", "N2_np"])

            # Expand with all verb combinations (keeps verb balance intact)
            base["key"] = 1
            vgrid = pd.MultiIndex.from_product([v1s, v2s], names=["V1", "V2"]).to_frame(
                index=False
            )
            vgrid["key"] = 1
            base = base.merge(vgrid, on="key").drop(columns=["key"])

            for order in ["VSO", "SVO"]:
                for qfor in ["V1", "V2"]:
                    tmp = base.copy()
                    tmp["order"] = order
                    tmp["question_for"] = qfor
                    tmp["N1_number"] = num
                    tmp["N1_gender"] = n1g
                    tmp["N2_gender"] = n2g

                    if order == "VSO":
                        tmp["Sentence"] = (
                            tmp["N1_np"]
                            + " που "
                            + tmp["V1"]
                            + " "
                            + tmp["N2_np"]
                            + " "
                            + tmp["V2"]
                            + "."
                        )
                    else:
                        tmp["Sentence"] = (
                            tmp["N1_np"]
                            + " που "
                            + tmp["N2_np"]
                            + " "
                            + tmp["V1"]
                            + " "
                            + tmp["V2"]
                            + "."
                        )

                    tmp["Question"] = (
                        who + " " + (tmp["V1"] if qfor == "V1" else tmp["V2"]) + "?"
                    )
                    tmp["N2_cap"] = tmp["N2_np"].map(cap_article)

                    tmp["correct_response"] = np.where(
                        tmp["question_for"] == "V1", tmp["N2_cap"], tmp["N1_np"]
                    )
                    tmp["false_response"] = np.where(
                        tmp["question_for"] == "V1", tmp["N1_np"], tmp["N2_cap"]
                    )

                    tmp["N1_equals_N2"] = tmp["N1_np"].apply(noun_token) == tmp[
                        "N2_np"
                    ].apply(noun_token)

                    frames.append(
                        tmp[
                            [
                                "Sentence",
                                "Question",
                                "order",
                                "question_for",
                                "N1_number",
                                "N1_gender",
                                "N2_gender",
                                "correct_response",
                                "false_response",
                                "N1_np",
                                "N2_np",
                                "V1",
                                "V2",
                                "N1_equals_N2",
                            ]
                        ]
                    )

out = pd.concat(frames, ignore_index=True)
out.insert(0, "stim_id", np.arange(1, len(out) + 1))

# Checks: no identical N1/N2 nouns and perfectly balanced 32 cells
assert out["N1_equals_N2"].sum() == 0
cell_counts = out.groupby(
    ["order", "question_for", "N1_number", "N1_gender", "N2_gender"]
).size()
assert int(cell_counts.min()) == int(cell_counts.max())

out.to_csv(OUT_PATH, index=False)
print("Wrote:", OUT_PATH)
print("Total rows:", len(out))
print("Rows per condition cell:", int(cell_counts.min()))
