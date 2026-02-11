import pandas as pd
import re
import unicodedata
import numpy as np

# ---------- CONFIG ----------
IN_PATH = "greek_sentences.csv"
OUT_PATH = "greek_sentences_balanced_noN1eqN2.csv"

# Extra profession nouns to use if pools need padding (gender-marked forms)
CANDIDATES = [
    ("ερευνητής","ερευνητές","ερευνήτρια","ερευνήτριες","researcher"),
    ("διευθυντής","διευθυντές","διευθύντρια","διευθύντριες","manager/director"),
    ("προγραμματιστής","προγραμματιστές","προγραμματίστρια","προγραμματίστριες","programmer"),
    ("σκηνοθέτης","σκηνοθέτες","σκηνοθέτρια","σκηνοθέτριες","director (film/theatre)"),
    ("τραγουδιστής","τραγουδιστές","τραγουδίστρια","τραγουδίστριες","singer"),
    ("νοσηλευτής","νοσηλευτές","νοσηλεύτρια","νοσηλεύτριες","nurse"),
    ("τεχνίτης","τεχνίτες","τεχνίτρια","τεχνίτριες","technician/craftsperson"),
    ("ζαχαροπλάστης","ζαχαροπλάστες","ζαχαροπλάστρια","ζαχαροπλάστριες","pastry chef"),
    ("σερβιτόρος","σερβιτόροι","σερβιτόρα","σερβιτόρες","waiter/waitress"),
]

# ---------- HELPERS ----------
def norm(s):
    return unicodedata.normalize("NFC", str(s).strip())

def get_tokens(s):
    s = norm(s)
    s = re.sub(r"[?.!]", "", s)
    return s.split()

def extract_np(sentence, position, order):
    toks = get_tokens(sentence)
    if position == "N1":
        return " ".join(toks[:2])  # Article + noun
    idx = toks.index("που")
    after = toks[idx + 1:]
    return " ".join(after[1:3]) if order == "VSO" else " ".join(after[0:2])

def extract_v1(sentence, order):
    toks = get_tokens(sentence)
    idx = toks.index("που")
    after = toks[idx + 1:]
    return after[0] if order == "VSO" else after[2]

def extract_v2(sentence):
    toks = get_tokens(sentence)
    return toks[-1]

def noun_token(np_str):
    toks = norm(np_str).split()
    return toks[1] if len(toks) >= 2 else np_str

def cap_article(np_str):
    np_str = norm(np_str)
    if np_str.startswith("η "):  return "Η " + np_str[2:]
    if np_str.startswith("ο "):  return "Ο " + np_str[2:]
    if np_str.startswith("οι "): return "Οι " + np_str[3:]
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
    if num == "sing" and gen == "m": return ms
    if num == "sing" and gen == "f": return fs
    if num == "plur" and gen == "m": return mp
    return fp

# ---------- LOAD & EXTRACT ----------
df = pd.read_csv(IN_PATH)
df["N1_np"] = df.apply(lambda r: extract_np(r["Sentence"], "N1", r["order"]), axis=1)
df["N2_np"] = df.apply(lambda r: extract_np(r["Sentence"], "N2", r["order"]), axis=1)
df["V1"]    = df.apply(lambda r: extract_v1(r["Sentence"], r["order"]), axis=1)
df["V2"]    = df["Sentence"].apply(extract_v2)

# Pools from original stimuli
n1_pool, n2_pool = {}, {}
target_n1 = None
for num in ["sing", "plur"]:
    for gen in ["m", "f"]:
        n1_pool[(num, gen)] = sorted(df.loc[(df.N1_number==num)&(df.N1_gender==gen), "N1_np"].unique())
        n2_pool[(num, gen)] = sorted(df.loc[(df.N1_number==num)&(df.N2_gender==gen), "N2_np"].unique())

v1_pool = {
    "sing": sorted(df.loc[df.N1_number=="sing", "V1"].unique()),
    "plur": sorted(df.loc[df.N1_number=="plur", "V1"].unique()),
}
v2_pool = {
    "sing": sorted(df.loc[df.N1_number=="sing", "V2"].unique()),
    "plur": sorted(df.loc[df.N1_number=="plur", "V2"].unique()),
}

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
            vgrid = pd.MultiIndex.from_product([v1s, v2s], names=["V1", "V2"]).to_frame(index=False)
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
                        tmp["Sentence"] = tmp["N1_np"] + " που " + tmp["V1"] + " " + tmp["N2_np"] + " " + tmp["V2"] + "."
                    else:
                        tmp["Sentence"] = tmp["N1_np"] + " που " + tmp["N2_np"] + " " + tmp["V1"] + " " + tmp["V2"] + "."

                    tmp["Question"] = who + " " + (tmp["V1"] if qfor == "V1" else tmp["V2"]) + "?"
                    tmp["N2_cap"] = tmp["N2_np"].map(cap_article)

                    tmp["correct_response"] = np.where(tmp["question_for"]=="V1", tmp["N2_cap"], tmp["N1_np"])
                    tmp["false_response"]   = np.where(tmp["question_for"]=="V1", tmp["N1_np"], tmp["N2_cap"])

                    tmp["N1_equals_N2"] = (tmp["N1_np"].apply(noun_token) == tmp["N2_np"].apply(noun_token))

                    frames.append(tmp[[
                        "Sentence","Question","order","question_for","N1_number","N1_gender","N2_gender",
                        "correct_response","false_response","N1_np","N2_np","V1","V2","N1_equals_N2"
                    ]])

out = pd.concat(frames, ignore_index=True)
out.insert(0, "stim_id", np.arange(1, len(out) + 1))

# Checks: no identical N1/N2 nouns and perfectly balanced 32 cells
assert out["N1_equals_N2"].sum() == 0
cell_counts = out.groupby(["order","question_for","N1_number","N1_gender","N2_gender"]).size()
assert int(cell_counts.min()) == int(cell_counts.max())

out.to_csv(OUT_PATH, index=False)
print("Wrote:", OUT_PATH)
print("Total rows:", len(out))
print("Rows per condition cell:", int(cell_counts.min()))
